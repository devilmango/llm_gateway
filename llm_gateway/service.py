from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any
from urllib.parse import quote

import httpx
from opentelemetry import trace

from .cache import EmbeddingCache, RedisResponseCache, ResponseCache
from .circuit import ProviderCircuitBreaker
from .distributed import (
    RateLimitBackendUnavailable,
    RedisConcurrencyLimiter,
    RedisSlidingWindowLimiter,
)
from .limiting import RateLimitExceeded, SlidingWindowLimiter
from .observability import PROVIDER_RETRIES, gen_ai_provider_name
from .provider_health import ProviderHealth
from .providers import ProviderCapabilityError, ProviderClient, ProviderError
from .tool_validation import validate_tool_calls


class GatewayService:
    def __init__(self, config: dict[str, Any], http: Any, shared_redis: Any | None = None):
        self.config = config
        self.http = http
        self.providers = config["providers"]
        self.routes = config["routes"]
        self.client = ProviderClient(http, self.providers, float(config["server"].get("request_timeout_seconds", 45)))
        self.limiter = RedisSlidingWindowLimiter(shared_redis) if shared_redis else SlidingWindowLimiter()
        self.cache = RedisResponseCache(shared_redis) if shared_redis else ResponseCache()
        self.embedding_cache = EmbeddingCache(self.cache)
        self.max_concurrent_requests = int(config["server"].get("max_concurrent_requests", 100))
        self.shared_redis = shared_redis
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.application_limiter = self.limiter if shared_redis else SlidingWindowLimiter()
        breaker_config = config.get("circuit_breaker", {})
        self.breaker = ProviderCircuitBreaker(
            list(self.providers),
            failure_threshold=breaker_config.get("failure_threshold", 5),
            recovery_seconds=breaker_config.get("recovery_seconds", 30),
            redis=shared_redis,
        )
        self.provider_health = ProviderHealth(shared_redis)

    async def acquire_application_quota(self, application: str, limit: int) -> None:
        await self.application_limiter.acquire(application, limit)

    def budget_reservation(self, route_name: str, payload: dict[str, Any], policy: dict[str, Any],
                           tenant_policy: dict[str, Any] | None = None) -> float:
        max_input_bytes = int(policy.get("max_input_bytes", 32_768))
        input_bytes = sum(len((content if isinstance(content, str) else
                               json.dumps(content, ensure_ascii=False)).encode("utf-8"))
                          for message in payload["messages"]
                          for content in [message.get("content") or ""])
        if payload.get("tools") or payload.get("response_format"):
            input_bytes += len(json.dumps({"tools": payload.get("tools"),
                "response_format": payload.get("response_format")}, ensure_ascii=False).encode("utf-8"))
        if input_bytes > max_input_bytes:
            raise ValueError(f"Request exceeds application input limit of {max_input_bytes} bytes")
        max_completion_tokens = int(policy.get("max_completion_tokens", 1_024))
        requested_tokens = payload.get("max_tokens", max_completion_tokens)
        if requested_tokens > max_completion_tokens:
            raise ValueError(f"max_tokens exceeds application limit of {max_completion_tokens}")
        payload["max_tokens"] = requested_tokens
        candidates = self.route_candidates(route_name, tenant_policy=tenant_policy)
        # UTF-8 bytes conservatively bound input token count; use the most expensive
        # candidate because a fallback can serve the request instead of the primary.
        prices = [self.providers[c["provider"]].get("models", {}).get(c["model"], {}) for c in candidates]
        input_rate = max((float(p.get("input_cost_per_1m", 0)) for p in prices), default=0)
        output_rate = max((float(p.get("output_cost_per_1m", 0)) for p in prices), default=0)
        return (input_bytes * input_rate + requested_tokens * output_rate) / 1_000_000

    def route_candidates(self, route_name: str, request_id: str | None = None,
                         tenant_policy: dict[str, Any] | None = None,
                         include_canary: bool = True,
                         canary_enabled: bool | None = None) -> list[dict[str, str]]:
        if route_name not in self.routes:
            raise KeyError(route_name)
        route = self.routes[route_name]
        candidates = [{"provider": route["provider"], "model": route["model"]}]
        candidates.extend(route.get("fallbacks", []))
        for candidate in candidates:
            provider = candidate.get("provider")
            if provider not in self.providers:
                raise ValueError(f"Route {route_name!r} refers to unknown provider {provider!r}")
        canary = route.get("canary") if include_canary else None
        if canary:
            percentage = float(canary.get("traffic_percent", 0))
            if not 0 <= percentage <= 100:
                raise ValueError(f"Route {route_name!r} canary traffic_percent must be between 0 and 100")
            canary_candidate = {"provider": canary.get("provider"), "model": canary.get("model")}
            if canary_candidate["provider"] not in self.providers or not canary_candidate["model"]:
                raise ValueError(f"Route {route_name!r} canary must reference a configured provider and model")
            # Without a request id return the union so budget reservations account for the
            # most expensive possible target. Live requests use a stable sample per request.
            sampled = (canary_enabled if canary_enabled is not None else
                percentage > 0 and (request_id is None or
                int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100 < percentage))
            if sampled and self._tenant_allows(canary_candidate, tenant_policy):
                candidates = [canary_candidate, *candidates]
        candidates = [candidate for candidate in candidates if self._tenant_allows(candidate, tenant_policy)]
        if not candidates:
            raise ValueError("Tenant data policy does not permit any provider/model candidate for this route")
        return candidates

    async def _feature_flag_canary(self, route_name: str, request_id: str | None,
                                   context: dict[str, Any] | None) -> tuple[bool | None, str | None]:
        """Evaluate an optional OFREP boolean flag; failures safely leave canary disabled."""
        flag_config = self.config.get("routing", {}).get("feature_flags", {})
        if not flag_config.get("enabled"):
            return None, None
        route = self.routes.get(route_name, {})
        if not route.get("canary"):
            return None, None
        endpoint = str(flag_config.get("base_url", "")).rstrip("/")
        key = str(flag_config.get("flag_key", "")).strip()
        if not endpoint or not key:
            return False, "feature_flag_misconfigured"
        url = endpoint if "/ofrep/v1/evaluate/flags/" in endpoint else (
            f"{endpoint}/ofrep/v1/evaluate/flags/{quote(key, safe='')}")
        app_context = {k: v for k, v in (context or {}).items()
                       if k in {"application", "tenant_id"} and isinstance(v, (str, int, float, bool))}
        targeting_key = str(app_context.get("tenant_id") or app_context.get("application") or "llm-gateway")
        evaluation_context = {"targetingKey": targeting_key, **app_context}
        headers: dict[str, str] = {}
        token_env = flag_config.get("bearer_token_env")
        token = os.getenv(str(token_env)) if token_env else None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = await self.http.post(url, json={"context": evaluation_context}, headers=headers,
                timeout=float(flag_config.get("timeout_seconds", 0.5)))
            response.raise_for_status()
            result = response.json()
            value = result.get("value")
            if not isinstance(value, bool):
                return False, "feature_flag_invalid_value"
            reason = str(result.get("reason") or "")
            safe_reason = reason if re.fullmatch(r"[A-Za-z0-9_]{1,64}", reason) else (
                "feature_flag_enabled" if value else "feature_flag_disabled")
            return value, safe_reason
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
            # Do not include provider response bodies or credentials in logs/decisions.
            return False, f"feature_flag_error:{type(exc).__name__}"

    def _tenant_allows(self, candidate: dict[str, str], tenant_policy: dict[str, Any] | None) -> bool:
        if not tenant_policy:
            return True
        provider_name = candidate["provider"]
        provider = self.providers[provider_name]
        allowed_providers = tenant_policy.get("allowed_providers")
        if allowed_providers is not None and provider_name not in allowed_providers:
            return False
        allowed_models = tenant_policy.get("allowed_models")
        if allowed_models is not None and candidate["model"] not in allowed_models:
            return False
        allowed_regions = tenant_policy.get("allowed_regions")
        return allowed_regions is None or provider.get("data_region", "global") in allowed_regions

    async def health_snapshot(self) -> dict[str, dict[str, float | int]]:
        return await self.provider_health.snapshot(list(self.providers))

    @staticmethod
    def cache_safe_payload(payload: dict[str, Any]) -> bool:
        if payload.get("tools") or payload.get("modalities") or payload.get("audio"):
            return False
        for message in payload.get("messages", []):
            if message.get("tool_calls"):
                return False
            content = message.get("content")
            if isinstance(content, list) and any(part.get("type") in {"image_url", "input_audio"} for part in content):
                return False
        return True

    async def _order_candidates(self, candidates: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.config.get("routing", {}).get("strategy") != "latency_aware" or len(candidates) < 2:
            return candidates
        health = await self.health_snapshot()
        # Unseen providers retain configured order; observed failure rate is penalized
        # so one fast failure does not attract traffic over a healthy provider.
        indexed = list(enumerate(candidates))
        def score(item: tuple[int, dict[str, str]]) -> tuple[float, int]:
            index, candidate = item
            sample = health.get(candidate["provider"])
            if not sample:
                return (float("inf"), index)
            return (float(sample.get("ewma_latency_ms", 1e9)) *
                    (1.0 + float(sample.get("failure_rate", 0.0)) * 4.0), index)
        return [candidate for _, candidate in sorted(indexed, key=score)]

    @staticmethod
    def _decision(candidate: dict[str, str], decision: str, reason: str,
                  attempt_count: int = 0) -> dict[str, Any]:
        return {"provider": candidate["provider"], "model": candidate["model"],
                "decision": decision, "reason": reason, "attempt_count": attempt_count}

    def _selection_reason(self, candidate_index: int, candidate: dict[str, str],
                          preferred: tuple[str, str], canary_config: dict[str, Any] | None,
                          canary_selected: bool, previous_failure: str | None) -> str:
        if canary_selected and canary_config and (candidate["provider"], candidate["model"]) == (
                canary_config.get("provider"), canary_config.get("model")):
            return "canary_sampled"
        if candidate_index > 0:
            return f"prior_candidate_failed:{previous_failure or 'provider_error'}"
        if self.config.get("routing", {}).get("strategy") == "latency_aware":
            return "latency_aware_priority"
        if (candidate["provider"], candidate["model"]) == preferred:
            return "route_primary"
        return "preferred_candidate_filtered_by_policy"

    def routing_policy_exclusions(self, route_name: str, request_id: str | None = None,
                                  tenant_policy: dict[str, Any] | None = None,
                                  canary_enabled: bool | None = None) -> list[dict[str, Any]]:
        route = self.routes.get(route_name)
        if route is None:
            return []
        candidates = [{"provider": route["provider"], "model": route["model"]},
                      *route.get("fallbacks", [])]
        canary = route.get("canary")
        sampled_canary = (canary_enabled if canary_enabled is not None else
            bool(canary and request_id is not None and
                int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100 <
                float(canary.get("traffic_percent", 0))))
        if sampled_canary:
            candidates.insert(0, {"provider": canary["provider"], "model": canary["model"]})
        excluded = []
        seen = set()
        for candidate in candidates:
            key = (candidate["provider"], candidate["model"])
            if key not in seen and not self._tenant_allows(candidate, tenant_policy):
                excluded.append(self._decision(candidate, "skipped", "tenant_policy_excluded"))
            seen.add(key)
        return excluded

    @staticmethod
    def _provider_failure_reason(error: ProviderError) -> str:
        if isinstance(error, ProviderCapabilityError):
            return "unsupported_capability"
        if error.status_code == 429:
            return "upstream_rate_limited"
        if error.status_code is not None and error.status_code >= 500:
            return "upstream_server_error"
        if error.retryable:
            return "retryable_provider_error"
        return "provider_error"

    @classmethod
    def _skip_remaining(cls, decisions: list[dict[str, Any]], candidates: list[dict[str, str]],
                        start: int, reason: str) -> None:
        decisions.extend(cls._decision(candidate, "skipped", reason) for candidate in candidates[start:])

    async def complete(self, route_name: str, payload: dict[str, Any], request_id: str | None = None,
                       tenant_policy: dict[str, Any] | None = None,
                       evaluation_context: dict[str, Any] | None = None) -> dict[str, Any]:
        route = self.routes[route_name]
        preferred = (route["provider"], route["model"])
        feature_canary, feature_reason = await self._feature_flag_canary(route_name, request_id, evaluation_context)
        candidates = await self._order_candidates(self.route_candidates(route_name, request_id, tenant_policy,
            canary_enabled=feature_canary))
        policy_exclusions = self.routing_policy_exclusions(route_name, request_id, tenant_policy,
                                                           canary_enabled=feature_canary)
        canary_config = route.get("canary")
        canary_candidate = ({"provider": canary_config.get("provider"), "model": canary_config.get("model")}
                            if canary_config else None)
        static_canary_selected = bool(canary_config and request_id is not None and
            int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100 <
            float(canary_config.get("traffic_percent", 0)))
        canary_selected = bool(canary_config and canary_candidate in candidates and
                               (feature_canary is True or
                                (feature_canary is None and static_canary_selected)))
        if feature_reason:
            policy_exclusions.append(self._decision(
                canary_candidate,
                "selected" if canary_selected else "skipped", feature_reason))
        if canary_selected:
            candidates = self._prioritize_canary(candidates, canary_config)
        cache_cfg = route.get("cache", {})
        cache_key = self._cache_key(payload, preferred, candidates, tenant_policy)
        cache_safe = self.cache_safe_payload(payload) and not canary_selected
        if cache_cfg.get("enabled") and cache_safe:
            cached = await self.cache.get(cache_key)
            if cached:
                # A cache hit incurs no upstream token spend. The response still contains the
                # cached completion, while gateway usage represents billable upstream usage.
                cached["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                cached_decision = self._decision(
                    {"provider": cached["provider"], "model": cached.get("configured_model", cached["model"])},
                    "selected", "response_cache_hit")
                return {**cached, "cached": True, "fallback_used": False,
                        "routing_decisions": [cached_decision]}

        errors: list[str] = []
        routing_decisions: list[dict[str, Any]] = []
        previous_failure: str | None = None
        max_retries = int(self.config["server"].get("max_retries", 2))
        base_delay = float(self.config["server"].get("retry_base_delay_seconds", 0.25))
        async with self._concurrency():
            for candidate_index, candidate in enumerate(candidates):
                name, model = candidate["provider"], candidate["model"]
                provider_config = self.providers[name]
                limit = int(provider_config.get("requests_per_minute", 600))
                candidate_failure: str | None = None
                attempt_count = 0
                for attempt in range(max_retries + 1):
                    attempt_count = attempt + 1
                    started: float | None = None
                    with trace.get_tracer("llm_gateway").start_as_current_span("llm.gateway.provider_attempt") as span:
                        span.set_attribute("llm.gateway.provider", name)
                        span.set_attribute("llm.gateway.model", model)
                        span.set_attribute("gen_ai.operation.name", "chat")
                        span.set_attribute("gen_ai.provider.name", gen_ai_provider_name(provider_config))
                        span.set_attribute("gen_ai.request.model", model)
                        span.set_attribute("llm.gateway.attempt", attempt + 1)
                        span.set_attribute("llm.gateway.fallback_index", candidate_index)
                        try:
                            if not await self.breaker.allow(name):
                                errors.append(f"{name}: circuit open")
                                candidate_failure = "circuit_open"
                                break
                            await self.limiter.acquire(name, limit)
                            started = asyncio.get_running_loop().time()
                            result = await self.client.complete(name, model, payload, request_id=request_id)
                            span.set_attribute("gen_ai.response.model", str(result.get("model") or model))
                            usage = result.get("usage", {})
                            span.set_attribute("gen_ai.usage.input_tokens", int(usage.get("prompt_tokens", 0)))
                            span.set_attribute("gen_ai.usage.output_tokens", int(usage.get("completion_tokens", 0)))
                            try:
                                validate_tool_calls(result.get("tool_calls", []), payload.get("tools"),
                                                    reject_undeclared=False)
                            except ValueError as exc:
                                raise ProviderError(f"Provider returned invalid tool calls: {exc}", retryable=True) from exc
                            await self.provider_health.record(name, (asyncio.get_running_loop().time() - started) * 1000, True)
                            await self.breaker.success(name)
                            used_canary = bool(canary_selected and canary_config and
                                (name, model) == (canary_config.get("provider"), canary_config.get("model")))
                            result.update({"provider": name, "configured_model": model,
                                           "fallback_used": (name, model) != preferred and not used_canary,
                                           "cached": False, "canary_used": used_canary})
                            decision = "fallback" if result["fallback_used"] else "selected"
                            routing_decisions.append(self._decision(candidate, decision,
                                self._selection_reason(candidate_index, candidate, preferred, canary_config,
                                                       canary_selected, previous_failure), attempt_count))
                            self._skip_remaining(routing_decisions, candidates, candidate_index + 1,
                                                 "not_needed_after_prior_success")
                            routing_decisions.extend(policy_exclusions)
                            result["routing_decisions"] = routing_decisions
                            if cache_cfg.get("enabled") and cache_safe:
                                await self.cache.put(cache_key, result, int(cache_cfg.get("ttl_seconds", 60)))
                            return result
                        except RateLimitExceeded:
                            await self.breaker.abandon(name)
                            errors.append(f"{name}: rate limited")
                            candidate_failure = "provider_rate_limited"
                            break
                        except ProviderError as exc:
                            if started is not None and not isinstance(exc, ProviderCapabilityError):
                                await self.provider_health.record(name, (asyncio.get_running_loop().time() - started) * 1000, False)
                            span.set_attribute("error.type", type(exc).__name__)
                            span.set_attribute("llm.gateway.retryable", exc.retryable)
                            errors.append(f"{name}: {exc}")
                            candidate_failure = self._provider_failure_reason(exc)
                            if exc.retryable:
                                await self.breaker.failure(name)
                            else:
                                await self.breaker.abandon(name)
                            if not exc.retryable or attempt >= max_retries:
                                break
                            PROVIDER_RETRIES.labels(provider=name).inc()
                            await asyncio.sleep(min(base_delay * (2 ** attempt), 4.0))
                        except RateLimitBackendUnavailable as exc:
                            await self.breaker.abandon(name)
                            routing_decisions.append(self._decision(
                                candidate, "skipped", "shared_rate_limit_backend_unavailable", attempt_count))
                            self._skip_remaining(routing_decisions, candidates, candidate_index + 1,
                                                 "not_attempted_after_rate_limit_backend_failure")
                            routing_decisions.extend(policy_exclusions)
                            exc.routing_decisions = routing_decisions
                            raise
                routing_decisions.append(self._decision(candidate, "skipped",
                    candidate_failure or "candidate_failed", attempt_count))
                previous_failure = candidate_failure or "provider_error"
        routing_decisions.extend(policy_exclusions)
        failure = ProviderError("All configured providers failed: " + "; ".join(errors), retryable=False)
        failure.routing_decisions = routing_decisions
        raise failure

    async def stream_complete(self, route_name: str, payload: dict[str, Any], request_id: str | None = None,
                              tenant_policy: dict[str, Any] | None = None,
                              evaluation_context: dict[str, Any] | None = None):
        """Yield provider text deltas; retries/fallbacks stop once text was emitted."""
        route = self.routes[route_name]
        preferred = (route["provider"], route["model"])
        feature_canary, feature_reason = await self._feature_flag_canary(route_name, request_id, evaluation_context)
        candidates = await self._order_candidates(self.route_candidates(route_name, request_id, tenant_policy,
            canary_enabled=feature_canary))
        policy_exclusions = self.routing_policy_exclusions(route_name, request_id, tenant_policy,
                                                           canary_enabled=feature_canary)
        canary_config = route.get("canary")
        canary_candidate = ({"provider": canary_config.get("provider"), "model": canary_config.get("model")}
                            if canary_config else None)
        static_canary_selected = bool(canary_config and request_id is not None and
            int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100 <
            float(canary_config.get("traffic_percent", 0)))
        canary_selected = bool(canary_config and canary_candidate in candidates and
                               (feature_canary is True or
                                (feature_canary is None and static_canary_selected)))
        if feature_reason:
            policy_exclusions.append(self._decision(
                canary_candidate,
                "selected" if canary_selected else "skipped", feature_reason))
        if canary_selected:
            candidates = self._prioritize_canary(candidates, canary_config)
        cache_cfg = route.get("cache", {})
        cache_key = self._cache_key(payload, preferred, candidates, tenant_policy)
        cache_safe = self.cache_safe_payload(payload) and not canary_selected
        if cache_cfg.get("enabled") and cache_safe:
            cached = await self.cache.get(cache_key)
            if cached:
                yield {"delta": cached["text"]}
                cached_decision = self._decision(
                    {"provider": cached["provider"], "model": cached.get("configured_model", cached["model"])},
                    "selected", "response_cache_hit")
                yield {**cached, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                       "provider": cached["provider"], "configured_model": cached.get("configured_model", cached["model"]),
                       "fallback_used": False, "cached": True, "routing_decisions": [cached_decision], "final": True}
                return

        errors: list[str] = []
        routing_decisions: list[dict[str, Any]] = []
        previous_failure: str | None = None
        max_retries = int(self.config["server"].get("max_retries", 2))
        base_delay = float(self.config["server"].get("retry_base_delay_seconds", 0.25))
        async with self._concurrency():
            for candidate_index, candidate in enumerate(candidates):
                name, model = candidate["provider"], candidate["model"]
                provider_config = self.providers[name]
                emitted = False
                candidate_failure: str | None = None
                attempt_count = 0
                for attempt in range(max_retries + 1):
                    attempt_count = attempt + 1
                    started: float | None = None
                    try:
                        if not await self.breaker.allow(name):
                            errors.append(f"{name}: circuit open")
                            candidate_failure = "circuit_open"
                            break
                        await self.limiter.acquire(name, int(provider_config.get("requests_per_minute", 600)))
                        started = asyncio.get_running_loop().time()
                        text_parts: list[str] = []
                        final: dict[str, Any] = {}
                        used_canary = bool(canary_selected and canary_config and
                            (name, model) == (canary_config.get("provider"), canary_config.get("model")))
                        yield {"provider_progress": {"provider": name, "model": model,
                            "configured_model": model, "fallback_used": (name, model) != preferred and not used_canary,
                            "canary_used": used_canary}}
                        async for item in self.client.stream(name, model, payload, request_id=request_id):
                            if "usage_progress" in item:
                                yield {"usage_progress": item["usage_progress"]}
                                continue
                            if "delta" in item:
                                emitted = True
                                text_parts.append(item["delta"])
                                yield {"delta": item["delta"]}
                            elif "tool_call_delta" in item or "audio_delta" in item:
                                emitted = True
                                yield item
                            else:
                                final = item
                        elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000
                        await self.provider_health.record(name, elapsed_ms, True)
                        await self.breaker.success(name)
                        result = {"text": "".join(text_parts), "usage": final.get("usage", {}),
                                  "finish_reason": final.get("finish_reason"), "model": final.get("model", model),
                                  "provider": name, "configured_model": model,
                                  "fallback_used": (name, model) != preferred and not used_canary,
                                  "cached": False, "canary_used": used_canary}
                        decision = "fallback" if result["fallback_used"] else "selected"
                        routing_decisions.append(self._decision(candidate, decision,
                            self._selection_reason(candidate_index, candidate, preferred, canary_config,
                                                   canary_selected, previous_failure), attempt_count))
                        self._skip_remaining(routing_decisions, candidates, candidate_index + 1,
                                             "not_needed_after_prior_success")
                        routing_decisions.extend(policy_exclusions)
                        result["routing_decisions"] = routing_decisions
                        if cache_cfg.get("enabled") and cache_safe:
                            await self.cache.put(cache_key, result, int(cache_cfg.get("ttl_seconds", 60)))
                        yield {**result, "final": True}
                        return
                    except RateLimitExceeded:
                        await self.breaker.abandon(name)
                        errors.append(f"{name}: rate limited")
                        candidate_failure = "provider_rate_limited"
                        break
                    except ProviderError as exc:
                        unsupported_capability = isinstance(exc, ProviderCapabilityError)
                        if started is not None and not unsupported_capability:
                            await self.provider_health.record(name, (asyncio.get_running_loop().time() - started) * 1000, False)
                        errors.append(f"{name}: {exc}")
                        candidate_failure = self._provider_failure_reason(exc)
                        if exc.retryable:
                            await self.breaker.failure(name)
                        else:
                            await self.breaker.abandon(name)
                        if unsupported_capability and not emitted:
                            break
                        # Once a delta is visible to the caller, retrying could duplicate text.
                        if emitted or not exc.retryable:
                            routing_decisions.append(self._decision(candidate, "skipped",
                                "stream_failed_after_output" if emitted else candidate_failure or "provider_error",
                                attempt_count))
                            self._skip_remaining(routing_decisions, candidates, candidate_index + 1,
                                                 "stream_stopped_after_provider_error")
                            routing_decisions.extend(policy_exclusions)
                            exc.routing_decisions = routing_decisions
                            raise
                        if attempt >= max_retries:
                            # Retries are exhausted for this candidate, but no caller-visible
                            # delta has been emitted, so the next configured candidate is safe.
                            break
                        PROVIDER_RETRIES.labels(provider=name).inc()
                        await asyncio.sleep(min(base_delay * (2 ** attempt), 4.0))
                    except RateLimitBackendUnavailable as exc:
                        await self.breaker.abandon(name)
                        routing_decisions.append(self._decision(
                            candidate, "skipped", "shared_rate_limit_backend_unavailable", attempt_count))
                        self._skip_remaining(routing_decisions, candidates, candidate_index + 1,
                                             "not_attempted_after_rate_limit_backend_failure")
                        routing_decisions.extend(policy_exclusions)
                        exc.routing_decisions = routing_decisions
                        raise
                routing_decisions.append(self._decision(candidate, "skipped",
                    candidate_failure or "candidate_failed", attempt_count))
                previous_failure = candidate_failure or "provider_error"
        routing_decisions.extend(policy_exclusions)
        failure = ProviderError("All configured providers failed: " + "; ".join(errors), retryable=False)
        failure.routing_decisions = routing_decisions
        raise failure

    def _concurrency(self):
        if self.shared_redis:
            return RedisConcurrencyLimiter(self.shared_redis, self.max_concurrent_requests)
        return _SemaphoreLease(self.semaphore)

    @staticmethod
    def _prioritize_canary(candidates: list[dict[str, str]], canary: dict[str, Any]) -> list[dict[str, str]]:
        target = {"provider": canary["provider"], "model": canary["model"]}
        return [target, *[candidate for candidate in candidates if candidate != target]]

    def _cache_key(self, payload: dict[str, Any], preferred: tuple[str, str],
                   candidates: list[dict[str, str]], tenant_policy: dict[str, Any] | None) -> str:
        cache_payload = payload
        if tenant_policy is not None:
            cache_payload = {**payload, "_gateway_tenant_cache_scope": {
                "tenant_id": tenant_policy.get("_tenant_id"), "candidates": candidates,
                "allowed_providers": tenant_policy.get("allowed_providers"),
                "allowed_models": tenant_policy.get("allowed_models"),
                "allowed_regions": tenant_policy.get("allowed_regions"),
            }}
        return self.cache.key(cache_payload, preferred[0], preferred[1])


class _SemaphoreLease:
    def __init__(self, semaphore: asyncio.Semaphore):
        self.semaphore = semaphore

    async def __aenter__(self):
        await self.semaphore.acquire()

    async def __aexit__(self, *_: object):
        self.semaphore.release()
