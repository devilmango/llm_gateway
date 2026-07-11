from __future__ import annotations

import asyncio
import json
from typing import Any

from opentelemetry import trace

from .cache import RedisResponseCache, ResponseCache
from .circuit import ProviderCircuitBreaker
from .distributed import RateLimitBackendUnavailable, RedisSlidingWindowLimiter
from .limiting import RateLimitExceeded, SlidingWindowLimiter
from .observability import PROVIDER_RETRIES
from .provider_health import ProviderHealth
from .providers import ProviderCapabilityError, ProviderClient, ProviderError


class GatewayService:
    def __init__(self, config: dict[str, Any], http: Any, shared_redis: Any | None = None):
        self.config = config
        self.providers = config["providers"]
        self.routes = config["routes"]
        self.client = ProviderClient(http, self.providers, float(config["server"].get("request_timeout_seconds", 45)))
        self.limiter = RedisSlidingWindowLimiter(shared_redis) if shared_redis else SlidingWindowLimiter()
        self.cache = RedisResponseCache(shared_redis) if shared_redis else ResponseCache()
        self.semaphore = asyncio.Semaphore(int(config["server"].get("max_concurrent_requests", 100)))
        self.application_limiter = self.limiter if shared_redis else SlidingWindowLimiter()
        breaker_config = config.get("circuit_breaker", {})
        self.breaker = ProviderCircuitBreaker(
            list(self.providers),
            failure_threshold=breaker_config.get("failure_threshold", 5),
            recovery_seconds=breaker_config.get("recovery_seconds", 30),
        )
        self.provider_health = ProviderHealth(shared_redis)

    async def acquire_application_quota(self, application: str, limit: int) -> None:
        await self.application_limiter.acquire(application, limit)

    def budget_reservation(self, route_name: str, payload: dict[str, Any], policy: dict[str, Any]) -> float:
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
        candidates = self.route_candidates(route_name)
        # UTF-8 bytes conservatively bound input token count; use the most expensive
        # candidate because a fallback can serve the request instead of the primary.
        prices = [self.providers[c["provider"]].get("models", {}).get(c["model"], {}) for c in candidates]
        input_rate = max((float(p.get("input_cost_per_1m", 0)) for p in prices), default=0)
        output_rate = max((float(p.get("output_cost_per_1m", 0)) for p in prices), default=0)
        return (input_bytes * input_rate + requested_tokens * output_rate) / 1_000_000

    def route_candidates(self, route_name: str) -> list[dict[str, str]]:
        if route_name not in self.routes:
            raise KeyError(route_name)
        route = self.routes[route_name]
        candidates = [{"provider": route["provider"], "model": route["model"]}]
        candidates.extend(route.get("fallbacks", []))
        for candidate in candidates:
            provider = candidate.get("provider")
            if provider not in self.providers:
                raise ValueError(f"Route {route_name!r} refers to unknown provider {provider!r}")
        return candidates

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

    async def complete(self, route_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        route = self.routes[route_name]
        preferred = (route["provider"], route["model"])
        candidates = await self._order_candidates(self.route_candidates(route_name))
        cache_cfg = route.get("cache", {})
        cache_key = self.cache.key(payload, preferred[0], preferred[1])
        cache_safe = self.cache_safe_payload(payload)
        if cache_cfg.get("enabled") and cache_safe:
            cached = await self.cache.get(cache_key)
            if cached:
                # A cache hit incurs no upstream token spend. The response still contains the
                # cached completion, while gateway usage represents billable upstream usage.
                cached["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                return {**cached, "cached": True, "fallback_used": False}

        errors: list[str] = []
        max_retries = int(self.config["server"].get("max_retries", 2))
        base_delay = float(self.config["server"].get("retry_base_delay_seconds", 0.25))
        async with self.semaphore:
            for candidate_index, candidate in enumerate(candidates):
                name, model = candidate["provider"], candidate["model"]
                provider_config = self.providers[name]
                limit = int(provider_config.get("requests_per_minute", 600))
                for attempt in range(max_retries + 1):
                    started: float | None = None
                    with trace.get_tracer("llm_gateway").start_as_current_span("llm.gateway.provider_attempt") as span:
                        span.set_attribute("llm.gateway.provider", name)
                        span.set_attribute("llm.gateway.model", model)
                        span.set_attribute("llm.gateway.attempt", attempt + 1)
                        span.set_attribute("llm.gateway.fallback_index", candidate_index)
                        try:
                            if not await self.breaker.allow(name):
                                errors.append(f"{name}: circuit open")
                                break
                            await self.limiter.acquire(name, limit)
                            started = asyncio.get_running_loop().time()
                            result = await self.client.complete(name, model, payload)
                            await self.provider_health.record(name, (asyncio.get_running_loop().time() - started) * 1000, True)
                            await self.breaker.success(name)
                            result.update({"provider": name, "configured_model": model,
                                           "fallback_used": (name, model) != preferred, "cached": False})
                            if cache_cfg.get("enabled") and cache_safe:
                                await self.cache.put(cache_key, result, int(cache_cfg.get("ttl_seconds", 60)))
                            return result
                        except RateLimitExceeded:
                            await self.breaker.abandon(name)
                            errors.append(f"{name}: rate limited")
                            break
                        except ProviderError as exc:
                            if started is not None and not isinstance(exc, ProviderCapabilityError):
                                await self.provider_health.record(name, (asyncio.get_running_loop().time() - started) * 1000, False)
                            span.set_attribute("error.type", type(exc).__name__)
                            span.set_attribute("llm.gateway.retryable", exc.retryable)
                            errors.append(f"{name}: {exc}")
                            if exc.retryable:
                                await self.breaker.failure(name)
                            else:
                                await self.breaker.abandon(name)
                            if not exc.retryable or attempt >= max_retries:
                                break
                            PROVIDER_RETRIES.labels(provider=name).inc()
                            await asyncio.sleep(min(base_delay * (2 ** attempt), 4.0))
                        except RateLimitBackendUnavailable:
                            await self.breaker.abandon(name)
                            raise
        raise ProviderError("All configured providers failed: " + "; ".join(errors), retryable=False)

    async def stream_complete(self, route_name: str, payload: dict[str, Any]):
        """Yield provider text deltas; retries/fallbacks stop once text was emitted."""
        route = self.routes[route_name]
        preferred = (route["provider"], route["model"])
        candidates = await self._order_candidates(self.route_candidates(route_name))
        cache_cfg = route.get("cache", {})
        cache_key = self.cache.key(payload, preferred[0], preferred[1])
        cache_safe = self.cache_safe_payload(payload)
        if cache_cfg.get("enabled") and cache_safe:
            cached = await self.cache.get(cache_key)
            if cached:
                yield {"delta": cached["text"]}
                yield {**cached, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                       "provider": cached["provider"], "configured_model": cached.get("configured_model", cached["model"]),
                       "fallback_used": False, "cached": True, "final": True}
                return

        errors: list[str] = []
        max_retries = int(self.config["server"].get("max_retries", 2))
        base_delay = float(self.config["server"].get("retry_base_delay_seconds", 0.25))
        async with self.semaphore:
            for candidate_index, candidate in enumerate(candidates):
                name, model = candidate["provider"], candidate["model"]
                provider_config = self.providers[name]
                emitted = False
                for attempt in range(max_retries + 1):
                    started: float | None = None
                    try:
                        if not await self.breaker.allow(name):
                            errors.append(f"{name}: circuit open")
                            break
                        await self.limiter.acquire(name, int(provider_config.get("requests_per_minute", 600)))
                        started = asyncio.get_running_loop().time()
                        text_parts: list[str] = []
                        final: dict[str, Any] = {}
                        async for item in self.client.stream(name, model, payload):
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
                                  "fallback_used": (name, model) != preferred, "cached": False}
                        if cache_cfg.get("enabled") and cache_safe:
                            await self.cache.put(cache_key, result, int(cache_cfg.get("ttl_seconds", 60)))
                        yield {**result, "final": True}
                        return
                    except RateLimitExceeded:
                        await self.breaker.abandon(name)
                        errors.append(f"{name}: rate limited")
                        break
                    except ProviderError as exc:
                        unsupported_capability = isinstance(exc, ProviderCapabilityError)
                        if started is not None and not unsupported_capability:
                            await self.provider_health.record(name, (asyncio.get_running_loop().time() - started) * 1000, False)
                        errors.append(f"{name}: {exc}")
                        if exc.retryable:
                            await self.breaker.failure(name)
                        else:
                            await self.breaker.abandon(name)
                        if unsupported_capability and not emitted:
                            break
                        # Once a delta is visible to the caller, retrying could duplicate text.
                        if emitted or not exc.retryable or attempt >= max_retries:
                            raise
                        PROVIDER_RETRIES.labels(provider=name).inc()
                        await asyncio.sleep(min(base_delay * (2 ** attempt), 4.0))
                    except RateLimitBackendUnavailable:
                        await self.breaker.abandon(name)
                        raise
        raise ProviderError("All configured providers failed: " + "; ".join(errors), retryable=False)
