from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry import trace

from .cache import ResponseCache
from .limiting import RateLimitExceeded, SlidingWindowLimiter
from .observability import PROVIDER_RETRIES
from .providers import ProviderClient, ProviderError


class GatewayService:
    def __init__(self, config: dict[str, Any], http: Any):
        self.config = config
        self.providers = config["providers"]
        self.routes = config["routes"]
        self.client = ProviderClient(http, self.providers, float(config["server"].get("request_timeout_seconds", 45)))
        self.limiter = SlidingWindowLimiter()
        self.cache = ResponseCache()
        self.semaphore = asyncio.Semaphore(int(config["server"].get("max_concurrent_requests", 100)))
        self.application_limiter = SlidingWindowLimiter()

    async def acquire_application_quota(self, application: str, limit: int) -> None:
        await self.application_limiter.acquire(application, limit)

    def budget_reservation(self, route_name: str, payload: dict[str, Any], policy: dict[str, Any]) -> float:
        max_input_bytes = int(policy.get("max_input_bytes", 32_768))
        input_bytes = sum(len(message["content"].encode("utf-8")) for message in payload["messages"])
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

    async def complete(self, route_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        candidates = self.route_candidates(route_name)
        route = self.routes[route_name]
        cache_cfg = route.get("cache", {})
        cache_key = self.cache.key(payload, candidates[0]["provider"], candidates[0]["model"])
        if cache_cfg.get("enabled"):
            cached = self.cache.get(cache_key)
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
                    with trace.get_tracer("llm_gateway").start_as_current_span("llm.gateway.provider_attempt") as span:
                        span.set_attribute("llm.gateway.provider", name)
                        span.set_attribute("llm.gateway.model", model)
                        span.set_attribute("llm.gateway.attempt", attempt + 1)
                        span.set_attribute("llm.gateway.fallback_index", candidate_index)
                        try:
                            await self.limiter.acquire(name, limit)
                            result = await self.client.complete(name, model, payload)
                            result.update({"provider": name, "configured_model": model,
                                           "fallback_used": candidate_index > 0, "cached": False})
                            if cache_cfg.get("enabled"):
                                self.cache.put(cache_key, result, int(cache_cfg.get("ttl_seconds", 60)))
                            return result
                        except RateLimitExceeded:
                            errors.append(f"{name}: rate limited")
                            break
                        except ProviderError as exc:
                            span.set_attribute("error.type", type(exc).__name__)
                            span.set_attribute("llm.gateway.retryable", exc.retryable)
                            errors.append(f"{name}: {exc}")
                            if not exc.retryable or attempt >= max_retries:
                                break
                            PROVIDER_RETRIES.labels(provider=name).inc()
                            await asyncio.sleep(min(base_delay * (2 ** attempt), 4.0))
        raise ProviderError("All configured providers failed: " + "; ".join(errors), retryable=False)
