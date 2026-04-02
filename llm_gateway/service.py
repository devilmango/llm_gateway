from __future__ import annotations

import asyncio
import time
from typing import Any

from .cache import ResponseCache
from .limiting import RateLimitExceeded, SlidingWindowLimiter
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
                        errors.append(f"{name}: {exc}")
                        if not exc.retryable or attempt >= max_retries:
                            break
                        await asyncio.sleep(min(base_delay * (2 ** attempt), 4.0))
        raise ProviderError("All configured providers failed: " + "; ".join(errors), retryable=False)
