from __future__ import annotations

import asyncio
from typing import Any

from redis.exceptions import RedisError


class ProviderHealth:
    """Shared provider outcome and EWMA latency signals used for route ordering."""

    def __init__(self, redis: Any | None = None, prefix: str = "llm-gateway:provider-health:"):
        self.redis = redis
        self.prefix = prefix
        self._values: dict[str, dict[str, float | int]] = {}
        self._lock = asyncio.Lock()

    async def record(self, provider: str, latency_ms: float, success: bool) -> None:
        if self.redis:
            try:
                await self.redis.eval(
                    """
                    local key=KEYS[1]
                    local old=tonumber(redis.call('HGET',key,'ewma_latency_ms') or ARGV[1])
                    local alpha=tonumber(ARGV[2])
                    redis.call('HSET',key,'ewma_latency_ms',old*(1-alpha)+tonumber(ARGV[1])*alpha)
                    local field=ARGV[3]
                    redis.call('HINCRBY',key,field,1)
                    redis.call('HSET',key,'last_latency_ms',ARGV[1])
                    redis.call('EXPIRE',key,86400)
                    return 1
                    """, 1, self.prefix + provider, max(0.0, latency_ms), 0.2,
                    "successes" if success else "failures",
                )
                return
            except RedisError:
                # Health signals optimize route ordering; they must never break inference.
                pass
        async with self._lock:
            state = self._values.setdefault(provider, {"ewma_latency_ms": latency_ms,
                                                        "last_latency_ms": latency_ms,
                                                        "successes": 0, "failures": 0})
            state["ewma_latency_ms"] = float(state["ewma_latency_ms"]) * 0.8 + latency_ms * 0.2
            state["last_latency_ms"] = latency_ms
            field = "successes" if success else "failures"
            state[field] = int(state[field]) + 1

    async def snapshot(self, providers: list[str]) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for provider in providers:
            state = None
            if self.redis:
                try:
                    raw = await self.redis.hgetall(self.prefix + provider)
                    if raw:
                        state = {key: float(value) if key.endswith("ms") else int(value)
                                 for key, value in raw.items()}
                except RedisError:
                    state = None
            if state is None:
                async with self._lock:
                    state = dict(self._values.get(provider, {}))
            if state:
                total = int(state.get("successes", 0)) + int(state.get("failures", 0))
                state["failure_rate"] = int(state.get("failures", 0)) / total if total else 0.0
                result[provider] = state
        return result
