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

    @staticmethod
    def _state_key(provider: str, operation: str) -> str:
        return f"{provider}:{operation}"

    async def record(self, provider: str, latency_ms: float, success: bool,
                     operation: str = "chat") -> None:
        state_key = self._state_key(provider, operation)
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
                    """, 1, self.prefix + state_key, max(0.0, latency_ms), 0.2,
                    "successes" if success else "failures",
                )
                return
            except RedisError:
                # Health signals optimize route ordering; they must never break inference.
                pass
        async with self._lock:
            state = self._values.setdefault(state_key, {"ewma_latency_ms": latency_ms,
                                                        "last_latency_ms": latency_ms,
                                                        "successes": 0, "failures": 0})
            state["ewma_latency_ms"] = float(state["ewma_latency_ms"]) * 0.8 + latency_ms * 0.2
            state["last_latency_ms"] = latency_ms
            field = "successes" if success else "failures"
            state[field] = int(state[field]) + 1

    async def snapshot(self, providers: list[str]) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for provider in providers:
            state_key = self._state_key(provider, "chat")
            state = None
            if self.redis:
                try:
                    raw = await self.redis.hgetall(self.prefix + state_key)
                    if raw:
                        state = {key: float(value) if key.endswith("ms") else int(value)
                                 for key, value in raw.items()}
                except RedisError:
                    state = None
            if state is None:
                async with self._lock:
                    state = dict(self._values.get(state_key, {}))
            if state:
                total = int(state.get("successes", 0)) + int(state.get("failures", 0))
                state["failure_rate"] = int(state.get("failures", 0)) / total if total else 0.0
                result[provider] = state
        return result

    async def snapshot_scoped(self, providers: list[str], operations: list[str]
                              ) -> dict[str, dict[str, dict[str, float | int]]]:
        result: dict[str, dict[str, dict[str, float | int]]] = {}
        for provider in providers:
            for operation in operations:
                state_key = self._state_key(provider, operation)
                state = None
                if self.redis:
                    try:
                        raw = await self.redis.hgetall(self.prefix + state_key)
                        if raw:
                            state = {key: float(value) if key.endswith("ms") else int(value)
                                     for key, value in raw.items()}
                    except RedisError:
                        pass
                if state is None:
                    async with self._lock:
                        state = dict(self._values.get(state_key, {}))
                if state:
                    total = int(state.get("successes", 0)) + int(state.get("failures", 0))
                    state["failure_rate"] = int(state.get("failures", 0)) / total if total else 0.0
                    result.setdefault(provider, {})[operation] = state
        return result
