from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import ClassVar

from redis.exceptions import RedisError

from .observability import PROVIDER_CIRCUIT_STATE


@dataclass
class _ProviderState:
    state: str = "closed"
    failures: int = 0
    opened_at: float = 0.0
    probe_in_flight: bool = False


class ProviderCircuitBreaker:
    """Circuit breaker with a shared Redis state when Redis is configured."""

    STATE_VALUE: ClassVar[dict[str, int]] = {"closed": 0, "half_open": 1, "open": 2}

    _ALLOW = """
    local key=KEYS[1]
    local now=tonumber(redis.call('TIME')[1])*1000+math.floor(tonumber(redis.call('TIME')[2])/1000)
    local state=redis.call('HGET',key,'state') or 'closed'
    local opened=tonumber(redis.call('HGET',key,'opened_at_ms') or '0')
    if state == 'open' then
      if now-opened < tonumber(ARGV[1]) then return 0 end
      state='half_open'
      redis.call('HSET',key,'state',state,'probe_in_flight','0')
    end
    if state == 'half_open' then
      if redis.call('HGET',key,'probe_in_flight') == '1' then return 0 end
      redis.call('HSET',key,'probe_in_flight','1')
    end
    redis.call('EXPIRE',key,86400)
    return 1
    """
    _SUCCESS = "redis.call('HSET',KEYS[1],'state','closed','failures','0','opened_at_ms','0','probe_in_flight','0'); redis.call('EXPIRE',KEYS[1],86400); return 1"
    _FAILURE = """
    local key=KEYS[1]
    local failures=redis.call('HINCRBY',key,'failures',1)
    local state=redis.call('HGET',key,'state') or 'closed'
    if state == 'half_open' or failures >= tonumber(ARGV[1]) then
      local t=redis.call('TIME')
      local now=tonumber(t[1])*1000+math.floor(tonumber(t[2])/1000)
      redis.call('HSET',key,'state','open','opened_at_ms',now,'probe_in_flight','0')
    end
    redis.call('EXPIRE',key,86400)
    return failures
    """
    _ABANDON = "if redis.call('HGET',KEYS[1],'state') == 'half_open' then redis.call('HSET',KEYS[1],'probe_in_flight','0') end; return 1"

    def __init__(self, providers: list[str], failure_threshold: int = 5, recovery_seconds: float = 30,
                 redis=None, prefix: str = "llm-gateway:provider-circuit:"):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_seconds = max(0.1, float(recovery_seconds))
        self.redis = redis
        self.prefix = prefix
        self._states = {provider: _ProviderState() for provider in providers}
        self._lock = asyncio.Lock()
        for provider in providers:
            self._set_metric(provider, "closed")

    @staticmethod
    def _set_metric(provider: str, state: str) -> None:
        PROVIDER_CIRCUIT_STATE.labels(provider=provider).set(ProviderCircuitBreaker.STATE_VALUE[state])

    async def allow(self, provider: str) -> bool:
        if self.redis:
            try:
                allowed = await self.redis.eval(self._ALLOW, 1, self.prefix + provider,
                                                int(self.recovery_seconds * 1000))
                if not allowed:
                    raw = await self.redis.hget(self.prefix + provider, "state")
                    self._set_metric(provider, raw or "open")
                return bool(allowed)
            except RedisError:
                pass
        async with self._lock:
            status = self._states.setdefault(provider, _ProviderState())
            if status.state == "open":
                if time.monotonic() - status.opened_at < self.recovery_seconds:
                    return False
                status.state = "half_open"
                status.probe_in_flight = False
                self._set_metric(provider, status.state)
            if status.state == "half_open":
                if status.probe_in_flight:
                    return False
                status.probe_in_flight = True
            return True

    async def success(self, provider: str) -> None:
        if self.redis:
            try:
                await self.redis.eval(self._SUCCESS, 1, self.prefix + provider)
                self._set_metric(provider, "closed")
                return
            except RedisError:
                pass
        async with self._lock:
            status = self._states.setdefault(provider, _ProviderState())
            status.state = "closed"
            status.failures = 0
            status.opened_at = 0
            status.probe_in_flight = False
            self._set_metric(provider, status.state)

    async def failure(self, provider: str) -> None:
        if self.redis:
            try:
                await self.redis.eval(self._FAILURE, 1, self.prefix + provider,
                                      self.failure_threshold)
                raw = await self.redis.hget(self.prefix + provider, "state")
                self._set_metric(provider, raw or "closed")
                return
            except RedisError:
                pass
        async with self._lock:
            status = self._states.setdefault(provider, _ProviderState())
            status.failures += 1
            if status.state == "half_open" or status.failures >= self.failure_threshold:
                status.state = "open"
                status.opened_at = time.monotonic()
                status.probe_in_flight = False
                self._set_metric(provider, status.state)

    async def abandon(self, provider: str) -> None:
        """Release a half-open probe when no provider-health result was obtained."""
        if self.redis:
            try:
                await self.redis.eval(self._ABANDON, 1, self.prefix + provider)
                return
            except RedisError:
                pass
        async with self._lock:
            status = self._states.setdefault(provider, _ProviderState())
            if status.state == "half_open":
                status.probe_in_flight = False

    async def snapshot(self) -> dict[str, dict[str, int | str | float]]:
        if self.redis:
            result = {}
            for provider in self._states:
                try:
                    raw = await self.redis.hgetall(self.prefix + provider)
                except RedisError:
                    raw = {}
                if raw:
                    state = raw.get("state", "closed")
                    opened = int(raw.get("opened_at_ms", 0)) / 1000
                    failures = int(raw.get("failures", 0))
                    result[provider] = {"state": state, "failures": failures,
                        "retry_after_seconds": max(0.0, self.recovery_seconds - (time.time() - opened))
                        if state == "open" else 0.0}
                    self._set_metric(provider, state)
            if result:
                return result
        async with self._lock:
            return {provider: {"state": status.state, "failures": status.failures,
                               "retry_after_seconds": max(0.0, self.recovery_seconds - (time.monotonic() - status.opened_at))
                               if status.state == "open" else 0.0}
                    for provider, status in self._states.items()}
