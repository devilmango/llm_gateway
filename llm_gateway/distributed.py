from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from typing import Any

from redis.exceptions import RedisError

from .limiting import RateLimitExceeded


class RateLimitBackendUnavailable(Exception):
    """Shared quota state could not be checked; do not send an unmetered request."""


_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
local count = redis.call('ZCARD', key)
if count >= limit then return 0 end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, 65000)
return 1
"""


class RedisSlidingWindowLimiter:
    """Atomic cross-replica request-per-minute limiter backed by Redis sorted sets."""

    def __init__(self, redis: Any, prefix: str = "llm-gateway:limit:") -> None:
        self.redis = redis
        self.prefix = prefix

    async def acquire(self, key: str, limit: int) -> None:
        digest = hashlib.sha256(key.encode()).hexdigest()
        redis_key = self.prefix + digest
        now_ms = int(time.time() * 1_000)
        try:
            accepted = await self.redis.eval(
                _SLIDING_WINDOW_SCRIPT, 1, redis_key, now_ms, now_ms - 60_000, limit,
                f"{now_ms}:{uuid.uuid4().hex}",
            )
        except RedisError as exc:
            raise RateLimitBackendUnavailable("Shared rate-limit store is unavailable") from exc
        if not accepted:
            raise RateLimitExceeded(f"Request quota for {key!r} is exhausted")


_CONCURRENCY_ACQUIRE = """
local key=KEYS[1]
local now=tonumber(ARGV[1])
local expiry=tonumber(ARGV[2])
local limit=tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE',key,'-inf',now)
if redis.call('ZCARD',key) >= limit then return 0 end
redis.call('ZADD',key,expiry,ARGV[4])
redis.call('PEXPIRE',key,math.max(1000,expiry-now+1000))
return 1
"""

_CONCURRENCY_REFRESH = """
if redis.call('ZSCORE',KEYS[1],ARGV[1]) then
  redis.call('ZADD',KEYS[1],ARGV[3],ARGV[1])
  redis.call('PEXPIRE',KEYS[1],ARGV[4])
  return 1
end
return 0
"""


class RedisConcurrencyLimiter:
    """Cross-replica concurrency leases. Expired leases are reclaimed atomically."""

    def __init__(self, redis: Any, limit: int, prefix: str = "llm-gateway:concurrency:", lease_seconds: int = 120):
        self.redis = redis
        self.limit = max(1, int(limit))
        self.prefix = prefix
        self.lease_ms = max(10, int(lease_seconds)) * 1000

    async def __aenter__(self):
        self.token = uuid.uuid4().hex
        self.key = self.prefix + "requests"
        while True:
            now = int(time.time() * 1000)
            expiry = now + self.lease_ms
            try:
                acquired = await self.redis.eval(_CONCURRENCY_ACQUIRE, 1, self.key, now, expiry,
                                                 self.limit, self.token)
            except RedisError as exc:
                raise RateLimitBackendUnavailable("Shared concurrency store is unavailable") from exc
            if acquired:
                break
            await asyncio.sleep(0.05)
        self._renew_task = asyncio.create_task(self._renew())
        return self

    async def _renew(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.lease_ms / 3000)
                now = int(time.time() * 1000)
                expiry = now + self.lease_ms
                await self.redis.eval(_CONCURRENCY_REFRESH, 1, self.key, self.token, now,
                                      expiry, self.lease_ms + 1000)
        except (asyncio.CancelledError, RedisError):
            return

    async def __aexit__(self, *_: object) -> None:
        self._renew_task.cancel()
        try:
            await self._renew_task
        except asyncio.CancelledError:
            pass
        try:
            await self.redis.zrem(self.key, self.token)
        except RedisError:
            # The lease expires automatically if Redis is briefly unavailable.
            pass
