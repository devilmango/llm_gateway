from __future__ import annotations

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
