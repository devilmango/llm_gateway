from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimitExceeded(Exception):
    pass


class SlidingWindowLimiter:
    """Process-local request-per-minute limiter, keyed by provider."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, limit: int) -> None:
        async with self._lock:
            now = time.monotonic()
            events = self._events.setdefault(key, deque())
            while events and now - events[0] >= 60:
                events.popleft()
            if len(events) >= limit:
                raise RateLimitExceeded(f"Provider {key!r} is at its configured request limit")
            events.append(now)
