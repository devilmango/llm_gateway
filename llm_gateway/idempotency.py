"""Short-lived idempotency records for successful chat completions."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class IdempotencyRegistry:
    """Deduplicate requests per gateway process, or across replicas with Redis.

    Only a digest of the caller key is used as a backend key. Successful response
    bodies are retained for a short TTL so a retry can receive the original result.
    """

    def __init__(self, redis: Any | None = None, ttl_seconds: int = 86_400) -> None:
        self.redis = redis
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._records: dict[str, tuple[float, dict[str, Any]]] = {}

    @staticmethod
    def _storage_key(scope: str) -> str:
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        return f"llm-gateway:idempotency:{digest}"

    async def claim(self, scope: str, fingerprint: str) -> tuple[bool, dict[str, Any] | None]:
        key = self._storage_key(scope)
        pending = {"state": "pending", "fingerprint": fingerprint}
        if self.redis is not None:
            created = await self.redis.set(key, json.dumps(pending), nx=True, ex=self.ttl_seconds)
            if created:
                return True, None
            value = await self.redis.get(key)
            return False, json.loads(value) if value else None
        now = time.monotonic()
        for old_key, (expires, _) in list(self._records.items()):
            if expires <= now:
                del self._records[old_key]
        previous = self._records.get(key)
        if previous:
            return False, previous[1]
        self._records[key] = (now + self.ttl_seconds, pending)
        return True, None

    async def complete(self, scope: str, fingerprint: str, response: dict[str, Any]) -> None:
        key = self._storage_key(scope)
        value = {"state": "complete", "fingerprint": fingerprint, "response": response}
        if len(json.dumps(value, separators=(",", ":")).encode("utf-8")) > 1_000_000:
            value = {"state": "incomplete", "fingerprint": fingerprint}
        if self.redis is not None:
            await self.redis.set(key, json.dumps(value), ex=self.ttl_seconds)
        else:
            self._records[key] = (time.monotonic() + self.ttl_seconds, value)

    async def abandon(self, scope: str, fingerprint: str) -> None:
        """Keep an incomplete marker until expiry to prevent duplicate spend."""
        key = self._storage_key(scope)
        value = {"state": "incomplete", "fingerprint": fingerprint}
        if self.redis is not None:
            await self.redis.set(key, json.dumps(value), ex=self.ttl_seconds)
        else:
            self._records[key] = (time.monotonic() + self.ttl_seconds, value)
