from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Vector cache whose key never contains the source text and is tenant scoped."""

    def __init__(self, backend: Any | None = None) -> None:
        self.backend = backend or ResponseCache()

    @staticmethod
    def key(text: str, *, tenant: str, provider: str, model: str,
            pricing_version: str, secret: str) -> str:
        digest = hmac.new(secret.encode(), text.encode("utf-8"), hashlib.sha256).hexdigest()
        namespace = json.dumps([tenant, provider, model, pricing_version, digest], separators=(",", ":"))
        return "embedding:" + hashlib.sha256(namespace.encode()).hexdigest()

    async def get(self, key: str) -> list[float] | None:
        item = await self.backend.get(key)
        vector = item.get("embedding") if item else None
        return vector if isinstance(vector, list) else None

    async def put(self, key: str, vector: list[float], ttl_seconds: int) -> None:
        await self.backend.put(key, {"embedding": vector}, ttl_seconds)

class ResponseCache:
    def __init__(self, max_entries: int = 2_000) -> None:
        self._items: dict[str, tuple[float, dict[str, Any]]] = {}
        self.max_entries = max_entries

    @staticmethod
    def key(payload: dict[str, Any], provider: str, model: str) -> str:
        stable = json.dumps([provider, model, payload], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode()).hexdigest()

    async def get(self, key: str) -> dict[str, Any] | None:
        item = self._items.get(key)
        if item is None:
            return None
        expiry, value = item
        if expiry <= time.monotonic():
            self._items.pop(key, None)
            return None
        return value.copy()

    async def put(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        if len(self._items) >= self.max_entries:
            oldest = min(self._items, key=lambda k: self._items[k][0])
            self._items.pop(oldest, None)
        self._items[key] = (time.monotonic() + ttl_seconds, value.copy())


class RedisResponseCache:
    """Shared response cache. Redis failures are fail-open: call the provider instead."""

    def __init__(self, redis: Any, prefix: str = "llm-gateway:cache:") -> None:
        self.redis = redis
        self.prefix = prefix

    key = staticmethod(ResponseCache.key)

    async def get(self, key: str) -> dict[str, Any] | None:
        try:
            value = await self.redis.get(self.prefix + key)
            decoded = json.loads(value) if value else None
            return decoded if isinstance(decoded, dict) else None
        except (RedisError, ValueError, TypeError):
            logger.exception("redis_cache_read_failed")
            return None

    async def put(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        try:
            await self.redis.set(self.prefix + key, json.dumps(value), ex=ttl_seconds)
        except RedisError:
            logger.exception("redis_cache_write_failed")
