from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class ResponseCache:
    def __init__(self, max_entries: int = 2_000) -> None:
        self._items: dict[str, tuple[float, dict[str, Any]]] = {}
        self.max_entries = max_entries

    @staticmethod
    def key(payload: dict[str, Any], provider: str, model: str) -> str:
        stable = json.dumps([provider, model, payload], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._items.get(key)
        if item is None:
            return None
        expiry, value = item
        if expiry <= time.monotonic():
            self._items.pop(key, None)
            return None
        return value.copy()

    def put(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        if len(self._items) >= self.max_entries:
            oldest = min(self._items, key=lambda k: self._items[k][0])
            self._items.pop(oldest, None)
        self._items[key] = (time.monotonic() + ttl_seconds, value.copy())
