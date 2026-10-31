"""Small receiver-side helper for validating gateway webhook deliveries."""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass


class WebhookVerificationError(ValueError):
    """A webhook's signature, timestamp, key, or event identity is invalid."""


class DuplicateWebhookEvent(WebhookVerificationError):
    """The event ID was already accepted by this verifier instance."""


@dataclass(frozen=True)
class VerifiedWebhook:
    event_id: str
    event_type: str | None
    key_id: str
    timestamp: int
    body: bytes


class WebhookVerifier:
    """Verify timestamped HMAC signatures and deduplicate accepted event IDs.

    ``secrets`` maps signing key IDs to UTF-8 strings or raw bytes. Keep old
    keys in the mapping during rotation so deliveries signed before rotation
    remain verifiable. Deduplication is bounded and local to this process; use
    a shared idempotency store when the receiver runs multiple replicas.
    """

    def __init__(self, secrets: Mapping[str, str | bytes], *, tolerance_seconds: int = 300,
                 deduplication_ttl_seconds: int = 86_400, max_seen_events: int = 10_000) -> None:
        if not secrets or any(not key_id or not secret for key_id, secret in secrets.items()):
            raise ValueError("secrets must map non-empty key IDs to non-empty secrets")
        if tolerance_seconds < 0 or deduplication_ttl_seconds < 1 or max_seen_events < 1:
            raise ValueError("tolerances, deduplication TTL, and capacity must be positive")
        self._secrets = {key_id: secret.encode("utf-8") if isinstance(secret, str) else secret
                         for key_id, secret in secrets.items()}
        self.tolerance_seconds = tolerance_seconds
        self.deduplication_ttl_seconds = deduplication_ttl_seconds
        self.max_seen_events = max_seen_events
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def verify(self, headers: Mapping[str, str], body: bytes | str, *, now: float | None = None,
               deduplicate: bool = True) -> VerifiedWebhook:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        signature_header = normalized.get("x-llm-gateway-signature", "")
        fields = {}
        for part in signature_header.split(","):
            name, separator, value = part.strip().partition("=")
            if not separator or name in fields:
                raise WebhookVerificationError("malformed signature header")
            fields[name] = value
        try:
            timestamp = int(fields["t"])
            key_id = fields["kid"]
            signature = fields["v1"]
        except (KeyError, ValueError) as exc:
            raise WebhookVerificationError("malformed signature header") from exc
        current = time.time() if now is None else float(now)
        if abs(current - timestamp) > self.tolerance_seconds:
            raise WebhookVerificationError("webhook timestamp is outside the allowed tolerance")
        secret = self._secrets.get(key_id)
        if secret is None:
            raise WebhookVerificationError("unknown webhook signing key")
        header_key_id = normalized.get("x-llm-gateway-key-id")
        if header_key_id is not None and not hmac.compare_digest(header_key_id, key_id):
            raise WebhookVerificationError("signature key ID header mismatch")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", signature):
            raise WebhookVerificationError("malformed webhook digest")
        raw_body = body.encode("utf-8") if isinstance(body, str) else body
        signed = str(timestamp).encode("ascii") + b"." + raw_body
        expected = hmac.new(secret, signed, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature.lower()):
            raise WebhookVerificationError("webhook signature mismatch")
        event_id = normalized.get("x-llm-gateway-event-id") or normalized.get("idempotency-key")
        idempotency_key = normalized.get("idempotency-key")
        if not event_id or (idempotency_key and not hmac.compare_digest(event_id, idempotency_key)):
            raise WebhookVerificationError("missing or inconsistent webhook event ID")
        if deduplicate:
            self._accept_once(event_id, current)
        return VerifiedWebhook(event_id, normalized.get("x-llm-gateway-event"), key_id,
                               timestamp, raw_body)

    def _accept_once(self, event_id: str, now: float) -> None:
        with self._lock:
            expired = [key for key, expiry in self._seen.items() if expiry <= now]
            for key in expired:
                self._seen.pop(key, None)
            if event_id in self._seen:
                raise DuplicateWebhookEvent("webhook event ID was already accepted")
            self._seen[event_id] = now + self.deduplication_ttl_seconds
            while len(self._seen) > self.max_seen_events:
                self._seen.popitem(last=False)
