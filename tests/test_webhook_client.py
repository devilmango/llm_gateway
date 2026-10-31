from __future__ import annotations

import hashlib
import hmac

import pytest

from llm_gateway.webhook_client import (
    DuplicateWebhookEvent,
    WebhookVerificationError,
    WebhookVerifier,
)


def signed_headers(body: bytes, *, secret: bytes, key_id: str, event_id: str,
                   timestamp: int) -> dict[str, str]:
    signature = hmac.new(secret, str(timestamp).encode() + b"." + body, hashlib.sha256).hexdigest()
    return {
        "X-LLM-Gateway-Signature": f"t={timestamp},kid={key_id},v1={signature}",
        "X-LLM-Gateway-Key-Id": key_id,
        "X-LLM-Gateway-Event": "completed",
        "X-LLM-Gateway-Event-Id": event_id,
        "Idempotency-Key": event_id,
    }


def test_webhook_verifier_accepts_rotating_key_and_deduplicates_event():
    old_secret = b"old-secret"
    body = b'{"job_id":"job-7","status":"completed"}'
    verifier = WebhookVerifier({"current": b"new-secret", "previous": old_secret})
    headers = signed_headers(body, secret=old_secret, key_id="previous", event_id="event-7", timestamp=500)

    verified = verifier.verify(headers, body, now=500)

    assert verified.event_id == "event-7"
    assert verified.event_type == "completed"
    assert verified.key_id == "previous"
    assert verified.body == body
    with pytest.raises(DuplicateWebhookEvent):
        verifier.verify(headers, body, now=500)


@pytest.mark.parametrize("mutation, message", [
    ("stale", "timestamp"),
    ("tampered", "signature"),
    ("key_id", "key ID"),
    ("event_id", "event ID"),
])
def test_webhook_verifier_rejects_invalid_timestamp_signature_and_identity(mutation, message):
    secret = b"test-secret"
    body = b'{"event":"test"}'
    headers = signed_headers(body, secret=secret, key_id="key-1", event_id="event-1", timestamp=1000)
    verifier = WebhookVerifier({"key-1": secret}, tolerance_seconds=60)
    now = 1000
    if mutation == "stale":
        now = 1200
    elif mutation == "tampered":
        body = b'{"event":"changed"}'
    elif mutation == "key_id":
        headers["X-LLM-Gateway-Key-Id"] = "key-2"
    elif mutation == "event_id":
        headers["Idempotency-Key"] = "different-event"

    with pytest.raises(WebhookVerificationError, match=message):
        verifier.verify(headers, body, now=now)
