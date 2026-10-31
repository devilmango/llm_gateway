from __future__ import annotations

import hashlib
import ipaddress
import socket
from urllib.parse import urlsplit


class UnsafeWebhookTarget(ValueError):
    """A webhook target violates the gateway's outbound network policy."""


def webhook_endpoint_id(url: str) -> str:
    """Return a stable, non-reversible identifier suitable for logs and admin APIs."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def validate_webhook_url(url: str, *, resolve_dns: bool = True,
                         allow_private_targets: bool = False) -> None:
    """Reject malformed, credential-bearing, local, and private webhook destinations."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeWebhookTarget("invalid_webhook_url") from exc
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment):
        raise UnsafeWebhookTarget("invalid_webhook_url")
    if (parsed.scheme == "https" and port not in {None, 443}) or (parsed.scheme == "http" and port not in {None, 80}):
        raise UnsafeWebhookTarget("webhook_port_not_allowed")

    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "metadata.google.internal"} or host.endswith((
            ".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        raise UnsafeWebhookTarget("local_webhook_target_not_allowed")
    if allow_private_targets:
        return
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        addresses = []
        if resolve_dns:
            try:
                records = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80),
                                             type=socket.SOCK_STREAM)
            except OSError as exc:
                raise UnsafeWebhookTarget("webhook_dns_resolution_failed") from exc
            addresses = [ipaddress.ip_address(record[4][0]) for record in records]
            if not addresses:
                raise UnsafeWebhookTarget("webhook_dns_resolution_failed")

    if not allow_private_targets and any(not address.is_global for address in addresses):
        raise UnsafeWebhookTarget("private_webhook_target_not_allowed")
