from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import ClassVar

from .observability import PROVIDER_CIRCUIT_STATE


@dataclass
class _ProviderState:
    state: str = "closed"
    failures: int = 0
    opened_at: float = 0.0
    probe_in_flight: bool = False


class ProviderCircuitBreaker:
    """In-process circuit breaker with one half-open recovery probe per provider."""

    STATE_VALUE: ClassVar[dict[str, int]] = {"closed": 0, "half_open": 1, "open": 2}

    def __init__(self, providers: list[str], failure_threshold: int = 5, recovery_seconds: float = 30):
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_seconds = max(0.1, float(recovery_seconds))
        self._states = {provider: _ProviderState() for provider in providers}
        self._lock = asyncio.Lock()
        for provider in providers:
            self._set_metric(provider, "closed")

    @staticmethod
    def _set_metric(provider: str, state: str) -> None:
        PROVIDER_CIRCUIT_STATE.labels(provider=provider).set(ProviderCircuitBreaker.STATE_VALUE[state])

    async def allow(self, provider: str) -> bool:
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
        async with self._lock:
            status = self._states.setdefault(provider, _ProviderState())
            status.state = "closed"
            status.failures = 0
            status.opened_at = 0
            status.probe_in_flight = False
            self._set_metric(provider, status.state)

    async def failure(self, provider: str) -> None:
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
        async with self._lock:
            status = self._states.setdefault(provider, _ProviderState())
            if status.state == "half_open":
                status.probe_in_flight = False

    async def snapshot(self) -> dict[str, dict[str, int | str | float]]:
        async with self._lock:
            return {provider: {"state": status.state, "failures": status.failures,
                               "retry_after_seconds": max(0.0, self.recovery_seconds - (time.monotonic() - status.opened_at))
                               if status.state == "open" else 0.0}
                    for provider, status in self._states.items()}
