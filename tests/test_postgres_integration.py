"""Integration checks for behaviors that depend on PostgreSQL locking semantics."""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from llm_gateway.storage import MaintenanceLeaseLost


def _postgres_store():
    dsn = os.getenv("TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("TEST_POSTGRES_URL is not configured; CI provides a PostgreSQL service")
    from llm_gateway.postgres_storage import PostgresUsageStore

    return PostgresUsageStore(dsn)


def test_postgres_competing_replicas_have_one_maintenance_lease_owner():
    first, second = _postgres_store(), _postgres_store()
    name = "integration-compete-" + uuid.uuid4().hex
    barrier = Barrier(2)

    def acquire(store, owner):
        barrier.wait(timeout=10)
        return owner, store.acquire_maintenance_lease(name, owner, 60)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda pair: acquire(*pair),
                ((first, "replica-a"), (second, "replica-b"))))
        winners = [(owner, token) for owner, token in results if token is not None]
        assert len(winners) == 1
        assert winners[0][1] == 1
        state = first.maintenance_lease_state(name)
        assert state["active"] is True
        assert state["owner_id"] == winners[0][0]
    finally:
        first.close()
        second.close()


def test_postgres_process_crash_and_lease_expiry_allow_fenced_takeover():
    store = _postgres_store()
    name = "integration-crash-" + uuid.uuid4().hex
    code = """
import os
from llm_gateway.postgres_storage import PostgresUsageStore
store = PostgresUsageStore(os.environ['TEST_POSTGRES_URL'])
token = store.acquire_maintenance_lease(os.environ['TEST_LEASE_NAME'], 'crashed-process', 2)
print(token, flush=True)
os._exit(23)
"""
    env = {**os.environ, "TEST_LEASE_NAME": name}
    try:
        crashed = subprocess.run([sys.executable, "-c", code], check=False, env=env,
            capture_output=True, text=True, timeout=30)
        assert crashed.returncode == 23, crashed.stderr
        old_token = int(crashed.stdout.strip())
        assert old_token == 1
        assert store.acquire_maintenance_lease(name, "replacement-before-expiry", 30) is None

        time.sleep(2.1)
        new_token = store.acquire_maintenance_lease(name, "replacement-after-expiry", 30)
        assert new_token == old_token + 1
        stale_lease = {"name": name, "owner_id": "crashed-process", "fencing_token": old_token}
        with pytest.raises(MaintenanceLeaseLost):
            store.prune_admin_audit_events(365, now=datetime.now(UTC), lease=stale_lease)
    finally:
        store.close()


def test_postgres_audit_chain_is_valid_after_concurrent_replica_writes():
    first, second = _postgres_store(), _postgres_store()
    prefix = "integration-concurrent-" + uuid.uuid4().hex
    before = first.audit_chain_status()["record_count"]
    writes = 48

    def write(index: int) -> None:
        store = first if index % 2 else second
        store.record_audit(f"replica-{index % 2}", f"{prefix}:{index}", 200)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(write, range(writes)))
        status = first.audit_chain_status()
        assert status["verified"] is True
        assert status["record_count"] == before + writes
        actions = [event["action"] for event in first.query_audit_events(limit=1000)["events"]]
        assert sum(action.startswith(prefix) for action in actions) == writes
    finally:
        first.close()
        second.close()


def test_postgres_audit_holds_are_transactional_and_audited():
    store = _postgres_store()
    # Keep synthetic event times monotonic even when CI runs this test twice
    # against the same service database.
    latest = store.audit_events(limit=1)
    latest_time = datetime.fromisoformat(latest[0]["occurred_at"]) if latest else datetime.now(UTC)
    now = max(datetime.now(UTC), latest_time) + timedelta(days=500)
    expired_at = now - timedelta(days=500)
    actor = "integration-retention-" + uuid.uuid4().hex
    try:
        store.record_audit(actor, "preserve-under-hold", 200, expired_at)
        event_id = store.query_audit_events(limit=1, actor=actor)["events"][0]["id"]
        review_at = now + timedelta(days=30)
        hold = store.create_admin_audit_hold(event_id, "integration-operator", "CI retention hold", now,
            case_reference="CI-RETENTION", notes="Integration coverage", review_at=review_at)
        assert hold["event_id"] == event_id
        assert hold["case_reference"] == "CI-RETENTION"
        assert hold["notes"] == "Integration coverage"
        assert hold["review_at"] == review_at.isoformat()
        assert any(item["event_id"] == event_id
                   for item in store.admin_audit_holds(active_only=True, limit=1000))
        assert store.audit_chain_status()["verified"] is True

        pruned = store.prune_admin_audit_events(365, now=now)
        assert pruned["held_event_id"] == event_id
        assert store.query_audit_events(limit=1000, actor=actor)["events"]

        released = store.release_admin_audit_hold(event_id, "integration-operator", now + timedelta(seconds=1))
        assert released["released_by"] == "integration-operator"
        assert store.admin_audit_holds(active_only=True, limit=1000) == []
        store.prune_admin_audit_events(365, now=now + timedelta(seconds=1))
        assert not store.query_audit_events(limit=1000, actor=actor)["events"]
        assert store.audit_chain_status()["verified"] is True
    finally:
        store.close()
