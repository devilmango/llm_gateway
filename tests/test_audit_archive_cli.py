from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from llm_gateway.artifacts import S3ArtifactStore, create_artifact_store
from llm_gateway.audit_archives import (
    create_admin_audit_archive,
    verify_admin_audit_archive,
)
from llm_gateway.audit_cli import main as audit_cli_main
from llm_gateway.audit_security import sign_audit_export
from llm_gateway.storage import create_usage_store


def _signed_archive(tmp_path: Path):
    private = subprocess.run(["openssl", "genrsa", "2048"], check=True, capture_output=True).stdout.decode()
    public = subprocess.run(["openssl", "rsa", "-pubout"], input=private.encode(),
                            check=True, capture_output=True).stdout.decode()
    store = create_usage_store(str(tmp_path / "source.sqlite3"))
    store.record_audit("investigator", "sample-event", 200)
    export = sign_audit_export(store.export_audit_events(), key_id="audit-test-key", private_key_pem=private)
    archive_root = tmp_path / "objects"
    manifest_path = archive_root / "audit-archives" / "sample" / "manifest.json"
    events_path = archive_root / "audit-archives" / "sample" / "events.json"
    manifest_path.parent.mkdir(parents=True)
    data = json.dumps(export, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    manifest = {"format": "llm-gateway-audit-archive-v1", "archive_id": "sample",
        "events_key": "audit-archives/sample/events.json", "manifest_key": "audit-archives/sample/manifest.json",
        "event_count": len(export["events"]),
        "first_event_id": min((row["id"] for row in export["events"]), default=None),
        "last_event_id": max((row["id"] for row in export["events"]), default=None),
        "chain": export["chain"], "export_sha256": export["export_sha256"],
        "data_sha256": hashlib.sha256(data).hexdigest(), "signature_key_id": "audit-test-key"}
    events_path.write_bytes(data)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    public_path = tmp_path / "trusted-public.pem"
    public_path.write_text(public, encoding="utf-8")
    return archive_root, manifest_path, public_path, manifest, events_path


def test_audit_archive_cli_verifies_and_restores_to_investigation_database(tmp_path, capsys):
    root, manifest, public, archive, _ = _signed_archive(tmp_path)
    assert audit_cli_main(["verify-archive", str(manifest), "--archive-root", str(root),
                           "--public-key", str(public), "--key-id", "audit-test-key"]) == 0
    assert "verified 1 archive segment" in capsys.readouterr().out

    database = tmp_path / "investigation.sqlite3"
    assert audit_cli_main(["restore", str(manifest), "--archive-root", str(root), "--database", str(database),
                           "--public-key", str(public)]) == 0
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT event_id,action FROM restored_audit_events WHERE archive_id='sample'").fetchall()
    assert rows == [(1, "sample-event")]
    assert audit_cli_main(["restore", str(manifest), "--archive-root", str(root), "--database", str(database),
                           "--public-key", str(public)]) == 1

    events_path = root / archive["events_key"]
    events_path.write_bytes(events_path.read_bytes() + b" ")
    assert audit_cli_main(["verify-archive", str(manifest), "--archive-root", str(root),
                           "--public-key", str(public)]) == 1
    assert "digest mismatch" in capsys.readouterr().err


def test_audit_archive_retention_does_not_expire_before_manifest_deadline(tmp_path):
    store = create_usage_store(str(tmp_path / "archive-retention.sqlite3"))
    future = datetime.now(UTC) + timedelta(days=2)
    past = datetime.now(UTC) - timedelta(seconds=1)
    base = {"created_at": datetime.now(UTC).isoformat(), "created_by": "operator",
        "events_key": "audit-archives/a/events.json", "manifest_key": "audit-archives/a/manifest.json",
        "event_count": 0, "data_sha256": "digest"}
    future_manifest = {**base, "archive_id": "future", "retention_until": future.isoformat()}
    expired_manifest = {**base, "archive_id": "expired", "retention_until": past.isoformat()}
    parent_manifest = {**base, "archive_id": "parent", "retention_until": past.isoformat()}
    child_manifest = {**base, "archive_id": "child", "retention_until": past.isoformat(),
                      "previous_archive_id": "parent"}
    store.record_admin_audit_archive(future_manifest)
    store.record_admin_audit_archive(expired_manifest)
    store.record_admin_audit_archive(parent_manifest)
    store.record_admin_audit_archive(child_manifest)

    candidates = store.expired_admin_audit_archives()
    assert {row["archive_id"] for row in candidates} == {"expired", "child"}
    assert store.delete_expired_admin_audit_archive("future") is False
    assert store.delete_expired_admin_audit_archive("expired") is True
    assert store.delete_expired_admin_audit_archive("parent") is False
    assert store.delete_expired_admin_audit_archive("child") is True
    assert store.delete_expired_admin_audit_archive("parent") is True
    assert store.get_admin_audit_archive("future") is not None
    assert store.get_admin_audit_archive("expired") is None


def test_incremental_archive_links_and_verifies_complete_parent_lineage(tmp_path, monkeypatch):
    private = subprocess.run(["openssl", "genrsa", "2048"], check=True,
                             capture_output=True).stdout.decode()
    public = subprocess.run(["openssl", "rsa", "-pubout"], input=private.encode(),
                            check=True, capture_output=True).stdout.decode()
    monkeypatch.setenv("ARCHIVE_SIGNING_KEY", private)
    cfg = {"server": {"audit_signing_key_id": "incremental-key",
                       "audit_signing_private_key_env": "ARCHIVE_SIGNING_KEY"},
           "object_storage": {"audit_archive_retention_days": 90}}
    store = create_usage_store(str(tmp_path / "incremental.sqlite3"))
    artifacts = create_artifact_store({"backend": "local", "directory": str(tmp_path / "objects")},
                                     store.database)
    store.record_audit("operator", "before-first-archive", 200)
    first = create_admin_audit_archive(store, artifacts, cfg, "operator")
    store.record_audit("operator", "after-first-archive", 200)
    second = create_admin_audit_archive(store, artifacts, cfg, "operator")

    assert first["incremental"] is False
    assert second["incremental"] is True
    assert second["previous_archive_id"] == first["archive_id"]
    assert second["previous_head_id"] == first["chain"]["head_id"]
    verified = verify_admin_audit_archive(store, artifacts, second["archive_id"])
    assert verified["verification"]["verified"] is True
    assert verified["verification"]["archives_verified"] == 2
    assert {event["action"] for event in verified["export"]["events"]} == {
        "after-first-archive", f"audit_archive_created:{first['archive_id']}"
    }

    public_path = tmp_path / "incremental-public.pem"
    public_path.write_text(public, encoding="utf-8")
    latest_manifest_path = Path(tmp_path / "objects") / second["manifest_key"]
    assert audit_cli_main(["verify-archive", str(latest_manifest_path), "--archive-root",
        str(tmp_path / "objects"), "--public-key", str(public_path)]) == 0
    investigation_db = tmp_path / "incremental-investigation.sqlite3"
    assert audit_cli_main(["restore", str(latest_manifest_path), "--archive-root", str(tmp_path / "objects"),
        "--database", str(investigation_db), "--public-key", str(public_path)]) == 0
    with sqlite3.connect(investigation_db) as connection:
        restored = connection.execute("SELECT COUNT(*) FROM restored_audit_events").fetchone()[0]
    assert restored == first["event_count"] + second["event_count"]

    parent_path = Path(tmp_path / "objects") / first["events_key"]
    parent_path.write_bytes(parent_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_admin_audit_archive(store, artifacts, second["archive_id"])

def test_s3_artifact_store_applies_object_lock_only_to_audit_archives(monkeypatch):
    calls = []

    class FakeS3:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    fake_boto = ModuleType("boto3")
    fake_boto.client = lambda *args, **kwargs: FakeS3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto)
    store = S3ArtifactStore("audit-bucket", audit_archive_retention_days=14,
                            audit_archive_object_lock_mode="COMPLIANCE")

    store.put("audit-archives/archive/events.json", b"audit")
    store.put("jobs/output.json", b"job")

    assert calls[0]["ObjectLockMode"] == "COMPLIANCE"
    assert calls[0]["ObjectLockRetainUntilDate"] > datetime.now(UTC) + timedelta(days=13)
    assert "ObjectLockMode" not in calls[1]

