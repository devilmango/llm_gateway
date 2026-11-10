"""Create, retrieve, and verify signed admin-audit archive segments."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from .audit_cli import verify_export
from .audit_security import canonical_json, configured_audit_signature

logger = logging.getLogger(__name__)


def _json_object(body: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _verify_archive_objects(events_data: bytes, manifest_data: bytes,
                            expected: dict[str, Any] | None = None,
                            trusted_public_key: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _json_object(manifest_data, "archive manifest")
    if expected is not None:
        expected_manifest = expected.get("manifest", {})
        if expected_manifest != manifest:
            raise ValueError("stored archive manifest does not match its database record")
        for field in ("archive_id", "events_key", "manifest_key", "event_count", "data_sha256"):
            if expected.get(field) != manifest.get(field):
                raise ValueError(f"archive manifest {field} does not match its database record")
    if hashlib.sha256(events_data).hexdigest() != manifest.get("data_sha256"):
        raise ValueError("archive object SHA-256 does not match the manifest")
    export = _json_object(events_data, "archive export")
    if manifest.get("archive_id") != export.get("archive_id", manifest.get("archive_id")):
        raise ValueError("archive ID does not match its export")
    if manifest.get("chain") != export.get("chain"):
        raise ValueError("archive chain does not match its manifest")
    if manifest.get("export_sha256") != export.get("export_sha256"):
        raise ValueError("archive export digest does not match its manifest")
    events = export.get("events")
    if not isinstance(events, list) or len(events) != manifest.get("event_count"):
        raise ValueError("archive event count does not match its manifest")
    ids = [event.get("id") for event in events if isinstance(event, dict)]
    if len(ids) != len(events):
        raise ValueError("archive contains malformed audit events")
    if min(ids, default=None) != manifest.get("first_event_id"):
        raise ValueError("archive first event ID does not match its manifest")
    if max(ids, default=None) != manifest.get("last_event_id"):
        raise ValueError("archive last event ID does not match its manifest")
    signature = export.get("signature")
    if not isinstance(signature, dict) or signature.get("key_id") != manifest.get("signature_key_id"):
        raise ValueError("archive signing key ID does not match its manifest")
    valid, message = verify_export(export, trusted_public_key)
    if not valid:
        raise ValueError(message)
    return export, manifest


def verify_admin_audit_archive(store: Any, artifacts: Any, archive_id: str,
                               trusted_public_key: str | None = None) -> dict[str, Any]:
    """Fetch and verify an archive plus every incremental parent archive."""
    current_id: str | None = archive_id
    expected_parent_head: tuple[int, str] | None = None
    seen: set[str] = set()
    verified = 0
    requested_export: dict[str, Any] | None = None
    requested_manifest: dict[str, Any] | None = None
    while current_id is not None:
        if current_id in seen:
            raise ValueError("archive lineage contains a cycle")
        seen.add(current_id)
        record = store.admin_audit_archive(current_id)
        if record is None:
            raise ValueError(f"parent archive {current_id!r} was not found")
        events_data = artifacts.get(record["events_key"])
        manifest_data = artifacts.get(record["manifest_key"])
        export, manifest = _verify_archive_objects(
            events_data, manifest_data, record, trusted_public_key)
        chain = export["chain"]
        if (expected_parent_head is not None
                and (int(chain["head_id"]), chain["head_hash"]) != expected_parent_head):
            raise ValueError("archive parent head does not match the child archive link")
        if requested_export is None:
            requested_export, requested_manifest = export, manifest
        verified += 1
        parent_id = manifest.get("previous_archive_id")
        if parent_id is None:
            current_id = None
            continue
        anchor = (int(chain["anchor_id"]), chain["anchor_hash"])
        if manifest.get("previous_head_id") != anchor[0] or manifest.get("previous_head_hash") != anchor[1]:
            raise ValueError("archive previous-head fields do not match its chain anchor")
        expected_parent_head = anchor
        current_id = str(parent_id)
    assert requested_export is not None and requested_manifest is not None
    return {"export": requested_export, "manifest": requested_manifest,
            "verification": {"verified": True, "object_hash_valid": True,
                "signature_valid": True, "chain_valid": True, "lineage_valid": True,
                "signer_identity_pinned": trusted_public_key is not None,
                "archives_verified": verified}}


def create_admin_audit_archive(store: Any, artifacts: Any, config: dict[str, Any],
                               actor: str, lease: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a full initial archive or a verified incremental segment."""
    source_export = store.export_audit_events()
    if not source_export["chain"]["verified"]:
        raise ValueError(f"admin audit chain integrity check failed at event "
                         f"{source_export['chain']['first_invalid_id']}")

    previous_records = store.admin_audit_archives(limit=1)
    previous_id = None
    previous_head_id = None
    previous_head_hash = None
    archived_export = source_export
    if previous_records:
        previous = verify_admin_audit_archive(
            store, artifacts, previous_records[0]["archive_id"])
        previous_export = previous["export"]
        previous_chain = previous_export["chain"]
        previous_id = previous["manifest"]["archive_id"]
        previous_head_id = int(previous_chain["head_id"])
        previous_head_hash = previous_chain["head_hash"]
        source_chain = source_export["chain"]
        if int(source_chain["anchor_id"]) > previous_head_id:
            raise ValueError("audit history was pruned before it was incrementally archived")
        if (int(source_chain["anchor_id"]) == previous_head_id
                and source_chain["anchor_hash"] != previous_head_hash):
            raise ValueError("current audit anchor does not match the previous archive")
        delta_events = [event for event in source_export["events"] if int(event["id"]) > previous_head_id]
        if delta_events:
            first = min(delta_events, key=lambda event: int(event["id"]))
            if first["previous_hash"] != previous_head_hash:
                raise ValueError("new audit events do not continue the previous archive hash chain")
        delta_chain = {**source_chain, "anchor_id": previous_head_id,
            "anchor_hash": previous_head_hash, "record_count": len(delta_events),
            "verified": True, "first_invalid_id": None}
        archived_export = {"format": source_export["format"], "exported_at": source_export["exported_at"],
            "chain": delta_chain, "export_sha256": hashlib.sha256(canonical_json(delta_events)).hexdigest(),
            "events": delta_events}

    signed_export = configured_audit_signature(archived_export, config)
    archive_id = uuid.uuid4().hex
    created_at = datetime.now(UTC).isoformat()
    events_key = f"audit-archives/{archive_id}/events.json"
    manifest_key = f"audit-archives/{archive_id}/manifest.json"
    data = canonical_json({**signed_export, "archive_id": archive_id})
    events = archived_export["events"]
    signature = signed_export["signature"]
    manifest = {"format": "llm-gateway-audit-archive-v1", "archive_id": archive_id,
        "created_at": created_at, "created_by": actor, "events_key": events_key,
        "manifest_key": manifest_key, "event_count": len(events),
        "first_event_id": min((event["id"] for event in events), default=None),
        "last_event_id": max((event["id"] for event in events), default=None),
        "chain": archived_export["chain"], "export_sha256": archived_export["export_sha256"],
        "data_sha256": hashlib.sha256(data).hexdigest(), "signature_key_id": signature["key_id"],
        "previous_archive_id": previous_id, "previous_head_id": previous_head_id,
        "previous_head_hash": previous_head_hash, "incremental": previous_id is not None}
    archive_config = config.get("object_storage", {})
    retention_days = archive_config.get("audit_archive_retention_days")
    manifest["retention_until"] = ((datetime.now(UTC) + timedelta(days=int(retention_days))).isoformat()
        if retention_days else None)
    manifest["object_lock_mode"] = archive_config.get("audit_archive_object_lock_mode")
    manifest_data = canonical_json(manifest)
    written: list[str] = []
    try:
        artifacts.put(events_key, data, "application/json")
        written.append(events_key)
        artifacts.put(manifest_key, manifest_data, "application/json")
        written.append(manifest_key)
        store.record_admin_audit_archive(manifest, lease=lease)
    except Exception:
        for key in reversed(written):
            try:
                artifacts.delete(key)
            except Exception:
                logger.exception("failed to remove incomplete audit archive object key=%s", key)
        raise
    return manifest
