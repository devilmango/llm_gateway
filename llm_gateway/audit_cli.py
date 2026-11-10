"""Command-line verification for exported admin audit files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .audit_security import (
    audit_entry_hash,
    canonical_json,
    verify_audit_export_signature,
    verify_audit_manifest_signature,
)


def verify_export(export: dict[str, Any], trusted_public_key_pem: str | None = None) -> tuple[bool, str]:
    events = export.get("events")
    chain = export.get("chain")
    if not isinstance(events, list) or not isinstance(chain, dict):
        return False, "invalid export: expected events and chain fields"
    if any(not isinstance(event, dict) for event in events):
        return False, "invalid export: each audit event must be an object"
    previous = chain.get("anchor_hash")
    previous_id = int(chain.get("anchor_id", 0))
    for event in sorted(events, key=lambda row: int(row.get("id", -1))):
        event_id = int(event.get("id", -1))
        expected = audit_entry_hash(event_id, event["occurred_at"], event["actor"],
            event["action"], event["status_code"], previous)
        if event_id <= previous_id:
            return False, f"audit chain has duplicate or out-of-order event id {event_id}"
        if event.get("previous_hash") != previous or event.get("entry_hash") != expected:
            return False, f"audit chain broken at event id {event_id}"
        previous, previous_id = event["entry_hash"], event_id
    if len(events) != int(chain.get("record_count", -1)):
        return False, "audit chain record count mismatch"
    if previous != chain.get("head_hash") or previous_id != int(chain.get("head_id", -1)):
        return False, "audit chain head mismatch"
    if chain.get("verified") is not True:
        return False, "export reports an unverified audit chain"
    digest = hashlib.sha256(canonical_json(events)).hexdigest()
    if digest != export.get("export_sha256"):
        return False, "export digest mismatch"
    try:
        verify_audit_export_signature(export, trusted_public_key_pem)
    except (ValueError, KeyError, TypeError) as exc:
        return False, str(exc)
    return True, f"verified {len(events)} audit records; head id {previous_id}"


def _archive_object_path(root: Path, key: str) -> Path:
    relative = PurePosixPath(key)
    if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("archive manifest contains an unsafe object key")
    base = root.expanduser().resolve()
    path = (base / Path(*relative.parts)).resolve()
    if base not in path.parents:
        raise ValueError("archive object is outside the archive root")
    return path


def verify_archive(manifest: dict[str, Any], archive_root: Path,
                   trusted_public_key_pem: str) -> tuple[bool, str, dict[str, Any] | None]:
    if manifest.get("format") != "llm-gateway-audit-archive-v1":
        return False, "unsupported audit archive manifest format", None
    try:
        events_path = _archive_object_path(archive_root, str(manifest["events_key"]))
        raw_export = events_path.read_bytes()
        return verify_archive_payload(manifest, raw_export, trusted_public_key_pem)
    except FileNotFoundError as exc:
        return False, f"archive object not found: {exc.filename}", None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return False, f"archive verification failed: {exc}", None


def verify_archive_payload(manifest: dict[str, Any], raw_export: bytes,
                           trusted_public_key_pem: str) -> tuple[bool, str, dict[str, Any] | None]:
    if manifest.get("format") != "llm-gateway-audit-archive-v1":
        return False, "unsupported audit archive manifest format", None
    actual_sha = hashlib.sha256(raw_export).hexdigest()
    if actual_sha != str(manifest.get("data_sha256")):
        return False, "archive data object digest mismatch", None
    try:
        verify_audit_manifest_signature(manifest, trusted_public_key_pem)
        export = json.loads(raw_export)
        if not isinstance(export, dict):
            return False, "archive data object is not a JSON export", None
        valid, message = verify_export(export, trusted_public_key_pem)
        if not valid:
            return False, message, None
        if export.get("export_sha256") != manifest.get("export_sha256"):
            return False, "manifest export digest does not match the signed export", None
        if export.get("chain") != manifest.get("chain"):
            return False, "manifest chain summary does not match the signed export", None
        events = export["events"]
        first_id = min((int(event["id"]) for event in events), default=None)
        last_id = max((int(event["id"]) for event in events), default=None)
        if len(events) != int(manifest.get("event_count", -1)):
            return False, "manifest event count does not match the signed export", None
        if first_id != manifest.get("first_event_id") or last_id != manifest.get("last_event_id"):
            return False, "manifest event ID bounds do not match the signed export", None
        signature = export.get("signature", {})
        if signature.get("key_id") != manifest.get("signature_key_id"):
            return False, "manifest signing key ID does not match the signed export", None
        manifest_signature = manifest.get("manifest_signature", {})
        if manifest_signature.get("key_id") != signature.get("key_id"):
            return False, "manifest and export signatures use different key IDs", None
        if export.get("archive_id", manifest.get("archive_id")) != manifest.get("archive_id"):
            return False, "manifest archive ID does not match the signed export", None
        return True, f"verified archive {manifest.get('archive_id')}: {message}", export
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return False, f"archive verification failed: {exc}", None


def verify_archive_lineage(manifest_file: Path, archive_root: Path,
                           trusted_public_key_pem: str) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    manifest_path = manifest_file.expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read archive manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TypeError("archive manifest must be a JSON object")
    lineage: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    child_head: tuple[int, str] | None = None
    current = manifest
    while True:
        archive_id = str(current.get("archive_id", ""))
        if not archive_id or archive_id in seen:
            raise ValueError("archive lineage contains a missing ID or cycle")
        seen.add(archive_id)
        valid, message, export = verify_archive(current, archive_root, trusted_public_key_pem)
        if not valid:
            raise ValueError(message)
        chain = export["chain"]
        if child_head is not None and (int(chain["head_id"]), chain["head_hash"]) != child_head:
            raise ValueError(f"archive parent head mismatch at archive {archive_id}")
        lineage.append((export, current))
        parent_id = current.get("previous_archive_id")
        if parent_id is None:
            break
        anchor = (int(chain["anchor_id"]), chain["anchor_hash"])
        if (int(current.get("previous_head_id", -1)), current.get("previous_head_hash")) != anchor:
            raise ValueError(f"archive previous-head link mismatch at archive {archive_id}")
        child_head = anchor
        parent_manifest_path = _archive_object_path(archive_root,
            f"audit-archives/{parent_id}/manifest.json")
        try:
            current = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read parent archive {parent_id}: {exc}") from exc
        if not isinstance(current, dict) or current.get("archive_id") != parent_id:
            raise ValueError(f"parent archive manifest ID mismatch for {parent_id}")
    lineage.reverse()
    return lineage, f"verified {len(lineage)} archive segment(s), including chain and lineage"


def restore_archives(lineage: list[tuple[dict[str, Any], dict[str, Any]]], database: Path) -> int:
    database = database.expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("""CREATE TABLE IF NOT EXISTS restored_archives (
            archive_id TEXT PRIMARY KEY, restored_at TEXT NOT NULL, manifest_json TEXT NOT NULL,
            export_json TEXT NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS restored_audit_events (
            archive_id TEXT NOT NULL, event_id INTEGER NOT NULL, occurred_at TEXT NOT NULL,
            actor TEXT NOT NULL, action TEXT NOT NULL, status_code INTEGER NOT NULL,
            previous_hash TEXT NOT NULL, entry_hash TEXT NOT NULL,
            PRIMARY KEY (archive_id, event_id),
            FOREIGN KEY (archive_id) REFERENCES restored_archives(archive_id)
        )""")
        connection.execute("BEGIN IMMEDIATE")
        inserted = 0
        for export, manifest in lineage:
            archive_id = manifest["archive_id"]
            if connection.execute("SELECT 1 FROM restored_archives WHERE archive_id=?", (archive_id,)).fetchone():
                raise ValueError(f"archive {archive_id} is already restored in {database}")
            connection.execute("INSERT INTO restored_archives VALUES (?,?,?,?)", (
                archive_id, datetime.now(UTC).isoformat(), json.dumps(manifest, sort_keys=True),
                json.dumps(export, sort_keys=True)))
            rows = [(archive_id, int(event["id"]), event["occurred_at"], event["actor"],
                     event["action"], int(event["status_code"]), event["previous_hash"], event["entry_hash"])
                    for event in sorted(export["events"], key=lambda row: int(row["id"]))]
            connection.executemany("INSERT INTO restored_audit_events VALUES (?,?,?,?,?,?,?,?)", rows)
            inserted += len(rows)
    return inserted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-gateway-audit",
        description="Verify the integrity and signature of an LLM Gateway admin audit export")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify an exported JSON audit file")
    verify.add_argument("export_file", type=Path)
    verify.add_argument("--public-key", type=Path,
        help="trusted RSA public key PEM (recommended; pins signer identity)")
    verify.add_argument("--key-id", help="require a particular signing key ID")
    archive_verify = subparsers.add_parser("verify-archive", help="verify an archive manifest and its objects")
    archive_verify.add_argument("manifest_file", type=Path)
    archive_verify.add_argument("--archive-root", type=Path, required=True,
        help="local object-store root containing the manifest's events_key")
    archive_verify.add_argument("--public-key", type=Path, required=True,
        help="trusted RSA public key PEM")
    archive_verify.add_argument("--key-id", help="require a particular signing key ID")
    restore = subparsers.add_parser("restore", help="verify and restore archive events to a separate SQLite store")
    restore.add_argument("manifest_file", type=Path)
    restore.add_argument("--archive-root", type=Path, required=True,
        help="local object-store root containing the manifest's events_key")
    restore.add_argument("--database", type=Path, required=True,
        help="separate SQLite investigation database to create or extend")
    restore.add_argument("--public-key", type=Path, required=True,
        help="trusted RSA public key PEM")
    restore.add_argument("--key-id", help="require a particular signing key ID")
    args = parser.parse_args(argv)
    try:
        if args.command in {"verify-archive", "restore"}:
            trusted_key = args.public_key.read_text(encoding="utf-8")
            lineage, message = verify_archive_lineage(args.manifest_file, args.archive_root, trusted_key)
            if args.key_id and any(row[1].get("signature_key_id") != args.key_id for row in lineage):
                raise ValueError(f"archive lineage contains a signing key ID different from {args.key_id!r}")
            if args.command == "restore":
                database_path = args.database.expanduser().resolve()
                archive_paths = {args.manifest_file.expanduser().resolve()}
                for _, archive_manifest in lineage:
                    archive_paths.add(_archive_object_path(args.archive_root, archive_manifest["manifest_key"]))
                    archive_paths.add(_archive_object_path(args.archive_root, archive_manifest["events_key"]))
                if database_path in archive_paths:
                    raise ValueError("restore database must be separate from the archive files")
                count = restore_archives(lineage, database_path)
                message += f"; restored {count} records to {database_path}"
            print(message)
            return 0
        export = json.loads(args.export_file.read_text(encoding="utf-8"))
        if not isinstance(export, dict):
            raise TypeError("audit export must be a JSON object")
        trusted_key = args.public_key.read_text(encoding="utf-8") if args.public_key else None
        signature = export.get("signature", {})
        if not isinstance(signature, dict):
            raise TypeError("audit export signature must be an object")
        if args.key_id and signature.get("key_id") != args.key_id:
            raise ValueError(f"unexpected signing key ID: {signature.get('key_id')!r}")
        valid, message = verify_export(export, trusted_key)
        if not args.public_key and valid:
            message += "; signer identity was not pinned (embedded key only)"
        print(message)
        return 0 if valid else 1
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
