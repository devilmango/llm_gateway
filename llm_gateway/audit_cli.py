"""Command-line verification for exported admin audit files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .audit_security import (
    audit_entry_hash,
    canonical_json,
    verify_audit_export_signature,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-gateway-audit",
        description="Verify the integrity and signature of an LLM Gateway admin audit export")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify an exported JSON audit file")
    verify.add_argument("export_file", type=Path)
    verify.add_argument("--public-key", type=Path,
        help="trusted RSA public key PEM (recommended; pins signer identity)")
    verify.add_argument("--key-id", help="require a particular signing key ID")
    args = parser.parse_args(argv)
    try:
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
