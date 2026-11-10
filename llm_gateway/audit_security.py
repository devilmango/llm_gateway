"""Signing and verification helpers for portable admin-audit exports."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def audit_entry_hash(event_id: int, occurred_at: str, actor: str, action: str,
                     status_code: int, previous_hash: str) -> str:
    payload = {"id": event_id, "occurred_at": occurred_at, "actor": actor,
        "action": action, "status_code": int(status_code), "previous_hash": previous_hash}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def audit_signature_payload(export: dict[str, Any], key_id: str) -> bytes:
    return canonical_json({"format": export["format"], "exported_at": export["exported_at"],
        "chain": export["chain"], "export_sha256": export["export_sha256"],
        "signature_algorithm": "RSA-SHA256", "key_id": key_id})


def audit_manifest_signature_payload(manifest: dict[str, Any], key_id: str) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_signature"}
    return canonical_json({"manifest": unsigned, "signature_algorithm": "RSA-SHA256", "key_id": key_id})


def _write_private_key(directory: str, pem: str | bytes) -> str:
    raw = pem.encode("utf-8") if isinstance(pem, str) else pem
    path = os.path.join(directory, "private.pem")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
    try:
        subprocess.run(["openssl", "rsa", "-in", path, "-check", "-noout"],
            check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Audit signing key must be an unencrypted RSA private key in PEM format") from exc
    return path


def sign_audit_export(export: dict[str, Any], *, key_id: str, private_key_pem: str | bytes) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="llm-gateway-audit-key-") as directory:
        private_path = _write_private_key(directory, private_key_pem)
        public_pem = subprocess.run(["openssl", "rsa", "-in", private_path, "-pubout"],
            check=True, capture_output=True).stdout.decode("ascii")
        signature = subprocess.run(["openssl", "dgst", "-sha256", "-sign", private_path],
            input=audit_signature_payload(export, key_id), check=True, capture_output=True).stdout
    signed = dict(export)
    signed["signature"] = {"algorithm": "RSA-SHA256", "key_id": key_id,
        "public_key_pem": public_pem, "value": base64.b64encode(signature).decode("ascii")}
    return signed


def sign_audit_manifest(manifest: dict[str, Any], *, key_id: str,
                        private_key_pem: str | bytes) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="llm-gateway-audit-manifest-key-") as directory:
        private_path = _write_private_key(directory, private_key_pem)
        public_pem = subprocess.run(["openssl", "rsa", "-in", private_path, "-pubout"],
            check=True, capture_output=True).stdout.decode("ascii")
        signature = subprocess.run(["openssl", "dgst", "-sha256", "-sign", private_path],
            input=audit_manifest_signature_payload(manifest, key_id), check=True, capture_output=True).stdout
    signed = dict(manifest)
    signed["manifest_signature"] = {"algorithm": "RSA-SHA256", "key_id": key_id,
        "public_key_pem": public_pem, "value": base64.b64encode(signature).decode("ascii")}
    return signed


def configured_audit_signature(export: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    server = config.get("server", {})
    env_name = server.get("audit_signing_private_key_env")
    key_id = server.get("audit_signing_key_id")
    if not env_name or not key_id:
        raise ValueError("Audit export signing is not configured")
    key_pem = os.getenv(str(env_name))
    if not key_pem:
        raise ValueError(f"Audit signing key environment variable {env_name!r} is empty")
    try:
        return sign_audit_export(export, key_id=str(key_id), private_key_pem=key_pem)
    except OSError as exc:
        raise ValueError("OpenSSL is required to sign audit exports") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Audit export signing failed; check the configured RSA key") from exc


def configured_audit_manifest_signature(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    server = config.get("server", {})
    env_name = server.get("audit_signing_private_key_env")
    key_id = server.get("audit_signing_key_id")
    if not env_name or not key_id:
        raise ValueError("Audit manifest signing is not configured")
    key_pem = os.getenv(str(env_name))
    if not key_pem:
        raise ValueError(f"Audit signing key environment variable {env_name!r} is empty")
    try:
        return sign_audit_manifest(manifest, key_id=str(key_id), private_key_pem=key_pem)
    except OSError as exc:
        raise ValueError("OpenSSL is required to sign audit manifests") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Audit manifest signing failed; check the configured RSA key") from exc


def verify_audit_export_signature(export: dict[str, Any], trusted_public_key_pem: str | bytes | None = None) -> None:
    signature = export.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "RSA-SHA256":
        raise ValueError("Export has no supported RSA-SHA256 signature")
    key_pem = (trusted_public_key_pem if trusted_public_key_pem is not None
               else signature.get("public_key_pem"))
    if not key_pem:
        raise ValueError("No trusted public key was supplied and export has no public key")
    raw = key_pem.encode("utf-8") if isinstance(key_pem, str) else key_pem
    try:
        with tempfile.TemporaryDirectory(prefix="llm-gateway-audit-verify-") as directory:
            public_path = os.path.join(directory, "public.pem")
            sig_path = os.path.join(directory, "signature.bin")
            with open(public_path, "wb") as stream:
                stream.write(raw)
            with open(sig_path, "wb") as stream:
                stream.write(base64.b64decode(signature["value"], validate=True))
            subprocess.run(["openssl", "rsa", "-pubin", "-in", public_path, "-noout"], check=True, capture_output=True)
            if trusted_public_key_pem is not None and signature.get("public_key_pem"):
                embedded_path = os.path.join(directory, "embedded.pem")
                with open(embedded_path, "w", encoding="utf-8") as stream:
                    stream.write(signature["public_key_pem"])
                trusted_canonical = subprocess.run(["openssl", "rsa", "-pubin", "-in", public_path,
                    "-pubout"], check=True, capture_output=True).stdout
                embedded_canonical = subprocess.run(["openssl", "rsa", "-pubin", "-in", embedded_path,
                    "-pubout"], check=True, capture_output=True).stdout
                if embedded_canonical != trusted_canonical:
                    raise ValueError("Export signing key does not match the trusted public key")
            subprocess.run(["openssl", "dgst", "-sha256", "-verify", public_path,
                "-signature", sig_path], input=audit_signature_payload(export, str(signature["key_id"])),
                check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        raise ValueError("Audit export signature verification failed") from exc


def verify_audit_manifest_signature(manifest: dict[str, Any],
                                    trusted_public_key_pem: str | bytes | None = None) -> None:
    signature = manifest.get("manifest_signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "RSA-SHA256":
        raise ValueError("Archive manifest has no supported RSA-SHA256 signature")
    key_pem = trusted_public_key_pem if trusted_public_key_pem is not None else signature.get("public_key_pem")
    if not key_pem:
        raise ValueError("No trusted public key was supplied and manifest has no public key")
    raw = key_pem.encode("utf-8") if isinstance(key_pem, str) else key_pem
    try:
        with tempfile.TemporaryDirectory(prefix="llm-gateway-audit-manifest-verify-") as directory:
            public_path = os.path.join(directory, "public.pem")
            signature_path = os.path.join(directory, "signature.bin")
            with open(public_path, "wb") as stream:
                stream.write(raw)
            with open(signature_path, "wb") as stream:
                stream.write(base64.b64decode(signature["value"], validate=True))
            subprocess.run(["openssl", "rsa", "-pubin", "-in", public_path, "-noout"],
                check=True, capture_output=True)
            embedded = signature.get("public_key_pem")
            if trusted_public_key_pem is not None and embedded:
                embedded_path = os.path.join(directory, "embedded.pem")
                with open(embedded_path, "w", encoding="utf-8") as stream:
                    stream.write(embedded)
                pinned = subprocess.run(["openssl", "rsa", "-pubin", "-in", public_path, "-pubout"],
                    check=True, capture_output=True).stdout
                actual = subprocess.run(["openssl", "rsa", "-pubin", "-in", embedded_path, "-pubout"],
                    check=True, capture_output=True).stdout
                if actual != pinned:
                    raise ValueError("Manifest signing key does not match the trusted public key")
            subprocess.run(["openssl", "dgst", "-sha256", "-verify", public_path,
                "-signature", signature_path], input=audit_manifest_signature_payload(
                    manifest, str(signature["key_id"])),
                check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        raise ValueError("Archive manifest signature verification failed") from exc
