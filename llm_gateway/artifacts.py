"""Artifact storage for large asynchronous job outputs."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any


class ArtifactStore:
    def put(self, key: str, body: bytes, content_type: str = "application/json") -> None:
        raise NotImplementedError

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        relative = PurePosixPath(key)
        if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
            raise ValueError("Invalid artifact key")
        path = (self.directory / Path(*relative.parts)).resolve()
        if self.directory not in path.parents:
            raise ValueError("Invalid artifact key")
        return path

    def put(self, key: str, body: bytes, content_type: str = "application/json") -> None:
        del content_type
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(body)
        os.replace(temporary, path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3ArtifactStore(ArtifactStore):
    def __init__(self, bucket: str, prefix: str = "llm-gateway", region: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3 artifact storage requires the optional llm-gateway[s3] dependency") from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", region_name=region)

    def _key(self, key: str) -> str:
        relative = PurePosixPath(key)
        if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
            raise ValueError("Invalid artifact key")
        return f"{self.prefix}/{relative.as_posix()}" if self.prefix else relative.as_posix()

    def put(self, key: str, body: bytes, content_type: str = "application/json") -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=body, ContentType=content_type)

    def get(self, key: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(key) from exc
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


def create_artifact_store(config: dict[str, Any], database: str) -> ArtifactStore:
    backend = config.get("backend", "local")
    if backend == "s3":
        bucket = config.get("bucket")
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("object_storage.bucket is required for S3 storage")
        return S3ArtifactStore(bucket, config.get("prefix", "llm-gateway"), config.get("region"))
    if backend != "local":
        raise ValueError("object_storage.backend must be 'local' or 's3'")
    directory = config.get("directory")
    if not directory:
        shared_database = database.startswith(("postgres://", "postgresql://"))
        directory = f"{database}.artifacts" if database != ":memory:" and not shared_database else "./gateway-artifacts"
    return LocalArtifactStore(str(directory))
