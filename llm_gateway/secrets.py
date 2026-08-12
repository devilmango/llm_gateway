"""Resolve credentials dynamically from environment variables or rotated secret files."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


class SecretUnavailable(Exception):
    """A configured external secret source could not provide a credential."""


async def resolve_secret(env_name: str | None = None, file_path: str | None = None) -> str:
    """Read the current secret value; files are re-read on every use for live rotation."""
    if file_path:
        try:
            value = await asyncio.to_thread(Path(file_path).expanduser().read_text, encoding="utf-8")
        except OSError as exc:
            raise SecretUnavailable("Configured credential file is unavailable") from exc
        return value.strip()
    return os.getenv(env_name or "", "")
