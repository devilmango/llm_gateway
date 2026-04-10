from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("GATEWAY_CONFIG", "config.yaml"))
    with config_path.open("r", encoding="utf-8") as handle:
        config = _expand_env(yaml.safe_load(handle) or {})
    for key in ("server", "security", "providers", "routes"):
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    if not config["providers"] or not config["routes"]:
        raise ValueError("At least one provider and route must be configured")
    return config
