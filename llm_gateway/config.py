from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .capabilities import provider_capabilities

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
MIN_TENANT_USAGE_RETENTION_DAYS = 35


def validate_tenant_policies(config: dict[str, Any]) -> None:
    tenants = config.get("tenants", config.get("security", {}).get("tenants", {}))
    for tenant_id, tenant in tenants.items():
        if not isinstance(tenant, dict):
            raise TypeError(f"Tenant {tenant_id!r} configuration must be an object")
        for key in ("allowed_routes", "allowed_providers", "allowed_models", "allowed_regions"):
            value = tenant.get(key)
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                raise ValueError(f"Tenant {tenant_id!r} {key} must be a list of strings")
        data_policy = tenant.get("data_policy", {})
        if not isinstance(data_policy, dict):
            raise TypeError(f"Tenant {tenant_id!r} data_policy must be an object")
        retention_days = data_policy.get("usage_retention_days")
        if retention_days is not None:
            if not isinstance(retention_days, int) or isinstance(retention_days, bool):
                raise ValueError(f"Tenant {tenant_id!r} usage_retention_days must be an integer")
            if retention_days < MIN_TENANT_USAGE_RETENTION_DAYS:
                raise ValueError(f"Tenant {tenant_id!r} usage_retention_days must be at least "
                                 f"{MIN_TENANT_USAGE_RETENTION_DAYS} to preserve current-month budget accounting")


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
    for tenant_id, tenant in config.get("tenants", config.get("security", {}).get("tenants", {})).items():
        unknown_routes = set(tenant.get("allowed_routes", [])) - set(config["routes"])
        unknown_providers = set(tenant.get("allowed_providers", [])) - set(config["providers"])
        if unknown_routes:
            raise ValueError(f"Tenant {tenant_id!r} refers to unknown route(s): {', '.join(sorted(unknown_routes))}")
        if unknown_providers:
            raise ValueError(f"Tenant {tenant_id!r} refers to unknown provider(s): {', '.join(sorted(unknown_providers))}")
    for provider in config["providers"].values():
        provider_capabilities(provider)
    validate_tenant_policies(config)
    oidc = config.get("security", {}).get("oidc", {})
    if oidc.get("enabled") and not all(oidc.get(field) for field in ("issuer", "audience", "jwks_url")):
        raise ValueError("Enabled OIDC authentication requires issuer, audience, and jwks_url")
    return config
