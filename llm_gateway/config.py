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
    governance = config.get("tool_governance", {})
    if not isinstance(governance, dict):
        raise TypeError("tool_governance must be an object")
    for limit_name in ("max_tool_calls_per_request", "max_tool_calls_per_minute", "execution_timeout_seconds",
                       "approval_ttl_seconds"):
        limit = governance.get(limit_name)
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool)
                or limit < (1 if limit_name in {"execution_timeout_seconds", "approval_ttl_seconds"} else 0)):
            qualifier = "positive" if limit_name in {"execution_timeout_seconds", "approval_ttl_seconds"} else "non-negative"
            raise ValueError(f"tool_governance.{limit_name} must be a {qualifier} integer")
    for names_key in ("allowed_tools", "require_approval_tools"):
        names = governance.get(names_key)
        if names is not None and (not isinstance(names, list)
                or any(not isinstance(item, str) or not item for item in names)):
            raise ValueError(f"tool_governance.{names_key} must be a list of names")
    approval_webhooks = governance.get("approval_webhooks", {})
    if not isinstance(approval_webhooks, dict):
        raise TypeError("tool_governance.approval_webhooks must be an application map")
    for application, webhook in approval_webhooks.items():
        if (not application or not isinstance(webhook, dict) or not isinstance(webhook.get("url"), str)
                or not webhook["url"].startswith(("https://", "http://"))):
            raise ValueError("tool_governance.approval_webhooks entries require an HTTP(S) url")
        secret_env = webhook.get("secret_env")
        if secret_env is not None and (not isinstance(secret_env, str) or not secret_env):
            raise ValueError("tool_governance.approval_webhooks secret_env must be a non-empty string")
    tenants = config.get("tenants", config.get("security", {}).get("tenants", {}))
    for tenant_id, tenant in tenants.items():
        if not isinstance(tenant, dict):
            raise TypeError(f"Tenant {tenant_id!r} configuration must be an object")
        for key in ("allowed_routes", "allowed_providers", "allowed_models", "allowed_regions",
                    "allowed_tools", "require_approval_tools"):
            value = tenant.get(key)
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                raise ValueError(f"Tenant {tenant_id!r} {key} must be a list of strings")
        for policy_name, policy in [("default_policy", tenant.get("default_policy", {})),
                                    *[(f"applications.{name}", value)
                                      for name, value in tenant.get("applications", {}).items()]]:
            if not isinstance(policy, dict):
                raise TypeError(f"Tenant {tenant_id!r} {policy_name} must be an object")
            for names_key in ("allowed_tools", "require_approval_tools"):
                names = policy.get(names_key)
                if names is not None and (not isinstance(names, list)
                        or any(not isinstance(item, str) or not item for item in names)):
                    raise ValueError(f"Tenant {tenant_id!r} {policy_name}.{names_key} must be a list of names")
            for limit_name in ("max_tool_calls_per_request", "max_tool_calls_per_minute"):
                limit = policy.get(limit_name)
                if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
                    raise ValueError(f"Tenant {tenant_id!r} {policy_name}.{limit_name} must be a non-negative integer")
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
    for provider_name, provider in config["providers"].items():
        request_id_header = provider.get("request_id_header")
        if request_id_header is not None and not isinstance(request_id_header, str):
            raise TypeError(f"Provider {provider_name!r} request_id_header must be a string")
        if request_id_header and not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", request_id_header):
            raise ValueError(f"Provider {provider_name!r} request_id_header must be an HTTP header name")
        body_path = provider.get("request_id_body_field")
        if body_path is not None and (not isinstance(body_path, str)
                or any(not part for part in body_path.split("."))):
            raise ValueError(f"Provider {provider_name!r} request_id_body_field must be a dot-separated path")
        provider_capabilities(provider)
    for route_name, route in config["routes"].items():
        shadow = route.get("shadow")
        if shadow is None:
            continue
        if not isinstance(shadow, dict):
            raise TypeError(f"Route {route_name!r} shadow must be an object")
        if shadow.get("provider") not in config["providers"] or not isinstance(shadow.get("model"), str) or not shadow["model"]:
            raise ValueError(f"Route {route_name!r} shadow must reference a configured provider and model")
        if (shadow["provider"], shadow["model"]) == (route.get("provider"), route.get("model")):
            raise ValueError(f"Route {route_name!r} shadow candidate must differ from its primary model")
        sampling = shadow.get("traffic_percent", 0)
        if isinstance(sampling, bool) or not isinstance(sampling, (int, float)) or not 0 <= sampling <= 100:
            raise ValueError(f"Route {route_name!r} shadow.traffic_percent must be between 0 and 100")
        gates = shadow.get("promotion", {})
        if not isinstance(gates, dict):
            raise TypeError(f"Route {route_name!r} shadow.promotion must be an object")
        for key in ("min_samples", "max_quality_loss", "max_cost_regression_percent",
                    "max_latency_regression_percent", "max_failure_rate"):
            value = gates.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"Route {route_name!r} shadow.promotion.{key} must be non-negative")
        if gates.get("min_samples") is not None and int(gates["min_samples"]) != gates["min_samples"]:
            raise ValueError(f"Route {route_name!r} shadow.promotion.min_samples must be an integer")
        if gates.get("min_samples") is not None and gates["min_samples"] < 1:
            raise ValueError(f"Route {route_name!r} shadow.promotion.min_samples must be positive")
        for key in ("max_quality_loss", "max_failure_rate"):
            if gates.get(key) is not None and gates[key] > 1:
                raise ValueError(f"Route {route_name!r} shadow.promotion.{key} must be between 0 and 1")
    validate_tenant_policies(config)
    oidc = config.get("security", {}).get("oidc", {})
    if oidc.get("enabled") and not all(oidc.get(field) for field in ("issuer", "audience", "jwks_url")):
        raise ValueError("Enabled OIDC authentication requires issuer, audience, and jwks_url")
    return config
