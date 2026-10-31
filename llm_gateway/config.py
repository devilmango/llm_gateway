from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .capabilities import provider_capabilities
from .webhook_security import validate_webhook_url

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
MIN_TENANT_USAGE_RETENTION_DAYS = 35


def validate_tenant_policies(config: dict[str, Any]) -> None:
    server = config.get("server", {})
    for key, default in (("async_embedding_threshold", 256), ("async_job_lease_seconds", 300),
                         ("async_job_concurrency", 4), ("async_job_max_attempts", 5)):
        value = server.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"server.{key} must be a positive integer")
    for key, default in (("async_job_retry_base_delay_seconds", 1),
                         ("async_job_retry_max_delay_seconds", 300)):
        value = server.get(key, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"server.{key} must be a non-negative number")
    if server.get("async_job_retry_max_delay_seconds", 300) < server.get("async_job_retry_base_delay_seconds", 1):
        raise ValueError("server.async_job_retry_max_delay_seconds must be at least the retry base delay")
    for key, default in (("webhook_circuit_failure_threshold", 3),
                         ("webhook_circuit_recovery_seconds", 30)):
        value = server.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"server.{key} must be a positive integer")
    webhook_retention_days = server.get("webhook_retention_days", 90)
    if (not isinstance(webhook_retention_days, int) or isinstance(webhook_retention_days, bool)
            or webhook_retention_days < 1):
        raise ValueError("server.webhook_retention_days must be a positive integer")
    dns_timeout = server.get("webhook_dns_timeout_seconds", 2.0)
    if not isinstance(dns_timeout, (int, float)) or isinstance(dns_timeout, bool) or dns_timeout <= 0:
        raise ValueError("server.webhook_dns_timeout_seconds must be a positive number")
    if not isinstance(server.get("webhooks_allow_private_targets", False), bool):
        raise TypeError("server.webhooks_allow_private_targets must be a boolean")
    async_jobs = config.get("async_jobs", {})
    if not isinstance(async_jobs, dict):
        raise TypeError("async_jobs must be an object")
    job_webhooks = async_jobs.get("webhooks", {})
    if not isinstance(job_webhooks, dict):
        raise TypeError("async_jobs.webhooks must be an application map")
    valid_job_events = {"queued", "progress", "completed", "cancelled", "dead_letter"}
    for application, webhook in job_webhooks.items():
        if (not isinstance(application, str) or not application or not isinstance(webhook, dict)
                or not isinstance(webhook.get("url"), str)):
            raise ValueError("async_jobs.webhooks entries require an application and HTTP(S) url")
        validate_webhook_url(webhook["url"], resolve_dns=False, allow_private_targets=True)
        _validate_webhook_signing(webhook, "async_jobs.webhooks")
        events = webhook.get("events", list(valid_job_events))
        if not isinstance(events, list) or any(event not in valid_job_events for event in events):
            raise ValueError("async_jobs.webhooks events must contain supported job lifecycle events")
    object_storage = config.get("object_storage", {})
    if not isinstance(object_storage, dict) or object_storage.get("backend", "local") not in {"local", "s3"}:
        raise ValueError("object_storage must configure the 'local' or 's3' backend")
    if object_storage.get("backend") == "s3" and not object_storage.get("bucket"):
        raise ValueError("object_storage.bucket is required when using S3")
    tenant_data = config.get("tenant_data", {})
    if not isinstance(tenant_data, dict):
        raise TypeError("tenant_data must be an object")
    threshold = tenant_data.get("async_threshold_records", 10_000)
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 1:
        raise ValueError("tenant_data.async_threshold_records must be a positive integer")
    budget_config = config.get("budgets", {})
    if not isinstance(budget_config, dict):
        raise TypeError("budgets must be an object")
    budget_policies = [("default", budget_config.get("default", {})),
        *[(f"applications.{name}", policy)
          for name, policy in budget_config.get("applications", {}).items()]]
    for policy_name, policy in budget_policies:
        if not isinstance(policy, dict):
            raise TypeError(f"budgets.{policy_name} must be an object")
        webhook_url = policy.get("alert_webhook_url")
        if webhook_url is not None:
            if not isinstance(webhook_url, str):
                raise ValueError(f"budgets.{policy_name}.alert_webhook_url must be a URL")
            validate_webhook_url(webhook_url, resolve_dns=False, allow_private_targets=True)
            _validate_webhook_signing({"secret_env": policy.get("alert_webhook_secret_env"),
                "signing_keys": policy.get("alert_signing_keys", {}),
                "active_key_id": policy.get("alert_active_key_id")}, f"budgets.{policy_name}")
    for provider_name, provider in config.get("providers", {}).items():
        batch_size = provider.get("max_embedding_batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                                       or batch_size < 1 or batch_size > 50_000):
            raise ValueError(f"Provider {provider_name!r} max_embedding_batch_size must be between 1 and 50000")
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
        if not application or not isinstance(webhook, dict) or not isinstance(webhook.get("url"), str):
            raise ValueError("tool_governance.approval_webhooks entries require an HTTP(S) url")
        validate_webhook_url(webhook["url"], resolve_dns=False, allow_private_targets=True)
        _validate_webhook_signing(webhook, "tool_governance.approval_webhooks")
    tenants = config.get("tenants", config.get("security", {}).get("tenants", {}))
    for tenant_id, tenant in tenants.items():
        if not isinstance(tenant, dict):
            raise TypeError(f"Tenant {tenant_id!r} configuration must be an object")
        for key in ("allowed_routes", "allowed_providers", "allowed_models", "allowed_regions",
                    "allowed_tools", "require_approval_tools"):
            value = tenant.get(key)
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                raise ValueError(f"Tenant {tenant_id!r} {key} must be a list of strings")
        job_policy = tenant.get("job_policy", {})
        if not isinstance(job_policy, dict):
            raise TypeError(f"Tenant {tenant_id!r} job_policy must be an object")
        for limit_name in ("max_concurrent_jobs", "max_queued_jobs", "max_job_items",
                           "max_job_input_bytes", "retention_days"):
            value = job_policy.get(limit_name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"Tenant {tenant_id!r} job_policy.{limit_name} must be a positive integer")
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


def _validate_webhook_signing(webhook: dict[str, Any], section: str) -> None:
    secret_env = webhook.get("secret_env")
    if secret_env is not None and (not isinstance(secret_env, str) or not secret_env):
        raise ValueError(f"{section} secret_env must be a non-empty string")
    signing_keys = webhook.get("signing_keys", {})
    active_key_id = webhook.get("active_key_id")
    if not isinstance(signing_keys, dict) or any(
            not isinstance(key_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", key_id)
            or not isinstance(env_name, str) or not env_name
            for key_id, env_name in signing_keys.items()):
        raise ValueError(f"{section} signing_keys must map key IDs to environment variable names")
    if signing_keys and active_key_id not in signing_keys:
        raise ValueError(f"{section} active_key_id must name an entry in signing_keys")
    if active_key_id is not None and not signing_keys:
        raise ValueError(f"{section} active_key_id requires signing_keys")


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
    retrieval_routes = config.get("embedding_routes", {})
    if not isinstance(retrieval_routes, dict):
        raise TypeError("embedding_routes must be an object")
    for route_name, route in retrieval_routes.items():
        if not isinstance(route, dict) or route.get("provider") not in config["providers"]:
            raise ValueError(f"Embedding route {route_name!r} must reference a configured provider")
        if not isinstance(route.get("model"), str) or not route["model"]:
            raise ValueError(f"Embedding route {route_name!r} requires a model")
        if not isinstance(route.get("fallbacks", []), list):
            raise TypeError(f"Embedding route {route_name!r} fallbacks must be a list")
        candidates = [route, *route.get("fallbacks", [])]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise TypeError(f"Embedding route {route_name!r} candidates must be objects")
            provider_name = candidate.get("provider")
            if provider_name not in config["providers"]:
                raise ValueError(f"Embedding route {route_name!r} refers to unknown provider {provider_name!r}")
            if not provider_capabilities(config["providers"][provider_name]).get("embeddings", False):
                raise ValueError(f"Embedding route {route_name!r} provider {provider_name!r} lacks embeddings capability")
            if not isinstance(candidate.get("model"), str) or not candidate["model"]:
                raise ValueError(f"Embedding route {route_name!r} candidates require a model")
        reranker = route.get("reranker")
        if reranker is not None:
            if not isinstance(reranker, dict):
                raise TypeError(f"Embedding route {route_name!r} reranker must be an object")
            rerank_candidates = [reranker, *reranker.get("fallbacks", [])]
            if not isinstance(reranker.get("fallbacks", []), list):
                raise TypeError(f"Embedding route {route_name!r} reranker fallbacks must be a list")
            for candidate in rerank_candidates:
                if not isinstance(candidate, dict):
                    raise TypeError(f"Embedding route {route_name!r} reranker candidates must be objects")
                provider_name = candidate.get("provider")
                if provider_name not in config["providers"]:
                    raise ValueError(f"Embedding route {route_name!r} reranker refers to unknown provider {provider_name!r}")
                if not provider_capabilities(config["providers"][provider_name]).get("native_reranking", False):
                    raise ValueError(f"Embedding route {route_name!r} provider {provider_name!r} lacks native reranking capability")
                if not isinstance(candidate.get("model"), str) or not candidate["model"]:
                    raise ValueError(f"Embedding route {route_name!r} reranker candidates require a model")
            if reranker.get("fallback", "cosine") not in {"cosine", "error"}:
                raise ValueError(f"Embedding route {route_name!r} reranker fallback must be cosine or error")
    retrieval_policy = config.get("retrieval", {})
    if not isinstance(retrieval_policy, dict):
        raise TypeError("retrieval must be an object")
    embedding_cache = retrieval_policy.get("embedding_cache", {})
    if not isinstance(embedding_cache, dict):
        raise TypeError("retrieval.embedding_cache must be an object")
    ttl = embedding_cache.get("ttl_seconds", 300)
    if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl < 0:
        raise ValueError("retrieval.embedding_cache.ttl_seconds must be a non-negative integer")
    if not isinstance(embedding_cache.get("enabled", False), bool):
        raise TypeError("retrieval.embedding_cache.enabled must be a boolean")
    if embedding_cache.get("enabled", False) and not embedding_cache.get("hmac_key_env"):
        raise ValueError("retrieval.embedding_cache requires hmac_key_env when enabled")
    all_routes = set(config["routes"]) | set(retrieval_routes)
    for tenant_id, tenant in config.get("tenants", config.get("security", {}).get("tenants", {})).items():
        unknown_routes = set(tenant.get("allowed_routes", [])) - all_routes
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
        for model_name, pricing in provider.get("models", {}).items():
            rerank_price = pricing.get("rerank_cost_per_1k_documents")
            if rerank_price is not None and (isinstance(rerank_price, bool)
                    or not isinstance(rerank_price, (int, float)) or rerank_price < 0):
                raise ValueError(f"Provider {provider_name!r} model {model_name!r} rerank cost must be non-negative")
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
