"""Policy helpers for model-declared and model-generated function calls."""

from __future__ import annotations

from typing import Any


class ToolGovernanceError(Exception):
    def __init__(self, detail: str, status_code: int, reason: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.reason = reason


def requested_tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    return [tool.get("function", {}).get("name", "") for tool in tools or []]


def generated_tool_names(tool_calls: list[dict[str, Any]] | None) -> list[str]:
    return [call.get("function", {}).get("name", "") for call in tool_calls or []]


def effective_tool_policy(global_policy: dict[str, Any], tenant: dict[str, Any],
                          application_policy: dict[str, Any]) -> tuple[set[str] | None, int | None, int | None,
                                                                      set[str]]:
    allowlists = [value for value in (global_policy.get("allowed_tools"), tenant.get("allowed_tools"),
        tenant.get("default_policy", {}).get("allowed_tools"), application_policy.get("allowed_tools"))
        if value is not None]
    allowed = set.intersection(*(set(value) for value in allowlists)) if allowlists else None
    limits = [value for value in (global_policy.get("max_tool_calls_per_request"),
        tenant.get("default_policy", {}).get("max_tool_calls_per_request"),
        application_policy.get("max_tool_calls_per_request")) if value is not None]
    request_limit = min(int(value) for value in limits) if limits else None
    minute_limits = [value for value in (global_policy.get("max_tool_calls_per_minute"),
        tenant.get("default_policy", {}).get("max_tool_calls_per_minute"),
        application_policy.get("max_tool_calls_per_minute")) if value is not None]
    minute_limit = min(int(value) for value in minute_limits) if minute_limits else None
    approval_lists = [value for value in (global_policy.get("require_approval_tools"),
        tenant.get("require_approval_tools"), tenant.get("default_policy", {}).get("require_approval_tools"),
        application_policy.get("require_approval_tools")) if value is not None]
    require_approval = set().union(*(set(value) for value in approval_lists)) if approval_lists else set()
    return allowed, request_limit, minute_limit, require_approval
