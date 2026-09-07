"""Small client helpers for safely invoking tools returned by the gateway."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar
from urllib.parse import quote

import httpx

T = TypeVar("T")


class ToolApprovalError(RuntimeError):
    """Raised when the gateway has not approved a tool call for execution."""

    def __init__(self, status: str):
        super().__init__(f"Tool call cannot execute while approval status is {status!r}")
        self.status = status


async def execute_tool_with_approval(client: httpx.AsyncClient, *, request_id: str,
                                     tool_call_id: str, application: str,
                                     execute: Callable[[], Awaitable[T]]) -> T:
    """Check approval immediately before execution and report the resulting outcome.

    Configure ``client`` with the gateway base URL and the caller's tenant
    authorization header. Tool arguments stay in the calling application.
    """
    path = (f"/v1/tool-approvals/{quote(request_id, safe='')}/{quote(tool_call_id, safe='')}")
    response = await client.get(path, params={"application": application})
    response.raise_for_status()
    status = response.json().get("status")
    if status not in {"approved", "not_required"}:
        raise ToolApprovalError(str(status or "unknown"))

    outcome_path = "/v1/tool-executions"
    outcome = {"request_id": request_id, "tool_call_id": tool_call_id,
        "application": application, "outcome": "succeeded"}
    try:
        result = await execute()
    except Exception as exc:
        try:
            await client.post(outcome_path, json={**outcome, "outcome": "failed",
                "error_code": type(exc).__name__[:80]})
        except httpx.HTTPError:
            pass
        raise
    reported = await client.post(outcome_path, json=outcome)
    reported.raise_for_status()
    return result
