"""Normalize provider billing exports into the gateway reconciliation ledger."""

from __future__ import annotations

import csv
import io
import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any

_REQUEST_ID_ALIASES = ("request_id", "gateway_request_id", "llm_gateway_request_id")
_COST_ALIASES = ("actual_cost_usd", "cost_usd", "amount_usd", "cost")


def _path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _pick(row: dict[str, Any], explicit: str | None, aliases: tuple[str, ...]) -> Any:
    if explicit:
        return row[explicit] if explicit in row else _path(row, explicit)
    for name in aliases:
        if name in row:
            return row[name]
    return None


def normalize_billing_export(content: str, file_format: str,
                             mapping: dict[str, str] | None = None,
                             cost_multiplier: float = 1.0) -> list[dict[str, Any]]:
    """Parse CSV, JSON, or JSONL rows and map charge rows to gateway request IDs.

    A source export must include a gateway request ID (for example as a provider
    metadata tag) and an amount denominated in USD. Column/path mappings support
    provider-specific export schemas without guessing provider formats.
    """
    mapping = mapping or {}
    if not math.isfinite(cost_multiplier) or cost_multiplier <= 0:
        raise ValueError("cost_multiplier must be a positive finite number")
    if not content.strip():
        raise ValueError("Billing export is empty")
    if file_format == "csv":
        rows = list(csv.DictReader(io.StringIO(content)))
    elif file_format == "json":
        payload = json.loads(content)
        rows = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON billing export must be an array or an object with a records array")
    elif file_format == "jsonl":
        rows = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
    else:
        raise ValueError("format must be csv, json, or jsonl")
    if not rows:
        raise ValueError("Billing export has no rows")

    normalized = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Billing row {row_number} must be an object")
        request_id = _pick(row, mapping.get("request_id"), _REQUEST_ID_ALIASES)
        amount = _pick(row, mapping.get("actual_cost_usd"), _COST_ALIASES)
        currency = _pick(row, mapping.get("currency"), ("currency", "currency_code"))
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError(f"Billing row {row_number} has no gateway request ID")
        if currency is not None and str(currency).upper() != "USD":
            raise ValueError(f"Billing row {row_number} currency must be USD")
        try:
            actual_cost = Decimal(str(amount)) * Decimal(str(cost_multiplier))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"Billing row {row_number} has an invalid USD amount") from exc
        if not actual_cost.is_finite() or actual_cost < 0 or not math.isfinite(float(actual_cost)):
            raise ValueError(f"Billing row {row_number} has an invalid USD amount")
        request_id = request_id.strip()
        if request_id in seen_ids:
            raise ValueError(f"Duplicate request_id {request_id!r} in billing export")
        seen_ids.add(request_id)
        normalized.append({"request_id": request_id, "actual_cost_usd": float(actual_cost)})
    return normalized
