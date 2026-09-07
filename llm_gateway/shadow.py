"""Helpers for sampled shadow requests and conservative promotion gates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from jsonschema import Draft202012Validator

from .tool_validation import validate_tool_calls


def should_sample(request_id: str, percentage: float) -> bool:
    if percentage <= 0:
        return False
    bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 10_000
    return bucket < percentage * 100


def quality_passed(result: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Return a content-free quality proxy: valid non-empty output and valid requested shape."""
    text = result.get("text") or ""
    calls = result.get("tool_calls") or []
    if not text and not calls and not result.get("audio"):
        return False
    try:
        if calls:
            validate_tool_calls(calls, payload.get("tools"))
        response_format = payload.get("response_format") or {}
        if response_format.get("type") in {"json_object", "json_schema"}:
            parsed = json.loads(text)
            if response_format["type"] == "json_schema":
                schema = response_format["json_schema"]["schema"]
                Draft202012Validator(schema).validate(parsed)
        return True
    except (ValueError, TypeError, KeyError):
        return False


def promotion_gates(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    minimum_samples = int(config.get("min_samples", 30))
    max_quality_loss = float(config.get("max_quality_loss", 0.02))
    max_cost_regression = float(config.get("max_cost_regression_percent", 10))
    max_latency_regression = float(config.get("max_latency_regression_percent", 20))
    max_failure_rate = float(config.get("max_failure_rate", 0.01))
    quality_delta = (report["candidate"]["quality_pass_rate"] -
                     report["baseline"]["quality_pass_rate"])
    gates = {
        "minimum_samples": {"passed": report["samples"] >= minimum_samples,
            "observed": report["samples"], "required": minimum_samples},
        "quality": {"passed": quality_delta >= -max_quality_loss,
            "candidate_pass_rate": report["candidate"]["quality_pass_rate"],
            "baseline_pass_rate": report["baseline"]["quality_pass_rate"],
            "maximum_loss": max_quality_loss},
        "cost": {"passed": report["cost_regression_percent"] <= max_cost_regression,
            "regression_percent": report["cost_regression_percent"], "maximum_percent": max_cost_regression},
        "latency": {"passed": report["latency_regression_percent"] <= max_latency_regression,
            "regression_percent": report["latency_regression_percent"],
            "maximum_percent": max_latency_regression},
        "failures": {"passed": report["candidate"]["failure_rate"] <= max_failure_rate,
            "rate": report["candidate"]["failure_rate"], "maximum_rate": max_failure_rate},
    }
    return {"gates": gates, "passed": all(gate["passed"] for gate in gates.values())}
