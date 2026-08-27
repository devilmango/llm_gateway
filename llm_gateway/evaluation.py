"""Small offline-set runner for comparing gateway logical routes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


def _as_strings(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else value


def grade_response(case: dict[str, Any], content: str) -> bool | None:
    required = _as_strings(case.get("expected_contains"))
    forbidden = _as_strings(case.get("expected_not_contains"))
    if not required and not forbidden:
        return None
    lowered = content.casefold()
    return all(text.casefold() in lowered for text in required) and all(
        text.casefold() not in lowered for text in forbidden
    )


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials == 0:
        return None
    rate = successes / trials
    scale = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / scale
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / scale
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


async def evaluate_cases(cases: list[dict[str, Any]], base_url: str, api_key: str | None = None,
                        transport: httpx.AsyncBaseTransport | None = None,
                        application: str = "llm-gateway-eval",
                        route_override: str | None = None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), transport=transport,
                                 headers=headers, timeout=120) as client:
        for index, case in enumerate(cases):
            route = route_override or case.get("route", "simple")
            started = time.perf_counter()
            try:
                response = await client.post("/v1/chat/completions", json={
                    "model": route,
                    "messages": case["messages"],
                    "application": application,
                    "max_tokens": case.get("max_tokens", 512),
                })
                elapsed_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"].get("content") or ""
                passed = grade_response(case, content)
                results.append({"case": case.get("id", str(index)), "route": route,
                    "passed": passed, "latency_ms": round(elapsed_ms, 2),
                    "cost_usd": body.get("gateway", {}).get("cost_usd", 0),
                    "provider": body.get("gateway", {}).get("provider"),
                    "model": body.get("gateway", {}).get("upstream_model")})
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                results.append({"case": case.get("id", str(index)), "route": route,
                    "passed": False, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": type(exc).__name__})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["route"]].append(result)
    summaries = {}
    for route, route_results in grouped.items():
        graded = [item["passed"] for item in route_results if item["passed"] is not None]
        latencies = [item["latency_ms"] for item in route_results]
        summaries[route] = {
            "requests": len(route_results),
            "graded_cases": len(graded),
            "pass_rate": sum(graded) / len(graded) if graded else None,
            "pass_rate_ci95": _wilson_interval(sum(graded), len(graded)),
            "average_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
            "estimated_cost_usd": round(sum(float(item.get("cost_usd", 0)) for item in route_results), 8),
            "failures": sum(item["passed"] is False for item in route_results),
        }
    return {"routes": summaries, "results": results}


async def compare_routes(cases: list[dict[str, Any]], base_url: str, baseline_route: str,
                         candidate_route: str, api_key: str | None = None,
                         transport: httpx.AsyncBaseTransport | None = None,
                         application: str = "llm-gateway-eval",
                         min_pass_rate: float | None = None,
                         max_cost_regression_percent: float | None = None,
                         max_latency_regression_percent: float | None = None,
                         max_failure_rate: float | None = None) -> dict[str, Any]:
    """Run the same cases on two routes and evaluate optional promotion gates."""
    # Keep route runs sequential so each route's latency measurement is independent.
    baseline = await evaluate_cases(cases, base_url, api_key, transport, application, baseline_route)
    candidate = await evaluate_cases(cases, base_url, api_key, transport, application, candidate_route)
    before = baseline["routes"].get(baseline_route, {})
    after = candidate["routes"].get(candidate_route, {})

    def regression_percent(old: float, new: float) -> float | str:
        if old == 0:
            return 0.0 if new == 0 else "infinite"
        return round((new - old) / old * 100, 2)

    pass_delta = None
    if before.get("pass_rate") is not None and after.get("pass_rate") is not None:
        pass_delta = round(after["pass_rate"] - before["pass_rate"], 4)
    comparison = {
        "baseline_route": baseline_route,
        "candidate_route": candidate_route,
        "pass_rate_delta": pass_delta,
        "cost_regression_percent": regression_percent(
            float(before.get("estimated_cost_usd", 0)), float(after.get("estimated_cost_usd", 0))),
        "latency_regression_percent": regression_percent(
            float(before.get("average_latency_ms", 0)), float(after.get("average_latency_ms", 0))),
        "candidate_failure_rate": sum("error" in item for item in candidate["results"]) /
            max(after.get("requests", 0), 1),
        "candidate_checks_passed": not any(item["passed"] is False for item in candidate["results"]),
    }
    gates = []
    if min_pass_rate is not None:
        gates.append(after.get("pass_rate") is not None and after["pass_rate"] >= min_pass_rate)
    if max_cost_regression_percent is not None:
        gates.append(comparison["cost_regression_percent"] != "infinite" and
                     comparison["cost_regression_percent"] <= max_cost_regression_percent)
    if max_latency_regression_percent is not None:
        gates.append(comparison["latency_regression_percent"] != "infinite" and
                     comparison["latency_regression_percent"] <= max_latency_regression_percent)
    if max_failure_rate is not None:
        gates.append(comparison["candidate_failure_rate"] <= max_failure_rate)
    comparison["promotion_passed"] = all(gates) and comparison["candidate_checks_passed"]
    comparison["gates_configured"] = len(gates)
    return {"baseline": baseline, "candidate": candidate, "comparison": comparison}


def _saved_run_routes(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    routes: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    if isinstance(report.get("routes"), dict):
        routes.update(report["routes"])
        results.extend(report.get("results", []))
    for side in ("baseline", "candidate"):
        evaluation = report.get(side)
        if not isinstance(evaluation, dict):
            continue
        if isinstance(evaluation.get("routes"), dict):
            routes.update(evaluation["routes"])
        results.extend(evaluation.get("results", []))
    return routes, results


def _route_wilson_interval(summary: dict[str, Any], results: list[dict[str, Any]],
                           route: str, z: float) -> list[float] | None:
    graded = [row["passed"] for row in results
              if row.get("route") == route and row.get("passed") is not None]
    if graded:
        successes, trials = sum(graded), len(graded)
    else:
        trials = int(summary.get("graded_cases", 0))
        successes = round(float(summary.get("pass_rate") or 0) * trials)
    return _wilson_interval(successes, trials, z)


def compare_saved_runs(baseline_run: dict[str, Any], candidate_run: dict[str, Any],
                       *, min_pass_rate: float | None = None,
                       max_cost_regression_percent: float | None = None,
                       max_latency_regression_percent: float | None = None,
                       max_failure_rate: float | None = None) -> dict[str, Any]:
    """Compare stored run summaries by route, including a conservative pass-rate CI."""
    baseline_routes, baseline_results = _saved_run_routes(baseline_run["report"])
    candidate_routes, candidate_results = _saved_run_routes(candidate_run["report"])
    route_diffs: dict[str, Any] = {}
    checks: list[bool] = []

    def regression(before: float, after: float) -> float | str:
        if before == 0:
            return 0.0 if after == 0 else "infinite"
        return round((after - before) / before * 100, 2)

    for route in sorted(set(baseline_routes) | set(candidate_routes)):
        before = baseline_routes.get(route)
        after = candidate_routes.get(route)
        if before is None or after is None:
            route_diffs[route] = {"change": "added" if before is None else "removed",
                                  "baseline": before, "candidate": after, "deltas": None}
            continue
        before_pass = before.get("pass_rate")
        after_pass = after.get("pass_rate")
        pass_delta = round(after_pass - before_pass, 4) if before_pass is not None and after_pass is not None else None
        # Bonferroni-adjusted component intervals give at least 95% coverage
        # for the difference without assuming the two samples are independent.
        z_difference = 2.241402727604947
        before_ci = _route_wilson_interval(before, baseline_results, route, z_difference)
        after_ci = _route_wilson_interval(after, candidate_results, route, z_difference)
        delta_ci = ([round(after_ci[0] - before_ci[1], 4), round(after_ci[1] - before_ci[0], 4)]
                    if before_ci and after_ci else None)
        cost_regression = regression(float(before.get("estimated_cost_usd", 0)),
                                     float(after.get("estimated_cost_usd", 0)))
        latency_regression = regression(float(before.get("average_latency_ms", 0)),
                                        float(after.get("average_latency_ms", 0)))
        candidate_errors = sum(1 for row in candidate_results
                               if row.get("route") == route and "error" in row)
        failure_rate = candidate_errors / max(int(after.get("requests", 0)), 1)
        route_diffs[route] = {
            "change": "compared", "baseline": before, "candidate": after,
            "deltas": {"pass_rate": pass_delta,
                "pass_rate_ci95_conservative": delta_ci,
                "pass_rate_direction_confident": bool(delta_ci and (delta_ci[0] > 0 or delta_ci[1] < 0)),
                "estimated_cost_usd": round(float(after.get("estimated_cost_usd", 0)) -
                                              float(before.get("estimated_cost_usd", 0)), 8),
                "cost_regression_percent": cost_regression,
                "average_latency_ms": round(float(after.get("average_latency_ms", 0)) -
                                             float(before.get("average_latency_ms", 0)), 2),
                "latency_regression_percent": latency_regression,
                "failures": int(after.get("failures", 0)) - int(before.get("failures", 0)),
                "candidate_failure_rate": round(failure_rate, 4)},
        }
        if min_pass_rate is not None:
            checks.append(after_pass is not None and after_pass >= min_pass_rate)
        if max_cost_regression_percent is not None:
            checks.append(cost_regression != "infinite" and cost_regression <= max_cost_regression_percent)
        if max_latency_regression_percent is not None:
            checks.append(latency_regression != "infinite" and latency_regression <= max_latency_regression_percent)
        if max_failure_rate is not None:
            checks.append(failure_rate <= max_failure_rate)

    candidate_check_passed = not any(row.get("passed") is False for row in candidate_results)
    return {"baseline_run_id": baseline_run["run_id"], "candidate_run_id": candidate_run["run_id"],
        "baseline_dataset": {key: baseline_run.get(key) for key in ("dataset_name", "dataset_version", "dataset_sha256")},
        "candidate_dataset": {key: candidate_run.get(key) for key in ("dataset_name", "dataset_version", "dataset_sha256")},
        "routes": route_diffs, "candidate_checks_passed": candidate_check_passed,
        "promotion_passed": all(checks) and candidate_check_passed if checks else None,
        "gates_configured": len(checks)}


async def persist_report(base_url: str, api_key: str, metadata: dict[str, Any],
                         report: dict[str, Any]) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=30) as client:
        response = await client.post("/v1/admin/evaluations", json={**metadata, "report": report})
        response.raise_for_status()
        return response.json()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if not isinstance(case, dict) or not isinstance(case.get("messages"), list):
            raise TypeError(f"Line {line_number} must contain an object with a messages array")
        cases.append(case)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate configured gateway routes against a JSONL prompt set")
    parser.add_argument("dataset", type=Path, help="JSONL file with route, messages, and optional expected text")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway base URL")
    parser.add_argument("--api-key", default=None, help="Gateway bearer key; prefer using a shell env wrapper")
    parser.add_argument("--application", default="llm-gateway-eval")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument("--dataset-name", help="Stable evaluation dataset name; defaults to the file name")
    parser.add_argument("--dataset-version", help="Immutable version label; defaults to the dataset SHA-256 prefix")
    parser.add_argument("--persist", action="store_true", help="Save the dataset version and report to the gateway")
    parser.add_argument("--admin-api-key", default=os.getenv("GATEWAY_ADMIN_API_KEY"),
                        help="Admin key with report scope; can also use GATEWAY_ADMIN_API_KEY")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="Run every case against both logical routes and compare them")
    parser.add_argument("--min-pass-rate", type=float, help="Candidate minimum graded pass rate (0..1)")
    parser.add_argument("--max-cost-regression-percent", type=float,
                        help="Maximum candidate estimated cost regression versus baseline")
    parser.add_argument("--max-latency-regression-percent", type=float,
                        help="Maximum candidate average latency regression versus baseline")
    parser.add_argument("--max-failure-rate", type=float,
                        help="Maximum candidate request failure rate (0..1)")
    args = parser.parse_args()
    try:
        cases = _load_jsonl(args.dataset)
        dataset_bytes = args.dataset.read_bytes()
        dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
        metadata = {
            "dataset_name": args.dataset_name or args.dataset.stem,
            "dataset_version": args.dataset_version or f"sha256:{dataset_sha256[:12]}",
            "dataset_sha256": dataset_sha256,
            "case_count": len(cases),
        }
        if args.persist and not args.admin_api_key:
            raise ValueError("--persist requires --admin-api-key or GATEWAY_ADMIN_API_KEY")
        if args.min_pass_rate is not None and not 0 <= args.min_pass_rate <= 1:
            raise ValueError("--min-pass-rate must be between 0 and 1")
        if args.max_failure_rate is not None and not 0 <= args.max_failure_rate <= 1:
            raise ValueError("--max-failure-rate must be between 0 and 1")
        for name in ("max_cost_regression_percent", "max_latency_regression_percent"):
            value = getattr(args, name)
            if value is not None and value < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
        if args.compare:
            report = asyncio.run(compare_routes(cases, args.url, *args.compare, args.api_key,
                application=args.application, min_pass_rate=args.min_pass_rate,
                max_cost_regression_percent=args.max_cost_regression_percent,
                max_latency_regression_percent=args.max_latency_regression_percent,
                max_failure_rate=args.max_failure_rate))
        else:
            report = asyncio.run(evaluate_cases(cases, args.url, args.api_key,
                                                application=args.application))
        report["dataset"] = metadata
        if args.persist:
            report["persistence"] = asyncio.run(persist_report(args.url, args.admin_api_key, metadata, report))
    except (OSError, TypeError, ValueError, httpx.HTTPError) as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if args.compare:
        failed = report["comparison"]["promotion_passed"] is False
    else:
        failed = any(item["passed"] is False for item in report["results"])
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
