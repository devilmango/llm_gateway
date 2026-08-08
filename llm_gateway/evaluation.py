"""Small offline-set runner for comparing gateway logical routes."""

from __future__ import annotations

import argparse
import asyncio
import json
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


async def evaluate_cases(cases: list[dict[str, Any]], base_url: str, api_key: str | None = None,
                        transport: httpx.AsyncBaseTransport | None = None,
                        application: str = "llm-gateway-eval") -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), transport=transport,
                                 headers=headers, timeout=120) as client:
        for index, case in enumerate(cases):
            route = case.get("route", "simple")
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
            "average_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
            "estimated_cost_usd": round(sum(float(item.get("cost_usd", 0)) for item in route_results), 8),
            "failures": sum(item["passed"] is False for item in route_results),
        }
    return {"routes": summaries, "results": results}


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
    args = parser.parse_args()
    try:
        report = asyncio.run(evaluate_cases(_load_jsonl(args.dataset), args.url, args.api_key,
                                            application=args.application))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if any(item["passed"] is False for item in report["results"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
