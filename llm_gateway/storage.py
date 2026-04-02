from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UsageStore:
    """Small durable usage ledger. Stores metadata only, never prompt/response text."""

    def __init__(self, database: str) -> None:
        self.database = database
        self._lock = threading.Lock()
        if database != ":memory:":
            Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL,
                application TEXT NOT NULL, route TEXT NOT NULL,
                provider TEXT, model TEXT, prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0, status TEXT NOT NULL,
                latency_ms INTEGER NOT NULL DEFAULT 0, fallback_used INTEGER NOT NULL DEFAULT 0,
                cached INTEGER NOT NULL DEFAULT 0, error_type TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(occurred_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_requests_application ON requests(application)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def record(self, record: dict[str, Any]) -> None:
        columns = ("request_id", "occurred_at", "application", "route", "provider", "model",
                   "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "status",
                   "latency_ms", "fallback_used", "cached", "error_type")
        values = [record.get(c) for c in columns]
        with self._lock, self._connect() as db:
            db.execute(f"INSERT OR REPLACE INTO requests ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", values)

    def report(self, start: str | None, end: str | None, application: str | None) -> dict[str, Any]:
        clauses, values = [], []
        if start:
            clauses.append("occurred_at >= ?"); values.append(start)
        if end:
            clauses.append("occurred_at < ?"); values.append(end)
        if application:
            clauses.append("application = ?"); values.append(application)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock, self._connect() as db:
            summary = dict(db.execute(f"""SELECT COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                COALESCE(SUM(total_tokens),0) AS total_tokens,
                COALESCE(SUM(cost_usd),0) AS cost_usd,
                COALESCE(AVG(cost_usd),0) AS avg_cost_usd,
                COALESCE(AVG(latency_ms),0) AS avg_latency_ms,
                COALESCE(SUM(status != 'success'),0) AS failures,
                COALESCE(SUM(fallback_used),0) AS fallbacks,
                COALESCE(SUM(cached),0) AS cache_hits
                FROM requests{where}""", values).fetchone())
            by_app = [dict(row) for row in db.execute(f"""SELECT application,
                COUNT(*) AS requests, SUM(total_tokens) AS total_tokens,
                SUM(cost_usd) AS cost_usd, SUM(status != 'success') AS failures,
                SUM(fallback_used) AS fallbacks FROM requests{where}
                GROUP BY application ORDER BY cost_usd DESC""", values).fetchall()]
        summary["failure_rate"] = summary["failures"] / summary["requests"] if summary["requests"] else 0
        summary["fallback_rate"] = summary["fallbacks"] / summary["requests"] if summary["requests"] else 0
        summary["generated_at"] = datetime.now(timezone.utc).isoformat()
        summary["by_application"] = by_app
        return summary
