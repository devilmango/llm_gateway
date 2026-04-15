from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class BudgetExceeded(Exception):
    pass


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
            db.execute("""CREATE TABLE IF NOT EXISTS budget_reservations (
                request_id TEXT PRIMARY KEY, application TEXT NOT NULL,
                amount_usd REAL NOT NULL, reserved_at TEXT NOT NULL, expires_at TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_budget_reservations_app ON budget_reservations(application, expires_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS budget_alerts (
                application TEXT NOT NULL, month_start TEXT NOT NULL,
                threshold REAL NOT NULL, emitted_at TEXT NOT NULL,
                PRIMARY KEY(application, month_start, threshold)
            )""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def reserve_budget(self, request_id: str, application: str, amount_usd: float, monthly_limit_usd: float,
                       now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        month_start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM budget_reservations WHERE expires_at <= ?", (now.isoformat(),))
            spent = db.execute("SELECT COALESCE(SUM(cost_usd),0) FROM requests WHERE application = ? AND occurred_at >= ? AND occurred_at < ?",
                               (application, month_start.isoformat(), month_end.isoformat())).fetchone()[0]
            reserved = db.execute("SELECT COALESCE(SUM(amount_usd),0) FROM budget_reservations WHERE application = ?",
                                  (application,)).fetchone()[0]
            if monthly_limit_usd <= 0 or spent + reserved + amount_usd > monthly_limit_usd:
                raise BudgetExceeded("Application monthly spend limit would be exceeded")
            db.execute("INSERT OR REPLACE INTO budget_reservations VALUES (?, ?, ?, ?, ?)",
                       (request_id, application, amount_usd, now.isoformat(), (now + timedelta(minutes=5)).isoformat()))

    def record(self, record: dict[str, Any]) -> None:
        columns = ("request_id", "occurred_at", "application", "route", "provider", "model",
                   "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "status",
                   "latency_ms", "fallback_used", "cached", "error_type")
        values = [record.get(c) for c in columns]
        with self._lock, self._connect() as db:
            db.execute(f"INSERT OR REPLACE INTO requests ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", values)
            db.execute("DELETE FROM budget_reservations WHERE request_id = ?", (record["request_id"],))

    def check_budget_alerts(self, application: str, monthly_limit_usd: float,
                            thresholds: list[float] | None = None,
                            now: datetime | None = None) -> list[dict[str, float]]:
        """Return newly crossed monthly spend thresholds, once per application/month."""
        now = now or datetime.now(UTC)
        month_start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = month_start.replace(year=month_start.year + (month_start.month == 12),
                                        month=1 if month_start.month == 12 else month_start.month + 1)
        thresholds = thresholds or [0.8, 1.0]
        with self._lock, self._connect() as db:
            spent = db.execute("SELECT COALESCE(SUM(cost_usd),0) FROM requests WHERE application = ? AND occurred_at >= ? AND occurred_at < ?",
                               (application, month_start.isoformat(), month_end.isoformat())).fetchone()[0]
            crossed = []
            for threshold in thresholds:
                if monthly_limit_usd <= 0 or spent < monthly_limit_usd * threshold:
                    continue
                cursor = db.execute("INSERT OR IGNORE INTO budget_alerts VALUES (?, ?, ?, ?)",
                                    (application, month_start.isoformat(), threshold, now.isoformat()))
                if cursor.rowcount:
                    crossed.append({"threshold": threshold, "spend_usd": spent, "limit_usd": monthly_limit_usd})
        return crossed

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
        summary["generated_at"] = datetime.now(UTC).isoformat()
        summary["by_application"] = by_app
        return summary


def create_usage_store(database: str) -> Any:
    if database.startswith(("postgres://", "postgresql://")):
        from .postgres_storage import PostgresUsageStore

        return PostgresUsageStore(database)
    return UsageStore(database)
