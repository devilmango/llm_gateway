from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .storage import BudgetExceeded


class PostgresUsageStore:
    """PostgreSQL implementation of the gateway usage ledger and budget store."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10) -> None:
        self.pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size,
                                  kwargs={"row_factory": dict_row}, open=True)
        try:
            self._initialize()
        except Exception:
            self.pool.close()
            raise

    def _initialize(self) -> None:
        statements = [
            """CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL,
                application TEXT NOT NULL, route TEXT NOT NULL,
                provider TEXT, model TEXT, prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0, status TEXT NOT NULL,
                latency_ms INTEGER NOT NULL DEFAULT 0, fallback_used INTEGER NOT NULL DEFAULT 0,
                cached INTEGER NOT NULL DEFAULT 0, error_type TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(occurred_at)",
            "CREATE INDEX IF NOT EXISTS idx_requests_application ON requests(application)",
            """CREATE TABLE IF NOT EXISTS budget_reservations (
                request_id TEXT PRIMARY KEY, application TEXT NOT NULL,
                amount_usd DOUBLE PRECISION NOT NULL, reserved_at TEXT NOT NULL, expires_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_budget_reservations_app ON budget_reservations(application, expires_at)",
            """CREATE TABLE IF NOT EXISTS budget_alerts (
                application TEXT NOT NULL, month_start TEXT NOT NULL,
                threshold DOUBLE PRECISION NOT NULL, emitted_at TEXT NOT NULL,
                PRIMARY KEY(application, month_start, threshold)
            )""",
        ]
        with self.pool.connection() as conn, conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

    def reserve_budget(self, request_id: str, application: str, amount_usd: float, monthly_limit_usd: float,
                       now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        month_start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = month_start.replace(year=month_start.year + (month_start.month == 12),
                                       month=1 if month_start.month == 12 else month_start.month + 1)
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cursor:
            # Serialize reservations for this app across gateway replicas.
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (application,))
            cursor.execute("DELETE FROM budget_reservations WHERE expires_at <= %s", (now.isoformat(),))
            cursor.execute("""SELECT COALESCE(SUM(cost_usd),0) AS total FROM requests
                WHERE application = %s AND occurred_at >= %s AND occurred_at < %s""",
                (application, month_start.isoformat(), month_end.isoformat()))
            spent = cursor.fetchone()["total"]
            cursor.execute("SELECT COALESCE(SUM(amount_usd),0) AS total FROM budget_reservations WHERE application = %s",
                           (application,))
            reserved = cursor.fetchone()["total"]
            if monthly_limit_usd <= 0 or spent + reserved + amount_usd > monthly_limit_usd:
                raise BudgetExceeded("Application monthly spend limit would be exceeded")
            cursor.execute("""INSERT INTO budget_reservations VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE SET application=EXCLUDED.application,
                amount_usd=EXCLUDED.amount_usd, reserved_at=EXCLUDED.reserved_at,
                expires_at=EXCLUDED.expires_at""",
                (request_id, application, amount_usd, now.isoformat(), (now + timedelta(minutes=5)).isoformat()))

    def record(self, record: dict[str, Any]) -> None:
        columns = ("request_id", "occurred_at", "application", "route", "provider", "model",
                   "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "status",
                   "latency_ms", "fallback_used", "cached", "error_type")
        assignments = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns[1:])
        placeholders = ", ".join("%s" for _ in columns)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(f"INSERT INTO requests ({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT (request_id) DO UPDATE SET {assignments}",
                           [record.get(column) for column in columns])
            cursor.execute("DELETE FROM budget_reservations WHERE request_id = %s", (record["request_id"],))

    def check_budget_alerts(self, application: str, monthly_limit_usd: float,
                            thresholds: list[float] | None = None,
                            now: datetime | None = None) -> list[dict[str, float]]:
        now = now or datetime.now(UTC)
        month_start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = month_start.replace(year=month_start.year + (month_start.month == 12),
                                       month=1 if month_start.month == 12 else month_start.month + 1)
        thresholds = thresholds or [0.8, 1.0]
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cursor:
            cursor.execute("""SELECT COALESCE(SUM(cost_usd),0) AS total FROM requests
                WHERE application = %s AND occurred_at >= %s AND occurred_at < %s""",
                (application, month_start.isoformat(), month_end.isoformat()))
            spent = cursor.fetchone()["total"]
            crossed = []
            for threshold in thresholds:
                if monthly_limit_usd <= 0 or spent < monthly_limit_usd * threshold:
                    continue
                cursor.execute("""INSERT INTO budget_alerts VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING""",
                    (application, month_start.isoformat(), threshold, now.isoformat()))
                if cursor.rowcount:
                    crossed.append({"threshold": threshold, "spend_usd": spent, "limit_usd": monthly_limit_usd})
        return crossed

    def report(self, start: str | None, end: str | None, application: str | None) -> dict[str, Any]:
        filters, values = [], []
        if start:
            filters.append("occurred_at >= %s")
            values.append(start)
        if end:
            filters.append("occurred_at < %s")
            values.append(end)
        if application:
            filters.append("application = %s")
            values.append(application)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(f"""SELECT COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                COALESCE(SUM(total_tokens),0) AS total_tokens,
                COALESCE(SUM(cost_usd),0) AS cost_usd,
                COALESCE(AVG(cost_usd),0) AS avg_cost_usd,
                COALESCE(AVG(latency_ms),0) AS avg_latency_ms,
                COALESCE(SUM(CASE WHEN status <> 'success' THEN 1 ELSE 0 END),0) AS failures,
                COALESCE(SUM(fallback_used),0) AS fallbacks,
                COALESCE(SUM(cached),0) AS cache_hits FROM requests{where}""", values)
            summary = cursor.fetchone()
            cursor.execute(f"""SELECT application, COUNT(*) AS requests,
                SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd,
                SUM(CASE WHEN status <> 'success' THEN 1 ELSE 0 END) AS failures,
                SUM(fallback_used) AS fallbacks FROM requests{where}
                GROUP BY application ORDER BY cost_usd DESC""", values)
            by_application = cursor.fetchall()
        summary["failure_rate"] = summary["failures"] / summary["requests"] if summary["requests"] else 0
        summary["fallback_rate"] = summary["fallbacks"] / summary["requests"] if summary["requests"] else 0
        summary["generated_at"] = datetime.now(UTC).isoformat()
        summary["by_application"] = by_application
        return summary

    def close(self) -> None:
        self.pool.close()
