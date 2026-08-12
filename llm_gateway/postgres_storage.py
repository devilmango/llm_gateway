from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import MIN_TENANT_USAGE_RETENTION_DAYS
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
                cached INTEGER NOT NULL DEFAULT 0, error_type TEXT,
                canary_used INTEGER NOT NULL DEFAULT 0, actual_cost_usd DOUBLE PRECISION,
                cost_reconciliation_id TEXT, cost_reconciled_at TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(occurred_at)",
            "CREATE INDEX IF NOT EXISTS idx_requests_application ON requests(application)",
            "ALTER TABLE requests ADD COLUMN IF NOT EXISTS cost_center TEXT",
            "ALTER TABLE requests ADD COLUMN IF NOT EXISTS canary_used INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN IF NOT EXISTS actual_cost_usd DOUBLE PRECISION",
            "ALTER TABLE requests ADD COLUMN IF NOT EXISTS cost_reconciliation_id TEXT",
            "ALTER TABLE requests ADD COLUMN IF NOT EXISTS cost_reconciled_at TEXT",
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
            """CREATE TABLE IF NOT EXISTS webhook_outbox (
                event_id TEXT PRIMARY KEY, application TEXT NOT NULL, webhook_url TEXT NOT NULL,
                secret_env TEXT, body TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT NOT NULL,
                created_at TEXT NOT NULL, delivered_at TEXT, last_error TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_webhook_outbox_due ON webhook_outbox(status, next_attempt_at)",
            """CREATE TABLE IF NOT EXISTS admin_audit_events (
                id BIGSERIAL PRIMARY KEY, occurred_at TEXT NOT NULL,
                actor TEXT NOT NULL, action TEXT NOT NULL, status_code INTEGER NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_admin_audit_time ON admin_audit_events(occurred_at)",
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
                   "latency_ms", "fallback_used", "cached", "error_type", "cost_center", "canary_used")
        assignments = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns[1:])
        placeholders = ", ".join("%s" for _ in columns)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(f"INSERT INTO requests ({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT (request_id) DO UPDATE SET {assignments}",
                           [record.get(column, 0 if column == "canary_used" else None) for column in columns])
            cursor.execute("DELETE FROM budget_reservations WHERE request_id = %s", (record["request_id"],))

    def reconcile_costs(self, records: list[dict[str, Any]], source: str,
                        reconciled_at: datetime | None = None) -> dict[str, Any]:
        reconciled_at = reconciled_at or datetime.now(UTC)
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cursor:
            for item in records:
                cursor.execute("""UPDATE requests SET actual_cost_usd=%s, cost_reconciliation_id=%s,
                    cost_reconciled_at=%s WHERE request_id=%s""",
                    (float(item["actual_cost_usd"]), source, reconciled_at.isoformat(), item["request_id"]))
                if cursor.rowcount != 1:
                    raise ValueError(f"Unknown request_id {item['request_id']!r}; batch was not applied")
        return {"source": source, "reconciled_requests": len(records), "reconciled_at": reconciled_at.isoformat()}

    def prune_tenant_requests(self, tenant_id: str, retention_days: int,
                              now: datetime | None = None) -> int:
        if int(retention_days) < MIN_TENANT_USAGE_RETENTION_DAYS:
            raise ValueError(f"usage_retention_days must be at least {MIN_TENANT_USAGE_RETENTION_DAYS}")
        now = now or datetime.now(UTC)
        cutoff = (now - timedelta(days=int(retention_days))).isoformat()
        prefix = f"{tenant_id}:"
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cursor:
            cursor.execute("DELETE FROM requests WHERE left(application, length(%s))=%s AND occurred_at < %s",
                           (prefix, prefix, cutoff))
            return cursor.rowcount

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

    def enqueue_webhook(self, event: dict[str, Any], webhook_url: str, secret_env: str | None = None,
                        now: datetime | None = None) -> None:
        import json
        now = now or datetime.now(UTC)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""INSERT INTO webhook_outbox
                (event_id, application, webhook_url, secret_env, body, status, attempts,
                 next_attempt_at, created_at) VALUES (%s,%s,%s,%s,%s,'pending',0,%s,%s)
                ON CONFLICT (event_id) DO NOTHING""",
                (event["alert_id"], event["application"], webhook_url, secret_env,
                 json.dumps(event, separators=(",", ":"), sort_keys=True), now.isoformat(), now.isoformat()))

    def due_webhooks(self, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(UTC)
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cursor:
            cursor.execute("""WITH due AS (
                SELECT event_id FROM webhook_outbox WHERE status IN ('pending','processing') AND next_attempt_at <= %s
                ORDER BY created_at LIMIT %s FOR UPDATE SKIP LOCKED
            ) UPDATE webhook_outbox AS w SET status='processing', next_attempt_at=%s
              FROM due WHERE w.event_id=due.event_id RETURNING w.*""",
                (now.isoformat(), limit, (now + timedelta(seconds=60)).isoformat()))
            return cursor.fetchall()

    def update_webhook(self, event_id: str, *, attempts: int, status: str,
                       next_attempt_at: datetime | None = None, delivered_at: datetime | None = None,
                       last_error: str | None = None) -> None:
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""UPDATE webhook_outbox SET attempts=%s,status=%s,next_attempt_at=%s,
                delivered_at=%s,last_error=%s WHERE event_id=%s""",
                (attempts, status, (next_attempt_at or datetime.now(UTC)).isoformat(),
                 delivered_at.isoformat() if delivered_at else None, last_error, event_id))

    def redrive_webhook(self, event_id: str, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("""UPDATE webhook_outbox SET status='pending', attempts=0, next_attempt_at=%s,
                delivered_at=NULL,last_error=NULL WHERE event_id=%s AND status='dead_letter'""",
                (now.isoformat(), event_id))
            return bool(cursor.rowcount)

    def webhook_events(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.pool.connection() as conn, conn.cursor() as cursor:
            if status:
                cursor.execute("SELECT event_id,application,status,attempts,created_at,delivered_at,last_error FROM webhook_outbox WHERE status=%s ORDER BY created_at DESC LIMIT %s", (status, limit))
            else:
                cursor.execute("SELECT event_id,application,status,attempts,created_at,delivered_at,last_error FROM webhook_outbox ORDER BY created_at DESC LIMIT %s", (limit,))
            return cursor.fetchall()

    def record_audit(self, actor: str, action: str, status_code: int,
                     now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("INSERT INTO admin_audit_events (occurred_at,actor,action,status_code) VALUES (%s,%s,%s,%s)",
                           (now.isoformat(), actor, action, status_code))

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT occurred_at,actor,action,status_code FROM admin_audit_events ORDER BY id DESC LIMIT %s",
                           (limit,))
            return cursor.fetchall()

    @staticmethod
    def _filters(start: str | None, end: str | None, application: str | None,
                 cost_center: str | None, provider: str | None, route: str | None) -> tuple[str, list[Any]]:
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
        if cost_center:
            filters.append("cost_center = %s")
            values.append(cost_center)
        if provider:
            filters.append("provider = %s")
            values.append(provider)
        if route:
            filters.append("route = %s")
            values.append(route)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        return where, values

    def report(self, start: str | None, end: str | None, application: str | None,
               cost_center: str | None = None, provider: str | None = None,
               route: str | None = None) -> dict[str, Any]:
        where, values = self._filters(start, end, application, cost_center, provider, route)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(f"""SELECT COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                COALESCE(SUM(total_tokens),0) AS total_tokens,
                COALESCE(SUM(cost_usd),0) AS cost_usd,
                COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd,
                COALESCE(SUM(CASE WHEN actual_cost_usd IS NOT NULL THEN actual_cost_usd-cost_usd ELSE 0 END),0) AS cost_variance_usd,
                COALESCE(SUM(CASE WHEN actual_cost_usd IS NOT NULL THEN 1 ELSE 0 END),0) AS reconciled_requests,
                COALESCE(AVG(cost_usd),0) AS avg_cost_usd,
                COALESCE(AVG(latency_ms),0) AS avg_latency_ms,
                COALESCE(SUM(CASE WHEN status <> 'success' THEN 1 ELSE 0 END),0) AS failures,
                COALESCE(SUM(fallback_used),0) AS fallbacks,
                COALESCE(SUM(cached),0) AS cache_hits,
                COALESCE(SUM(canary_used),0) AS canary_requests FROM requests{where}""", values)
            summary = cursor.fetchone()
            cursor.execute(f"""SELECT application, COUNT(*) AS requests,
                SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd,
                COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd,
                SUM(CASE WHEN status <> 'success' THEN 1 ELSE 0 END) AS failures,
                SUM(fallback_used) AS fallbacks FROM requests{where}
                GROUP BY application ORDER BY cost_usd DESC""", values)
            by_application = cursor.fetchall()
            breakdowns = {}
            for dimension in ("provider", "model", "route", "cost_center"):
                cursor.execute(f"SELECT COALESCE({dimension}::text,'unassigned') AS {dimension}, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY {dimension} ORDER BY cost_usd DESC", values)
                breakdowns[f"by_{dimension}"] = cursor.fetchall()
        summary["failure_rate"] = summary["failures"] / summary["requests"] if summary["requests"] else 0
        summary["fallback_rate"] = summary["fallbacks"] / summary["requests"] if summary["requests"] else 0
        summary["generated_at"] = datetime.now(UTC).isoformat()
        summary["by_application"] = by_application
        summary.update(breakdowns)
        return summary

    def export_rows(self, start: str | None, end: str | None, application: str | None,
                    cost_center: str | None = None, provider: str | None = None,
                    route: str | None = None, limit: int = 10_000) -> list[dict[str, Any]]:
        where, values = self._filters(start, end, application, cost_center, provider, route)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM requests{where} ORDER BY occurred_at DESC LIMIT %s", [*values, limit])
            return cursor.fetchall()

    def close(self) -> None:
        self.pool.close()
