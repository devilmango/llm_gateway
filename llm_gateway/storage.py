from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import MIN_TENANT_USAGE_RETENTION_DAYS


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
                cached INTEGER NOT NULL DEFAULT 0, error_type TEXT,
                canary_used INTEGER NOT NULL DEFAULT 0, actual_cost_usd REAL,
                cost_reconciliation_id TEXT, cost_reconciled_at TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(occurred_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_requests_application ON requests(application)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
            if "cost_center" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN cost_center TEXT")
            for column, declaration in (("canary_used", "INTEGER NOT NULL DEFAULT 0"),
                                        ("actual_cost_usd", "REAL"),
                                        ("cost_reconciliation_id", "TEXT"),
                                        ("cost_reconciled_at", "TEXT")):
                if column not in columns:
                    db.execute(f"ALTER TABLE requests ADD COLUMN {column} {declaration}")
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
            db.execute("""CREATE TABLE IF NOT EXISTS webhook_outbox (
                event_id TEXT PRIMARY KEY, application TEXT NOT NULL, webhook_url TEXT NOT NULL,
                secret_env TEXT, body TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT NOT NULL,
                created_at TEXT NOT NULL, delivered_at TEXT, last_error TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_webhook_outbox_due ON webhook_outbox(status, next_attempt_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS admin_audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL,
                actor TEXT NOT NULL, action TEXT NOT NULL, status_code INTEGER NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_admin_audit_time ON admin_audit_events(occurred_at)")

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
                   "latency_ms", "fallback_used", "cached", "error_type", "cost_center", "canary_used")
        values = [record.get(c, 0 if c == "canary_used" else None) for c in columns]
        assignments = ",".join(f"{column}=excluded.{column}" for column in columns[1:])
        with self._lock, self._connect() as db:
            db.execute(f"INSERT INTO requests ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                       f"ON CONFLICT(request_id) DO UPDATE SET {assignments}", values)
            db.execute("DELETE FROM budget_reservations WHERE request_id = ?", (record["request_id"],))

    def prune_tenant_requests(self, tenant_id: str, retention_days: int,
                              now: datetime | None = None) -> int:
        if int(retention_days) < MIN_TENANT_USAGE_RETENTION_DAYS:
            raise ValueError(f"usage_retention_days must be at least {MIN_TENANT_USAGE_RETENTION_DAYS}")
        now = now or datetime.now(UTC)
        cutoff = (now - timedelta(days=int(retention_days))).isoformat()
        prefix = f"{tenant_id}:"
        with self._lock, self._connect() as db:
            cursor = db.execute("DELETE FROM requests WHERE substr(application,1,length(?))=? AND occurred_at < ?",
                                (prefix, prefix, cutoff))
            return cursor.rowcount

    def reconcile_costs(self, records: list[dict[str, Any]], source: str,
                        reconciled_at: datetime | None = None) -> dict[str, Any]:
        reconciled_at = reconciled_at or datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for item in records:
                cursor = db.execute("""UPDATE requests SET actual_cost_usd=?, cost_reconciliation_id=?,
                    cost_reconciled_at=? WHERE request_id=?""",
                    (float(item["actual_cost_usd"]), source, reconciled_at.isoformat(), item["request_id"]))
                if cursor.rowcount != 1:
                    raise ValueError(f"Unknown request_id {item['request_id']!r}; batch was not applied")
        return {"source": source, "reconciled_requests": len(records), "reconciled_at": reconciled_at.isoformat()}

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

    def enqueue_webhook(self, event: dict[str, Any], webhook_url: str, secret_env: str | None = None,
                        now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        import json
        with self._lock, self._connect() as db:
            db.execute("""INSERT OR IGNORE INTO webhook_outbox
                (event_id, application, webhook_url, secret_env, body, status, attempts,
                 next_attempt_at, created_at) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
                (event["alert_id"], event["application"], webhook_url, secret_env,
                 json.dumps(event, separators=(",", ":"), sort_keys=True), now.isoformat(), now.isoformat()))

    def due_webhooks(self, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            due = db.execute("""SELECT event_id FROM webhook_outbox
                WHERE status IN ('pending','processing') AND next_attempt_at <= ? ORDER BY created_at LIMIT ?""",
                (now.isoformat(), limit)).fetchall()
            ids = [row[0] for row in due]
            for event_id in ids:
                db.execute("UPDATE webhook_outbox SET status='processing', next_attempt_at=? WHERE event_id=?",
                           ((now + timedelta(seconds=60)).isoformat(), event_id))
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            return [dict(row) for row in db.execute(f"SELECT * FROM webhook_outbox WHERE event_id IN ({placeholders})", ids).fetchall()]

    def update_webhook(self, event_id: str, *, attempts: int, status: str,
                       next_attempt_at: datetime | None = None, delivered_at: datetime | None = None,
                       last_error: str | None = None) -> None:
        with self._lock, self._connect() as db:
            db.execute("""UPDATE webhook_outbox SET attempts=?, status=?, next_attempt_at=?,
                delivered_at=?, last_error=? WHERE event_id=?""",
                (attempts, status, (next_attempt_at or datetime.now(UTC)).isoformat(),
                 delivered_at.isoformat() if delivered_at else None, last_error, event_id))

    def redrive_webhook(self, event_id: str, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            cursor = db.execute("""UPDATE webhook_outbox SET status='pending', attempts=0,
                next_attempt_at=?, delivered_at=NULL, last_error=NULL
                WHERE event_id=? AND status='dead_letter'""", (now.isoformat(), event_id))
            return bool(cursor.rowcount)

    def webhook_events(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            if status:
                rows = db.execute("SELECT event_id, application, status, attempts, created_at, delivered_at, last_error FROM webhook_outbox WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
            else:
                rows = db.execute("SELECT event_id, application, status, attempts, created_at, delivered_at, last_error FROM webhook_outbox ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]

    def record_audit(self, actor: str, action: str, status_code: int,
                     now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO admin_audit_events (occurred_at, actor, action, status_code) VALUES (?,?,?,?)",
                       (now.isoformat(), actor, action, status_code))

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT occurred_at, actor, action, status_code FROM admin_audit_events ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()]

    @staticmethod
    def _filters(start: str | None, end: str | None, application: str | None,
                 cost_center: str | None, provider: str | None, route: str | None) -> tuple[str, list[Any]]:
        clauses, values = [], []
        if start:
            clauses.append("occurred_at >= ?"); values.append(start)
        if end:
            clauses.append("occurred_at < ?"); values.append(end)
        if application:
            clauses.append("application = ?"); values.append(application)
        if cost_center:
            clauses.append("cost_center = ?"); values.append(cost_center)
        if provider:
            clauses.append("provider = ?"); values.append(provider)
        if route:
            clauses.append("route = ?"); values.append(route)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return where, values

    def report(self, start: str | None, end: str | None, application: str | None,
               cost_center: str | None = None, provider: str | None = None,
               route: str | None = None) -> dict[str, Any]:
        where, values = self._filters(start, end, application, cost_center, provider, route)
        with self._lock, self._connect() as db:
            summary = dict(db.execute(f"""SELECT COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                COALESCE(SUM(total_tokens),0) AS total_tokens,
                COALESCE(SUM(cost_usd),0) AS cost_usd,
                COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd,
                COALESCE(SUM(CASE WHEN actual_cost_usd IS NOT NULL THEN actual_cost_usd-cost_usd ELSE 0 END),0) AS cost_variance_usd,
                COALESCE(SUM(actual_cost_usd IS NOT NULL),0) AS reconciled_requests,
                COALESCE(AVG(cost_usd),0) AS avg_cost_usd,
                COALESCE(AVG(latency_ms),0) AS avg_latency_ms,
                COALESCE(SUM(status != 'success'),0) AS failures,
                COALESCE(SUM(fallback_used),0) AS fallbacks,
                COALESCE(SUM(cached),0) AS cache_hits
                ,COALESCE(SUM(canary_used),0) AS canary_requests
                FROM requests{where}""", values).fetchone())
            by_app = [dict(row) for row in db.execute(f"""SELECT application,
                COUNT(*) AS requests, SUM(total_tokens) AS total_tokens,
                SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd,
                SUM(status != 'success') AS failures,
                SUM(fallback_used) AS fallbacks FROM requests{where}
                GROUP BY application ORDER BY cost_usd DESC""", values).fetchall()]
            by_provider = [dict(row) for row in db.execute(f"SELECT COALESCE(provider,'unknown') AS provider, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY provider ORDER BY cost_usd DESC", values).fetchall()]
            by_model = [dict(row) for row in db.execute(f"SELECT COALESCE(model,'unknown') AS model, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY model ORDER BY cost_usd DESC", values).fetchall()]
            by_route = [dict(row) for row in db.execute(f"SELECT route, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY route ORDER BY cost_usd DESC", values).fetchall()]
            by_cost_center = [dict(row) for row in db.execute(f"SELECT COALESCE(cost_center,'unassigned') AS cost_center, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY cost_center ORDER BY cost_usd DESC", values).fetchall()]
        summary["failure_rate"] = summary["failures"] / summary["requests"] if summary["requests"] else 0
        summary["fallback_rate"] = summary["fallbacks"] / summary["requests"] if summary["requests"] else 0
        summary["generated_at"] = datetime.now(UTC).isoformat()
        summary["by_application"] = by_app
        summary["by_provider"] = by_provider
        summary["by_model"] = by_model
        summary["by_route"] = by_route
        summary["by_cost_center"] = by_cost_center
        return summary

    def export_rows(self, start: str | None, end: str | None, application: str | None,
                    cost_center: str | None = None, provider: str | None = None,
                    route: str | None = None, limit: int = 10_000) -> list[dict[str, Any]]:
        where, values = self._filters(start, end, application, cost_center, provider, route)
        with self._lock, self._connect() as db:
            return [dict(row) for row in db.execute(
                f"SELECT * FROM requests{where} ORDER BY occurred_at DESC LIMIT ?", [*values, limit]).fetchall()]


def create_usage_store(database: str) -> Any:
    if database.startswith(("postgres://", "postgresql://")):
        from .postgres_storage import PostgresUsageStore

        return PostgresUsageStore(database)
    return UsageStore(database)
