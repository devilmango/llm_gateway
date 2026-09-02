from __future__ import annotations

import json
import sqlite3
import threading
import uuid
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
                cost_reconciliation_id TEXT, cost_reconciled_at TEXT,
                pricing_catalog TEXT NOT NULL DEFAULT 'default', pricing_version TEXT NOT NULL DEFAULT 'unversioned'
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(occurred_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_requests_application ON requests(application)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
            if "cost_center" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN cost_center TEXT")
            for column, declaration in (("canary_used", "INTEGER NOT NULL DEFAULT 0"),
                                        ("actual_cost_usd", "REAL"),
                                        ("cost_reconciliation_id", "TEXT"),
                                        ("cost_reconciled_at", "TEXT"),
                                        ("pricing_catalog", "TEXT NOT NULL DEFAULT 'default'"),
                                        ("pricing_version", "TEXT NOT NULL DEFAULT 'unversioned'")):
                if column not in columns:
                    db.execute(f"ALTER TABLE requests ADD COLUMN {column} {declaration}")
            db.execute("CREATE INDEX IF NOT EXISTS idx_requests_anomaly_dimensions ON requests(occurred_at, application, provider, model, pricing_version)")
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
            db.execute("""CREATE TABLE IF NOT EXISTS tool_call_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
                tenant_id TEXT, application TEXT NOT NULL, occurred_at TEXT NOT NULL,
                tool_name TEXT NOT NULL, outcome TEXT NOT NULL, reason TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_tool_audit_time ON tool_call_audits(occurred_at)")
            audit_columns = {row[1] for row in db.execute("PRAGMA table_info(tool_call_audits)")}
            if "tool_call_id" not in audit_columns:
                db.execute("ALTER TABLE tool_call_audits ADD COLUMN tool_call_id TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS idx_tool_audit_execution ON tool_call_audits(request_id, tool_call_id, outcome)")
            db.execute("""CREATE TABLE IF NOT EXISTS tool_execution_outcomes (
                request_id TEXT NOT NULL, tool_call_id TEXT NOT NULL, tenant_id TEXT,
                application TEXT NOT NULL, tool_name TEXT NOT NULL, outcome TEXT NOT NULL,
                error_code TEXT, occurred_at TEXT NOT NULL,
                PRIMARY KEY(request_id, tool_call_id)
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_tool_execution_time ON tool_execution_outcomes(occurred_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS routing_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL, application TEXT NOT NULL, route TEXT NOT NULL,
                sequence INTEGER NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
                decision TEXT NOT NULL, reason TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(request_id, sequence)
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_routing_decisions_report ON routing_decisions(occurred_at, application, route, decision)")
            db.execute("""CREATE TABLE IF NOT EXISTS evaluation_datasets (
                dataset_name TEXT NOT NULL, dataset_version TEXT NOT NULL, dataset_sha256 TEXT NOT NULL,
                case_count INTEGER NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(dataset_name, dataset_version)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY, dataset_name TEXT NOT NULL, dataset_version TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL, report TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_time ON evaluation_runs(created_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS evaluation_comparisons (
                comparison_id TEXT PRIMARY KEY, baseline_run_id TEXT NOT NULL,
                candidate_run_id TEXT NOT NULL, report TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_comparisons_time ON evaluation_comparisons(created_at)")

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
                   "pricing_catalog", "pricing_version", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "status",
                   "latency_ms", "fallback_used", "cached", "error_type", "cost_center", "canary_used")
        defaults = {"canary_used": 0, "pricing_catalog": "default", "pricing_version": "unversioned"}
        values = [record.get(c, defaults.get(c)) for c in columns]
        assignments = ",".join(f"{column}=excluded.{column}" for column in columns[1:])
        with self._lock, self._connect() as db:
            db.execute(f"INSERT INTO requests ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                       f"ON CONFLICT(request_id) DO UPDATE SET {assignments}", values)
            db.execute("DELETE FROM budget_reservations WHERE request_id = ?", (record["request_id"],))

    def record_routing_decisions(self, request_id: str, application: str, route: str,
                                 decisions: list[dict[str, Any]],
                                 occurred_at: datetime | None = None) -> None:
        if not decisions:
            return
        occurred_at = occurred_at or datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM routing_decisions WHERE request_id=?", (request_id,))
            db.executemany("""INSERT INTO routing_decisions
                (request_id, occurred_at, application, route, sequence, provider, model,
                 decision, reason, attempt_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(request_id, occurred_at.isoformat(), application, route, index,
                  item["provider"], item["model"], item["decision"], item["reason"],
                  int(item.get("attempt_count", 0))) for index, item in enumerate(decisions)])

    def routing_decision_events(self, request_id: str | None = None, application: str | None = None,
                                route: str | None = None, provider: str | None = None,
                                decision: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        filters, values = [], []
        for field, value in (("request_id", request_id), ("application", application),
                             ("route", route), ("provider", provider), ("decision", decision)):
            if value is not None:
                filters.append(f"{field} = ?")
                values.append(value)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self._lock, self._connect() as db:
            rows = db.execute(f"""SELECT request_id, occurred_at, application, route, sequence,
                provider, model, decision, reason, attempt_count FROM routing_decisions{where}
                ORDER BY occurred_at DESC, sequence ASC LIMIT ?""", [*values, limit]).fetchall()
            return [dict(row) for row in rows]

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

    def record_tool_call_audits(self, request_id: str, tenant_id: str | None, application: str,
                                events: list[dict[str, Any]]) -> None:
        if not events:
            return
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.executemany("""INSERT INTO tool_call_audits
                (request_id, tenant_id, application, occurred_at, tool_call_id, tool_name, outcome, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(request_id, tenant_id, application, now, event.get("tool_call_id"), event["tool_name"],
                  event["outcome"], event.get("reason")) for event in events])

    def tool_call_audit_events(self, tenant_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            if tenant_id is None:
                rows = db.execute("""SELECT request_id, tenant_id, application, occurred_at,
                    tool_call_id, tool_name, outcome, reason FROM tool_call_audits ORDER BY id DESC LIMIT ?""", (limit,))
            else:
                rows = db.execute("""SELECT request_id, tenant_id, application, occurred_at,
                    tool_call_id, tool_name, outcome, reason FROM tool_call_audits WHERE tenant_id = ?
                    ORDER BY id DESC LIMIT ?""", (tenant_id, limit))
            return [dict(row) for row in rows.fetchall()]

    def pending_tool_execution_calls(self, timeout_seconds: int, tenant_id: str | None = None,
                                     limit: int = 100, now: datetime | None = None) -> dict[str, Any]:
        current = now or datetime.now(UTC)
        cutoff = (current - timedelta(seconds=timeout_seconds)).isoformat()
        tenant_clause = " AND a.tenant_id = ?" if tenant_id is not None else ""
        values: tuple[Any, ...] = (tenant_id,) if tenant_id is not None else ()
        with self._lock, self._connect() as db:
            counts = db.execute(f"""SELECT COUNT(*) AS pending_count,
                SUM(CASE WHEN a.occurred_at <= ? THEN 1 ELSE 0 END) AS overdue_count
                FROM tool_call_audits a LEFT JOIN tool_execution_outcomes o
                ON o.request_id=a.request_id AND o.tool_call_id=a.tool_call_id
                WHERE a.outcome='allowed' AND a.tool_call_id IS NOT NULL AND o.request_id IS NULL{tenant_clause}""",
                (cutoff, *values)).fetchone()
            rows = db.execute(f"""SELECT a.request_id, a.tool_call_id, a.tenant_id, a.application,
                a.occurred_at, a.tool_name FROM tool_call_audits a LEFT JOIN tool_execution_outcomes o
                ON o.request_id=a.request_id AND o.tool_call_id=a.tool_call_id
                WHERE a.outcome='allowed' AND a.tool_call_id IS NOT NULL AND o.request_id IS NULL{tenant_clause}
                ORDER BY a.occurred_at ASC LIMIT ?""", (*values, limit)).fetchall()
        calls = []
        for row in rows:
            call = dict(row)
            call["age_seconds"] = max(0, round((current - datetime.fromisoformat(call["occurred_at"])).total_seconds(), 2))
            call["overdue"] = call["occurred_at"] <= cutoff
            calls.append(call)
        return {"pending_count": counts["pending_count"] or 0, "overdue_count": counts["overdue_count"] or 0,
                "calls": calls}

    def record_tool_execution_outcome(self, request_id: str, tool_call_id: str, tenant_id: str | None,
                                      application: str, outcome: str, error_code: str | None = None) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            audit = db.execute("""SELECT tenant_id, application, tool_name FROM tool_call_audits
                WHERE request_id=? AND tool_call_id=? AND outcome='allowed'
                ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
            if audit is None or audit["tenant_id"] != tenant_id or audit["application"] != application:
                raise ValueError("No authorized generated tool call matches this request")
            existing = db.execute("""SELECT outcome, error_code FROM tool_execution_outcomes
                WHERE request_id=? AND tool_call_id=?""", (request_id, tool_call_id)).fetchone()
            if existing:
                if existing["outcome"] != outcome or existing["error_code"] != error_code:
                    raise ValueError("A different execution outcome is already recorded for this tool call")
                return {"request_id": request_id, "tool_call_id": tool_call_id,
                        "tool_name": audit["tool_name"], "outcome": outcome, "duplicate": True}
            db.execute("""INSERT INTO tool_execution_outcomes
                (request_id, tool_call_id, tenant_id, application, tool_name, outcome, error_code, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (request_id, tool_call_id, tenant_id, application, audit["tool_name"], outcome, error_code, now))
        return {"request_id": request_id, "tool_call_id": tool_call_id,
                "tool_name": audit["tool_name"], "outcome": outcome, "duplicate": False}

    def tool_execution_outcomes(self, tenant_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            if tenant_id is None:
                rows = db.execute("""SELECT request_id, tool_call_id, tenant_id, application,
                    tool_name, outcome, error_code, occurred_at FROM tool_execution_outcomes
                    ORDER BY occurred_at DESC LIMIT ?""", (limit,))
            else:
                rows = db.execute("""SELECT request_id, tool_call_id, tenant_id, application,
                    tool_name, outcome, error_code, occurred_at FROM tool_execution_outcomes
                    WHERE tenant_id=? ORDER BY occurred_at DESC LIMIT ?""", (tenant_id, limit))
            return [dict(row) for row in rows.fetchall()]

    def save_evaluation_report(self, dataset_name: str, dataset_version: str, dataset_sha256: str,
                              case_count: int, report: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        run_id = "eval_" + uuid.uuid4().hex
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""INSERT OR IGNORE INTO evaluation_datasets
                (dataset_name, dataset_version, dataset_sha256, case_count, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (dataset_name, dataset_version, dataset_sha256, case_count, now))
            dataset = db.execute("""SELECT dataset_sha256, case_count FROM evaluation_datasets
                WHERE dataset_name=? AND dataset_version=?""", (dataset_name, dataset_version)).fetchone()
            if dataset["dataset_sha256"] != dataset_sha256 or dataset["case_count"] != case_count:
                raise ValueError("Dataset name/version already exists with different content")
            db.execute("""INSERT INTO evaluation_runs
                (run_id, dataset_name, dataset_version, dataset_sha256, report, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, dataset_name, dataset_version, dataset_sha256,
                 json.dumps(report, separators=(",", ":"), sort_keys=True), now))
        return {"run_id": run_id, "dataset_name": dataset_name, "dataset_version": dataset_version,
                "dataset_sha256": dataset_sha256, "case_count": case_count, "created_at": now}

    def evaluation_reports(self, dataset_name: str | None = None, dataset_version: str | None = None,
                           limit: int = 100) -> list[dict[str, Any]]:
        clauses, values = [], []
        if dataset_name:
            clauses.append("dataset_name=?")
            values.append(dataset_name)
        if dataset_version:
            clauses.append("dataset_version=?")
            values.append(dataset_version)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock, self._connect() as db:
            rows = db.execute(f"""SELECT run_id, dataset_name, dataset_version, dataset_sha256,
                report, created_at FROM evaluation_runs{where} ORDER BY created_at DESC LIMIT ?""",
                [*values, limit]).fetchall()
            return [{**dict(row), "report": json.loads(row["report"])} for row in rows]

    def evaluation_report(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("""SELECT run_id, dataset_name, dataset_version, dataset_sha256,
                report, created_at FROM evaluation_runs WHERE run_id=?""", (run_id,)).fetchone()
            return {**dict(row), "report": json.loads(row["report"])} if row else None

    def save_evaluation_comparison(self, baseline_run_id: str, candidate_run_id: str,
                                   report: dict[str, Any]) -> dict[str, Any]:
        comparison_id = "cmp_" + uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.execute("""INSERT INTO evaluation_comparisons
                (comparison_id, baseline_run_id, candidate_run_id, report, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (comparison_id, baseline_run_id, candidate_run_id,
                 json.dumps(report, separators=(",", ":"), sort_keys=True), created_at))
        return {"comparison_id": comparison_id, "created_at": created_at}

    def evaluation_comparisons(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT comparison_id, baseline_run_id, candidate_run_id,
                report, created_at FROM evaluation_comparisons ORDER BY created_at DESC LIMIT ?""",
                (limit,)).fetchall()
            return [{**dict(row), "report": json.loads(row["report"])} for row in rows]

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
            variance = "COALESCE(SUM(CASE WHEN actual_cost_usd IS NOT NULL THEN actual_cost_usd-cost_usd ELSE 0 END),0) AS cost_variance_usd"
            by_provider = [dict(row) for row in db.execute(f"SELECT COALESCE(provider,'unknown') AS provider, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd, {variance} FROM requests{where} GROUP BY provider ORDER BY cost_usd DESC", values).fetchall()]
            by_model = [dict(row) for row in db.execute(f"SELECT COALESCE(model,'unknown') AS model, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd, {variance} FROM requests{where} GROUP BY model ORDER BY cost_usd DESC", values).fetchall()]
            by_pricing_catalog = [dict(row) for row in db.execute(f"SELECT COALESCE(provider,'unknown') AS provider, COALESCE(model,'unknown') AS model, pricing_catalog, pricing_version, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd, {variance} FROM requests{where} GROUP BY provider, model, pricing_catalog, pricing_version ORDER BY cost_usd DESC", values).fetchall()]
            by_route = [dict(row) for row in db.execute(f"SELECT route, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY route ORDER BY cost_usd DESC", values).fetchall()]
            by_cost_center = [dict(row) for row in db.execute(f"SELECT COALESCE(cost_center,'unassigned') AS cost_center, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY cost_center ORDER BY cost_usd DESC", values).fetchall()]
        summary["failure_rate"] = summary["failures"] / summary["requests"] if summary["requests"] else 0
        summary["fallback_rate"] = summary["fallbacks"] / summary["requests"] if summary["requests"] else 0
        summary["generated_at"] = datetime.now(UTC).isoformat()
        summary["by_application"] = by_app
        summary["by_provider"] = by_provider
        summary["by_model"] = by_model
        summary["by_pricing_catalog"] = by_pricing_catalog
        summary["by_route"] = by_route
        summary["by_cost_center"] = by_cost_center
        decision_filters, decision_values = [], []
        if start:
            decision_filters.append("occurred_at >= ?"); decision_values.append(start)
        if end:
            decision_filters.append("occurred_at < ?"); decision_values.append(end)
        if application:
            decision_filters.append("application = ?"); decision_values.append(application)
        if route:
            decision_filters.append("route = ?"); decision_values.append(route)
        if provider:
            decision_filters.append("provider = ?"); decision_values.append(provider)
        decision_where = " WHERE " + " AND ".join(decision_filters) if decision_filters else ""
        with self._lock, self._connect() as db:
            summary["by_routing_decision"] = [dict(row) for row in db.execute(f"""SELECT decision, reason,
                provider, model, COUNT(*) AS decision_events FROM routing_decisions{decision_where}
                GROUP BY decision, reason, provider, model ORDER BY decision_events DESC""",
                decision_values).fetchall()]
        return summary

    def slo_observations(self, start: str, end: str) -> list[dict[str, Any]]:
        """Return request outcome/latency distributions grouped by application and route."""
        with self._lock, self._connect() as db:
            return [dict(row) for row in db.execute("""SELECT application, route, status,
                latency_ms, COUNT(*) AS requests FROM requests
                WHERE occurred_at >= ? AND occurred_at < ?
                GROUP BY application, route, status, latency_ms
                ORDER BY application, route, status, latency_ms""", (start, end)).fetchall()]

    def cost_anomalies(self, window_hours: int = 24, threshold_percent: float = 50,
                       min_requests: int = 5, now: datetime | None = None) -> dict[str, Any]:
        current_end = now or datetime.now(UTC)
        current_start = current_end - timedelta(hours=window_hours)
        baseline_start = current_start - timedelta(hours=window_hours)
        query = """SELECT application, COALESCE(provider, 'unknown') AS provider,
            COALESCE(model, 'unknown') AS model, pricing_version,
            SUM(CASE WHEN occurred_at >= ? AND occurred_at < ? THEN 1 ELSE 0 END) AS baseline_requests,
            SUM(CASE WHEN occurred_at >= ? AND occurred_at < ? THEN 1 ELSE 0 END) AS current_requests,
            SUM(CASE WHEN occurred_at >= ? AND occurred_at < ? THEN cost_usd ELSE 0 END) AS baseline_cost_usd,
            SUM(CASE WHEN occurred_at >= ? AND occurred_at < ? THEN cost_usd ELSE 0 END) AS current_cost_usd,
            SUM(CASE WHEN occurred_at >= ? AND occurred_at < ? THEN total_tokens ELSE 0 END) AS baseline_tokens,
            SUM(CASE WHEN occurred_at >= ? AND occurred_at < ? THEN total_tokens ELSE 0 END) AS current_tokens
            FROM requests WHERE occurred_at >= ? AND occurred_at < ? AND status='success'
            GROUP BY application, provider, model, pricing_version"""
        params = (baseline_start.isoformat(), current_start.isoformat(),
                  current_start.isoformat(), current_end.isoformat(),
                  baseline_start.isoformat(), current_start.isoformat(),
                  current_start.isoformat(), current_end.isoformat(),
                  baseline_start.isoformat(), current_start.isoformat(),
                  current_start.isoformat(), current_end.isoformat(),
                  baseline_start.isoformat(), current_end.isoformat())
        with self._lock, self._connect() as db:
            rows = [dict(row) for row in db.execute(query, params).fetchall()]
        comparisons = []
        for row in rows:
            baseline_n, current_n = int(row["baseline_requests"]), int(row["current_requests"])
            if baseline_n < min_requests or current_n < min_requests:
                continue
            baseline_cost, current_cost = float(row["baseline_cost_usd"]), float(row["current_cost_usd"])
            baseline_tokens, current_tokens = int(row["baseline_tokens"]), int(row["current_tokens"])
            before_request = baseline_cost / baseline_n
            after_request = current_cost / current_n
            before_token = baseline_cost / baseline_tokens * 1000 if baseline_tokens else 0.0
            after_token = current_cost / current_tokens * 1000 if current_tokens else 0.0

            def change_percent(before: float, after: float) -> float | str:
                if before == 0:
                    return "infinite" if after > 0 else 0.0
                return round((after - before) / before * 100, 2)

            request_change = change_percent(before_request, after_request)
            token_change = change_percent(before_token, after_token)
            request_anomaly = request_change == "infinite" or abs(request_change) > threshold_percent
            token_anomaly = token_change == "infinite" or abs(token_change) > threshold_percent
            comparisons.append({
                "application": row["application"], "provider": row["provider"], "model": row["model"],
                "pricing_version": row["pricing_version"],
                "baseline": {"requests": baseline_n, "tokens": baseline_tokens,
                    "cost_usd": round(baseline_cost, 8), "cost_per_request_usd": round(before_request, 8),
                    "cost_per_1k_tokens_usd": round(before_token, 8)},
                "current": {"requests": current_n, "tokens": current_tokens,
                    "cost_usd": round(current_cost, 8), "cost_per_request_usd": round(after_request, 8),
                    "cost_per_1k_tokens_usd": round(after_token, 8)},
                "change_percent": {"per_request": request_change, "per_1k_tokens": token_change},
                "anomaly": request_anomaly or token_anomaly,
                "anomaly_dimensions": [name for name, flagged in (("per_request", request_anomaly),
                    ("per_1k_tokens", token_anomaly)) if flagged],
            })
        comparisons.sort(key=lambda item: item["anomaly"], reverse=True)
        return {"window_hours": window_hours, "threshold_percent": threshold_percent,
            "min_requests_per_window": min_requests, "baseline_start": baseline_start.isoformat(),
            "current_start": current_start.isoformat(), "current_end": current_end.isoformat(),
            "comparisons": comparisons, "anomalies": [item for item in comparisons if item["anomaly"]]}

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
