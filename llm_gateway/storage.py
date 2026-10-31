from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import MIN_TENANT_USAGE_RETENTION_DAYS
from .webhook_security import webhook_endpoint_id


class BudgetExceeded(Exception):
    pass


class AsyncJobIdempotencyConflict(ValueError):
    """An idempotency key was reused for a different job request."""


class AsyncJobQueueLimitExceeded(ValueError):
    """The tenant has reached its configured queued job limit."""


class WebhookEndpointBusy(ValueError):
    """An endpoint circuit cannot be reset while a delivery is in flight."""


AUDIT_GENESIS_HASH = "0" * 64


def audit_entry_hash(event_id: int, occurred_at: str, actor: str, action: str,
                     status_code: int, previous_hash: str) -> str:
    payload = json.dumps({"id": event_id, "occurred_at": occurred_at, "actor": actor,
        "action": action, "status_code": int(status_code), "previous_hash": previous_hash},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


class UsageStore:
    """Small durable usage ledger. Stores metadata only, never prompt/response text."""

    def __init__(self, database: str) -> None:
        self.database = database
        self._lock = threading.RLock()
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
                pricing_catalog TEXT NOT NULL DEFAULT 'default', pricing_version TEXT NOT NULL DEFAULT 'unversioned',
                operation TEXT NOT NULL DEFAULT 'chat'
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
            if "operation" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN operation TEXT NOT NULL DEFAULT 'chat'")
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
                created_at TEXT NOT NULL, delivered_at TEXT, last_error TEXT,
                job_id TEXT, event_type TEXT, updated_at TEXT,
                signing_keys_json TEXT, active_key_id TEXT
            )""")
            webhook_columns = {row[1] for row in db.execute("PRAGMA table_info(webhook_outbox)")}
            for column in ("job_id", "event_type", "updated_at", "signing_keys_json", "active_key_id"):
                if column not in webhook_columns:
                    db.execute(f"ALTER TABLE webhook_outbox ADD COLUMN {column} TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS idx_webhook_outbox_due ON webhook_outbox(status, next_attempt_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_webhook_outbox_job ON webhook_outbox(job_id, created_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS webhook_delivery_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                attempted_at TEXT NOT NULL, attempt INTEGER NOT NULL,
                outcome TEXT NOT NULL, http_status INTEGER, error TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_webhook_attempts_event ON webhook_delivery_attempts(event_id, attempt)")
            attempt_columns = {row[1] for row in db.execute("PRAGMA table_info(webhook_delivery_attempts)")}
            if "duration_ms" not in attempt_columns:
                db.execute("ALTER TABLE webhook_delivery_attempts ADD COLUMN duration_ms INTEGER")
            db.execute("""CREATE TABLE IF NOT EXISTS webhook_endpoint_controls (
                webhook_url TEXT PRIMARY KEY, paused INTEGER NOT NULL DEFAULT 0,
                reset_at TEXT, updated_at TEXT, updated_by TEXT
            )""")
            db.execute("""INSERT OR IGNORE INTO webhook_endpoint_controls (webhook_url)
                SELECT DISTINCT webhook_url FROM webhook_outbox""")
            db.execute("""CREATE TABLE IF NOT EXISTS admin_audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL,
                actor TEXT NOT NULL, action TEXT NOT NULL, status_code INTEGER NOT NULL,
                previous_hash TEXT, entry_hash TEXT
            )""")
            audit_columns = {row[1] for row in db.execute("PRAGMA table_info(admin_audit_events)")}
            for column in ("previous_hash", "entry_hash"):
                if column not in audit_columns:
                    db.execute(f"ALTER TABLE admin_audit_events ADD COLUMN {column} TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS admin_audit_chain_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), anchor_id INTEGER NOT NULL,
                anchor_hash TEXT NOT NULL
            )""")
            db.execute("INSERT OR IGNORE INTO admin_audit_chain_state VALUES (1, 0, ?)", (AUDIT_GENESIS_HASH,))
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
            if "actor" not in audit_columns:
                db.execute("ALTER TABLE tool_call_audits ADD COLUMN actor TEXT")
            if "expires_at" not in audit_columns:
                db.execute("ALTER TABLE tool_call_audits ADD COLUMN expires_at TEXT")
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
            db.execute("""CREATE TABLE IF NOT EXISTS shadow_observations (
                request_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL, application TEXT NOT NULL, route TEXT NOT NULL,
                baseline_provider TEXT NOT NULL, baseline_model TEXT NOT NULL,
                candidate_provider TEXT NOT NULL, candidate_model TEXT NOT NULL,
                baseline_quality INTEGER NOT NULL, candidate_quality INTEGER NOT NULL,
                baseline_cost_usd REAL NOT NULL, candidate_cost_usd REAL NOT NULL,
                baseline_latency_ms INTEGER NOT NULL, candidate_latency_ms INTEGER NOT NULL,
                candidate_status TEXT NOT NULL, failure_type TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_shadow_report ON shadow_observations(occurred_at, application, route)")
            db.execute("""CREATE TABLE IF NOT EXISTS tenant_data_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL,
                tenant_id TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
                record_counts TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_tenant_data_audit ON tenant_data_audits(tenant_id, occurred_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS async_jobs (
                job_id TEXT PRIMARY KEY, job_type TEXT NOT NULL, tenant_id TEXT, actor TEXT,
                status TEXT NOT NULL, progress_completed INTEGER NOT NULL DEFAULT 0,
                progress_total INTEGER NOT NULL DEFAULT 0, usage_json TEXT NOT NULL DEFAULT '{}',
                manifest_json TEXT, payload_json TEXT, result_json TEXT, error TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, started_at TEXT,
                completed_at TEXT, lease_until TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_async_jobs_pending ON async_jobs(status, lease_until, created_at)")
            job_columns = {row[1] for row in db.execute("PRAGMA table_info(async_jobs)")}
            for column, declaration in (("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                                        ("max_attempts", "INTEGER NOT NULL DEFAULT 5"),
                                        ("next_attempt_at", "TEXT"),
                                        ("cancel_requested", "INTEGER NOT NULL DEFAULT 0")):
                if column not in job_columns:
                    db.execute(f"ALTER TABLE async_jobs ADD COLUMN {column} {declaration}")
            db.execute("CREATE INDEX IF NOT EXISTS idx_async_jobs_due ON async_jobs(status, next_attempt_at, created_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_async_jobs_tenant_status_created ON async_jobs(tenant_id, status, created_at)")
            for column in ("idempotency_scope", "idempotency_key_hash", "request_fingerprint"):
                if column not in job_columns:
                    db.execute(f"ALTER TABLE async_jobs ADD COLUMN {column} TEXT")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_async_jobs_idempotency
                ON async_jobs(idempotency_scope, job_type, idempotency_key_hash)
                WHERE idempotency_key_hash IS NOT NULL""")
            db.execute("""CREATE TABLE IF NOT EXISTS maintenance_leases (
                lease_name TEXT PRIMARY KEY, owner_id TEXT NOT NULL, lease_until TEXT NOT NULL
            )""")
            self._backfill_audit_chain(db)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    @staticmethod
    def _backfill_audit_chain(db: sqlite3.Connection) -> None:
        state = db.execute("SELECT anchor_id,anchor_hash FROM admin_audit_chain_state WHERE singleton=1").fetchone()
        previous = state["anchor_hash"]
        rows = db.execute("SELECT id,occurred_at,actor,action,status_code,previous_hash,entry_hash "
                          "FROM admin_audit_events WHERE id>? ORDER BY id", (state["anchor_id"],)).fetchall()
        for row in rows:
            expected = audit_entry_hash(row["id"], row["occurred_at"], row["actor"], row["action"],
                                        row["status_code"], previous)
            if row["entry_hash"] is None:
                db.execute("UPDATE admin_audit_events SET previous_hash=?,entry_hash=? WHERE id=?",
                           (previous, expected, row["id"]))
            elif row["previous_hash"] != previous or not hmac.compare_digest(row["entry_hash"], expected):
                # Preserve the mismatch so the verification endpoint reports legacy tampering.
                previous = row["entry_hash"]
                continue
            previous = expected

    @staticmethod
    def _append_audit(db: sqlite3.Connection, actor: str, action: str, status_code: int,
                      occurred_at: str) -> None:
        if not db.in_transaction:
            db.execute("BEGIN IMMEDIATE")
        state = db.execute("SELECT anchor_hash FROM admin_audit_chain_state WHERE singleton=1").fetchone()
        tail = db.execute("SELECT id,entry_hash FROM admin_audit_events ORDER BY id DESC LIMIT 1").fetchone()
        previous = tail["entry_hash"] if tail else state["anchor_hash"]
        cursor = db.execute("INSERT INTO admin_audit_events (occurred_at,actor,action,status_code,previous_hash) "
                            "VALUES (?,?,?,?,?)", (occurred_at, actor, action, status_code, previous))
        entry_hash = audit_entry_hash(cursor.lastrowid, occurred_at, actor, action, status_code, previous)
        db.execute("UPDATE admin_audit_events SET entry_hash=? WHERE id=?", (entry_hash, cursor.lastrowid))

    def acquire_maintenance_lease(self, lease_name: str, owner_id: str, lease_seconds: int,
                                  now: datetime | None = None) -> bool:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        now = now or datetime.now(UTC)
        until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("""INSERT INTO maintenance_leases (lease_name,owner_id,lease_until)
                VALUES (?,?,?) ON CONFLICT(lease_name) DO UPDATE SET
                owner_id=excluded.owner_id,lease_until=excluded.lease_until
                WHERE maintenance_leases.lease_until<=? OR maintenance_leases.owner_id=?""",
                (lease_name, owner_id, until, now.isoformat(), owner_id))
            return cursor.rowcount == 1

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
                   "latency_ms", "fallback_used", "cached", "error_type", "cost_center", "canary_used", "operation")
        defaults = {"canary_used": 0, "pricing_catalog": "default", "pricing_version": "unversioned", "operation": "chat"}
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
            db.execute("BEGIN IMMEDIATE")
            deleted = 0
            for table in ("requests", "routing_decisions", "shadow_observations"):
                cursor = db.execute(f"DELETE FROM {table} WHERE substr(application,1,length(?))=? AND occurred_at < ?",
                                    (prefix, prefix, cutoff))
                deleted += cursor.rowcount
            for table in ("tool_call_audits", "tool_execution_outcomes"):
                cursor = db.execute(f"DELETE FROM {table} WHERE tenant_id=? AND occurred_at < ?", (tenant_id, cutoff))
                deleted += cursor.rowcount
            cursor = db.execute("""DELETE FROM async_jobs WHERE tenant_id=? AND created_at < ?
                AND (status != 'running' OR lease_until <= ?)""",
                (tenant_id, cutoff, datetime.now(UTC).isoformat()))
            deleted += cursor.rowcount
            return deleted

    def record_tenant_data_audit(self, tenant_id: str, actor: str, action: str,
                                 record_counts: dict[str, int]) -> None:
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO tenant_data_audits (occurred_at, tenant_id, actor, action, record_counts) VALUES (?, ?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), tenant_id, actor, action,
                 json.dumps(record_counts, sort_keys=True)))

    def tenant_data_audits(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT occurred_at, tenant_id, actor, action, record_counts FROM tenant_data_audits WHERE tenant_id=? ORDER BY id DESC LIMIT ?",
                              (tenant_id, limit)).fetchall()
            return [{**dict(row), "record_counts": json.loads(row["record_counts"])} for row in rows]

    def export_tenant_data(self, tenant_id: str, actor: str | None = None) -> dict[str, list[dict[str, Any]]]:
        prefix = f"{tenant_id}:"
        with self._lock, self._connect() as db:
            result: dict[str, list[dict[str, Any]]] = {}
            for table in ("requests", "routing_decisions", "shadow_observations"):
                result[table] = [dict(row) for row in db.execute(
                    f"SELECT * FROM {table} WHERE substr(application,1,length(?))=? ORDER BY occurred_at",
                    (prefix, prefix)).fetchall()]
            for table in ("tool_call_audits", "tool_execution_outcomes"):
                result[table] = [dict(row) for row in db.execute(
                    f"SELECT * FROM {table} WHERE tenant_id=? ORDER BY occurred_at", (tenant_id,)).fetchall()]
            jobs = db.execute("""SELECT job_id, job_type, tenant_id, actor, status,
                progress_completed, progress_total, usage_json, manifest_json, error,
                created_at, updated_at, started_at, completed_at
                FROM async_jobs WHERE tenant_id=? ORDER BY created_at""", (tenant_id,)).fetchall()
            result["async_jobs"] = []
            for row in jobs:
                job = dict(row)
                for column in ("usage_json", "manifest_json"):
                    raw = job.pop(column)
                    job[column.removesuffix("_json")] = json.loads(raw) if raw is not None else None
                result["async_jobs"].append(job)
            if actor:
                counts = {name: len(rows) for name, rows in result.items()}
                db.execute("INSERT INTO tenant_data_audits (occurred_at, tenant_id, actor, action, record_counts) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(UTC).isoformat(), tenant_id, actor, "export", json.dumps(counts, sort_keys=True)))
            audits = db.execute("SELECT occurred_at, tenant_id, actor, action, record_counts FROM tenant_data_audits WHERE tenant_id=? ORDER BY id DESC LIMIT 100",
                                (tenant_id,)).fetchall()
            result["audit_history"] = [{**dict(row), "record_counts": json.loads(row["record_counts"])}
                                       for row in audits]
            return result

    def delete_tenant_data(self, tenant_id: str, actor: str | None = None,
                           preserve_job_id: str | None = None) -> dict[str, int]:
        prefix = f"{tenant_id}:"
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            counts: dict[str, int] = {}
            for table in ("requests", "routing_decisions", "shadow_observations"):
                cursor = db.execute(f"DELETE FROM {table} WHERE substr(application,1,length(?))=?", (prefix, prefix))
                counts[table] = cursor.rowcount
            for table in ("tool_call_audits", "tool_execution_outcomes"):
                cursor = db.execute(f"DELETE FROM {table} WHERE tenant_id=?", (tenant_id,))
                counts[table] = cursor.rowcount
            for table in ("budget_reservations", "budget_alerts"):
                cursor = db.execute(f"DELETE FROM {table} WHERE substr(application,1,length(?))=?", (prefix, prefix))
                counts[table] = cursor.rowcount
            cursor = db.execute("""DELETE FROM webhook_delivery_attempts WHERE event_id IN
                (SELECT event_id FROM webhook_outbox WHERE substr(application,1,length(?))=? OR application=?)""",
                (prefix, prefix, tenant_id))
            counts["webhook_delivery_attempts"] = cursor.rowcount
            cursor = db.execute("DELETE FROM webhook_outbox WHERE substr(application,1,length(?))=? OR application=?",
                (prefix, prefix, tenant_id))
            counts["webhook_outbox"] = cursor.rowcount
            cursor = db.execute("DELETE FROM async_jobs WHERE tenant_id=? AND job_id != COALESCE(?, '')",
                                (tenant_id, preserve_job_id))
            counts["async_jobs"] = cursor.rowcount
            if actor:
                db.execute("INSERT INTO tenant_data_audits (occurred_at, tenant_id, actor, action, record_counts) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(UTC).isoformat(), tenant_id, actor, "delete", json.dumps(counts, sort_keys=True)))
            return counts

    def tenant_job_artifact_keys(self, tenant_id: str, preserve_job_id: str | None = None) -> list[str]:
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT manifest_json, result_json FROM async_jobs
                WHERE tenant_id=? AND job_id != COALESCE(?, '')""", (tenant_id, preserve_job_id)).fetchall()
        keys = set()
        for row in rows:
            for column in ("manifest_json", "result_json"):
                value = json.loads(row[column]) if row[column] else {}
                if isinstance(value, dict) and value.get("artifact_key"):
                    keys.add(value["artifact_key"])
        return sorted(keys)

    def tenant_data_counts(self, tenant_id: str) -> dict[str, int]:
        prefix = f"{tenant_id}:"
        with self._lock, self._connect() as db:
            counts = {}
            for table in ("requests", "routing_decisions", "shadow_observations"):
                counts[table] = int(db.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE substr(application,1,length(?))=?",
                    (prefix, prefix)).fetchone()[0])
            for table in ("tool_call_audits", "tool_execution_outcomes"):
                counts[table] = int(db.execute(f"SELECT COUNT(*) FROM {table} WHERE tenant_id=?",
                                               (tenant_id,)).fetchone()[0])
            counts["async_jobs"] = int(db.execute("SELECT COUNT(*) FROM async_jobs WHERE tenant_id=?",
                                                  (tenant_id,)).fetchone()[0])
            return counts

    def create_async_job(self, job_type: str, tenant_id: str | None, actor: str | None,
                         payload: dict[str, Any], progress_total: int,
                         max_attempts: int = 5, *, idempotency_key: str | None = None,
                         idempotency_scope: str | None = None,
                         max_queued_for_tenant: int | None = None) -> dict[str, Any]:
        job_id, now = "job_" + uuid.uuid4().hex, datetime.now(UTC).isoformat()
        scope = idempotency_scope or tenant_id or actor or "anonymous"
        key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest() if idempotency_key else None
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest() if key_hash else None
        existing_job_id = None
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if key_hash:
                existing = db.execute("""SELECT job_id, request_fingerprint FROM async_jobs
                    WHERE idempotency_scope=? AND job_type=? AND idempotency_key_hash=?""",
                    (scope, job_type, key_hash)).fetchone()
                if existing:
                    if existing["request_fingerprint"] != fingerprint:
                        raise AsyncJobIdempotencyConflict("Idempotency-Key was reused with a different request")
                    existing_job_id = existing["job_id"]
            if existing_job_id is None and tenant_id and max_queued_for_tenant is not None:
                queued = db.execute("SELECT COUNT(*) FROM async_jobs WHERE tenant_id=? AND status='queued'",
                                    (tenant_id,)).fetchone()[0]
                if queued >= max_queued_for_tenant:
                    raise AsyncJobQueueLimitExceeded("Tenant async job queue limit reached")
            if existing_job_id is None:
                db.execute("""INSERT INTO async_jobs (job_id, job_type, tenant_id, actor, status,
                progress_total, max_attempts, payload_json, created_at, updated_at,
                idempotency_scope, idempotency_key_hash, request_fingerprint)
                VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (job_id, job_type, tenant_id, actor, progress_total, max_attempts,
                 json.dumps(payload, separators=(",", ":")), now, now,
                     scope if key_hash else None, key_hash, fingerprint))
        return self.async_job(existing_job_id or job_id) or {}

    def async_job(self, job_id: str, *, include_payload: bool = False,
                  include_results: bool = False) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM async_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            job = dict(row)
        for source, target in (("usage_json", "usage"), ("manifest_json", "manifest"),
                               ("payload_json", "payload"), ("result_json", "result")):
            raw = job.pop(source)
            if raw is not None and (target in {"usage", "manifest"} or
                                    (target == "payload" and include_payload) or
                                    (target == "result" and include_results)):
                job[target] = json.loads(raw)
        if not include_payload:
            job.pop("payload", None)
        if not include_results:
            job.pop("result", None)
        job.pop("lease_until", None)
        return job

    def cancel_async_job(self, job_id: str) -> dict[str, Any] | None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""UPDATE async_jobs SET status='cancelled', cancel_requested=1,
                payload_json=NULL, completed_at=?, updated_at=?, lease_until=NULL
                WHERE job_id=? AND status='queued'""", (now, now, job_id))
            db.execute("""UPDATE async_jobs SET cancel_requested=1, updated_at=?
                WHERE job_id=? AND status='running'""", (now, job_id))
        return self.async_job(job_id)

    def requeue_dead_letter_async_job(self, job_id: str, actor: str) -> bool:
        """Requeue a dead-letter job while retaining its payload and recording the operator action."""
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("""UPDATE async_jobs SET status='queued', attempt_count=0,
                next_attempt_at=NULL, error=NULL, completed_at=NULL, cancel_requested=0,
                updated_at=? WHERE job_id=? AND status='dead_letter' AND payload_json IS NOT NULL""",
                (now, job_id))
            if cursor.rowcount:
                self._append_audit(db, actor, f"async_job_requeued:{job_id}", 200, now)
            return bool(cursor.rowcount)

    def async_job_cancel_requested(self, job_id: str) -> bool:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT cancel_requested FROM async_jobs WHERE job_id=?", (job_id,)).fetchone()
            return bool(row and row["cancel_requested"])

    def fail_async_job(self, job_id: str, error: str, *, max_attempts: int,
                       base_delay_seconds: float, max_delay_seconds: float) -> str:
        now = datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT attempt_count, cancel_requested FROM async_jobs WHERE job_id=?",
                             (job_id,)).fetchone()
            if job is None:
                return "missing"
            attempts = int(job["attempt_count"])
            if job["cancel_requested"]:
                status, next_at = "cancelled", None
            elif attempts >= max_attempts:
                status, next_at = "dead_letter", None
            else:
                delay = min(max_delay_seconds, base_delay_seconds * (2 ** max(attempts - 1, 0)))
                status, next_at = "queued", (now + timedelta(seconds=delay)).isoformat()
            terminal = status in {"cancelled", "dead_letter"}
            db.execute("""UPDATE async_jobs SET status=?, error=?, next_attempt_at=?, updated_at=?,
                completed_at=?, lease_until=NULL, payload_json=CASE WHEN ? THEN NULL ELSE payload_json END
                WHERE job_id=?""", (status, error[:500], next_at, now.isoformat(),
                now.isoformat() if terminal else None, int(status == "cancelled"), job_id))
            return status

    def recover_stalled_async_jobs(self, *, max_attempts: int, base_delay_seconds: float,
                                   max_delay_seconds: float) -> int:
        now = datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            stalled = db.execute("""SELECT job_id, attempt_count, cancel_requested FROM async_jobs
                WHERE status='running' AND lease_until <= ?""", (now.isoformat(),)).fetchall()
            for job in stalled:
                attempts = int(job["attempt_count"])
                cancelled = bool(job["cancel_requested"])
                terminal = cancelled or attempts >= max_attempts
                delay = min(max_delay_seconds, base_delay_seconds * (2 ** max(attempts - 1, 0)))
                next_at = None if terminal else (now + timedelta(seconds=delay)).isoformat()
                status = "cancelled" if cancelled else "dead_letter" if terminal else "queued"
                db.execute("""UPDATE async_jobs SET status=?, error='stalled_job_lease_expired',
                    next_attempt_at=?, updated_at=?, completed_at=?, lease_until=NULL,
                    payload_json=CASE WHEN ? THEN NULL ELSE payload_json END WHERE job_id=?""",
                    (status, next_at, now.isoformat(), now.isoformat() if terminal else None,
                    int(status == "cancelled"), job["job_id"]))
            return len(stalled)

    def expired_async_job_leases(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            return [row[0] for row in db.execute("""SELECT job_id FROM async_jobs
                WHERE status='running' AND lease_until <= ?""", (now.isoformat(),)).fetchall()]

    def async_job_stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT status, COUNT(*) AS count FROM async_jobs GROUP BY status").fetchall()
            duration = db.execute("""SELECT AVG((julianday(completed_at)-julianday(started_at))*86400000)
                FROM async_jobs WHERE status='completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL""").fetchone()[0]
            latest = db.execute("""SELECT started_at, completed_at FROM async_jobs
                WHERE started_at IS NOT NULL ORDER BY started_at DESC LIMIT 100""").fetchall()
            oldest_running = db.execute("SELECT MIN(started_at) FROM async_jobs WHERE status='running'").fetchone()[0]
        counts = {row["status"]: int(row["count"]) for row in rows}
        durations = [max(0, int((datetime.fromisoformat(row["completed_at"])
            - datetime.fromisoformat(row["started_at"])).total_seconds() * 1000))
            for row in latest if row["completed_at"]]
        oldest_running_seconds = (max(0, int((datetime.now(UTC) - datetime.fromisoformat(oldest_running)).total_seconds()))
                                  if oldest_running else 0)
        return {"queue_depth": counts.get("queued", 0), "running": counts.get("running", 0),
            "dead_letter": counts.get("dead_letter", 0), "counts_by_status": counts,
            "average_duration_ms": round(float(duration or 0), 2), "recent_durations_ms": durations,
            "oldest_running_seconds": oldest_running_seconds}

    def pending_async_jobs(self, limit: int = 20,
                           tenant_limits: dict[str, int] | None = None) -> list[str]:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            blocked: list[str] = []
            if tenant_limits:
                running = {row["tenant_id"]: int(row["count"]) for row in db.execute(
                    "SELECT tenant_id, COUNT(*) AS count FROM async_jobs "
                    "WHERE status='running' AND tenant_id IS NOT NULL GROUP BY tenant_id").fetchall()}
                blocked = [tenant_id for tenant_id, cap in tenant_limits.items()
                           if running.get(tenant_id, 0) >= cap]
            exclusion = (f" AND (queued.tenant_id IS NULL OR queued.tenant_id NOT IN "
                         f"({','.join('?' for _ in blocked)}))") if blocked else ""
            rows = db.execute("""WITH due AS (
                SELECT queued.job_id, queued.tenant_id, queued.actor, queued.created_at,
                    CASE WHEN queued.tenant_id IS NOT NULL THEN 'tenant:' || queued.tenant_id
                        ELSE 'actor:' || COALESCE(queued.actor, '') END AS fairness_key,
                    (SELECT COUNT(*) FROM async_jobs running WHERE running.status='running'
                        AND running.tenant_id IS queued.tenant_id
                        AND (queued.tenant_id IS NOT NULL OR running.actor IS queued.actor)) AS running_count,
                    (SELECT MAX(previous.started_at) FROM async_jobs previous
                        WHERE previous.tenant_id IS queued.tenant_id
                        AND (queued.tenant_id IS NOT NULL OR previous.actor IS queued.actor)) AS last_started_at
                FROM async_jobs queued WHERE queued.status='queued'""" + exclusion + """
                    AND (queued.next_attempt_at IS NULL OR queued.next_attempt_at <= ?)
            ), ranked AS (
                SELECT job_id, created_at, fairness_key, running_count, last_started_at,
                    ROW_NUMBER() OVER (PARTITION BY fairness_key ORDER BY created_at, job_id) AS tenant_position
                FROM due
            ) SELECT job_id FROM ranked
                ORDER BY running_count, last_started_at, tenant_position, created_at, fairness_key LIMIT ?""",
                (*blocked, now, limit)).fetchall()
            return [row["job_id"] for row in rows]

    def claim_async_job(self, job_id: str, lease_seconds: int = 300,
                        max_concurrent_for_tenant: int | None = None) -> bool:
        now = datetime.now(UTC)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT tenant_id, status FROM async_jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None or job["status"] != "queued":
                return False
            if job["tenant_id"] and max_concurrent_for_tenant is not None:
                running = db.execute("SELECT COUNT(*) FROM async_jobs WHERE tenant_id=? AND status='running'",
                                     (job["tenant_id"],)).fetchone()[0]
                if running >= max_concurrent_for_tenant:
                    return False
            cursor = db.execute("""UPDATE async_jobs SET status='running', started_at=COALESCE(started_at, ?),
                attempt_count=attempt_count+1, next_attempt_at=NULL, updated_at=?, lease_until=?
                WHERE job_id=? AND status='queued' AND (next_attempt_at IS NULL OR next_attempt_at <= ?)""",
                (now.isoformat(), now.isoformat(), lease_until, job_id, now.isoformat()))
            return cursor.rowcount == 1

    def expired_tenant_async_job_artifacts(self, tenant_id: str, retention_days: int) -> list[str]:
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT manifest_json, result_json FROM async_jobs WHERE tenant_id=?
                AND status IN ('completed','failed','cancelled','dead_letter')
                AND COALESCE(completed_at, updated_at, created_at) < ?""", (tenant_id, cutoff)).fetchall()
        keys: set[str] = set()
        for row in rows:
            for column in ("manifest_json", "result_json"):
                value = json.loads(row[column]) if row[column] else {}
                if isinstance(value, dict) and value.get("artifact_key"):
                    keys.add(value["artifact_key"])
        return sorted(keys)

    def prune_tenant_async_jobs(self, tenant_id: str, retention_days: int) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._lock, self._connect() as db:
            cursor = db.execute("""DELETE FROM async_jobs WHERE tenant_id=?
                AND status IN ('completed','failed','cancelled','dead_letter')
                AND COALESCE(completed_at, updated_at, created_at) < ?""", (tenant_id, cutoff))
            return cursor.rowcount

    def renew_async_job_lease(self, job_id: str, lease_seconds: int = 300) -> bool:
        now = datetime.now(UTC)
        with self._lock, self._connect() as db:
            cursor = db.execute("""UPDATE async_jobs SET updated_at=?, lease_until=?
                WHERE job_id=? AND status='running'""",
                (now.isoformat(), (now + timedelta(seconds=lease_seconds)).isoformat(), job_id))
            return cursor.rowcount == 1

    def update_async_job(self, job_id: str, *, status: str | None = None,
                         progress_completed: int | None = None, usage: dict[str, Any] | None = None,
                         manifest: dict[str, Any] | None = None, result: Any = None,
                         error: str | None = None, clear_payload: bool = False,
                         lease_seconds: int = 300) -> None:
        now = datetime.now(UTC)
        assignments, values = ["updated_at=?"], [now.isoformat()]
        if status is not None:
            assignments.append("status=?")
            values.append(status)
        if progress_completed is not None:
            assignments.append("progress_completed=?")
            values.append(int(progress_completed))
        if usage is not None:
            assignments.append("usage_json=?")
            values.append(json.dumps(usage, separators=(",", ":")))
        if manifest is not None:
            assignments.append("manifest_json=?")
            values.append(json.dumps(manifest, separators=(",", ":")))
        if result is not None:
            assignments.append("result_json=?")
            values.append(json.dumps(result, separators=(",", ":")))
        if error is not None:
            assignments.append("error=?")
            values.append(error[:500])
        elif status == "completed":
            assignments.append("error=NULL")
        if clear_payload:
            assignments.append("payload_json=NULL")
        terminal = status in {"completed", "failed", "cancelled", "dead_letter"}
        assignments.extend(["completed_at=?", "lease_until=?"])
        values.extend([now.isoformat() if terminal else None,
                       None if terminal else (now + timedelta(seconds=lease_seconds)).isoformat()])
        values.append(job_id)
        with self._lock, self._connect() as db:
            db.execute(f"UPDATE async_jobs SET {', '.join(assignments)} WHERE job_id=?", values)

    def finish_tenant_data_job(self, job_id: str, tenant_id: str, actor: str, action: str,
                               record_counts: dict[str, int], result: dict[str, Any],
                               manifest: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""UPDATE async_jobs SET status='completed', progress_completed=progress_total,
                manifest_json=?, result_json=?, payload_json=NULL, updated_at=?, completed_at=?, lease_until=NULL,
                error=NULL
                WHERE job_id=?""", (json.dumps(manifest, separators=(",", ":")),
                json.dumps(result, separators=(",", ":"), default=str), now, now, job_id))
            db.execute("""INSERT INTO tenant_data_audits
                (occurred_at, tenant_id, actor, action, record_counts) VALUES (?, ?, ?, ?, ?)""",
                (now, tenant_id, actor, action, json.dumps(record_counts, sort_keys=True)))

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
                        now: datetime | None = None, *, signing_keys: dict[str, str] | None = None,
                        active_key_id: str | None = None) -> None:
        now = now or datetime.now(UTC)
        import json
        with self._lock, self._connect() as db:
            db.execute("INSERT OR IGNORE INTO webhook_endpoint_controls (webhook_url) VALUES (?)", (webhook_url,))
            db.execute("""INSERT OR IGNORE INTO webhook_outbox
                (event_id, application, webhook_url, secret_env, body, status, attempts,
                 next_attempt_at, created_at, job_id, event_type, updated_at,
                 signing_keys_json, active_key_id)
                VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?)""",
                (event.get("event_id") or event.get("alert_id"), event["application"], webhook_url, secret_env,
                 json.dumps(event, separators=(",", ":"), sort_keys=True), now.isoformat(), now.isoformat(),
                 event.get("job_id"), event.get("type") or event.get("event"), now.isoformat(),
                 json.dumps(signing_keys, separators=(",", ":"), sort_keys=True) if signing_keys else None,
                 active_key_id))

    def due_webhooks(self, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            due = db.execute("""SELECT w.event_id FROM webhook_outbox w
                JOIN webhook_endpoint_controls c ON c.webhook_url=w.webhook_url
                WHERE w.status IN ('pending','processing') AND w.next_attempt_at <= ? AND c.paused=0
                ORDER BY w.created_at LIMIT ?""",
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
                       last_error: str | None = None, http_status: int | None = None,
                       duration_ms: int | None = None) -> None:
        now = datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("""UPDATE webhook_outbox SET attempts=?, status=?, next_attempt_at=?,
                delivered_at=?, last_error=?, updated_at=? WHERE event_id=?""",
                (attempts, status, (next_attempt_at or datetime.now(UTC)).isoformat(),
                 delivered_at.isoformat() if delivered_at else None, last_error, now.isoformat(), event_id))
            db.execute("""INSERT INTO webhook_delivery_attempts
                (event_id, attempted_at, attempt, outcome, http_status, error, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (event_id, now.isoformat(), attempts, status, http_status, last_error, duration_ms))

    def defer_webhook(self, event_id: str, next_attempt_at: datetime) -> None:
        with self._lock, self._connect() as db:
            db.execute("UPDATE webhook_outbox SET status='pending', next_attempt_at=?, updated_at=? WHERE event_id=?",
                (next_attempt_at.isoformat(), datetime.now(UTC).isoformat(), event_id))

    def redrive_webhook(self, event_id: str, now: datetime | None = None,
                        actor: str | None = None) -> bool:
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            cursor = db.execute("""UPDATE webhook_outbox SET status='pending', attempts=0,
                next_attempt_at=?, delivered_at=NULL, last_error=NULL
                WHERE event_id=? AND status='dead_letter'""", (now.isoformat(), event_id))
            redriven = bool(cursor.rowcount)
            if redriven and actor:
                self._append_audit(db, actor, f"webhook_redrive:{event_id}", 202, now.isoformat())
            return redriven

    def prune_webhook_history(self, retention_days: int, now: datetime | None = None) -> dict[str, int]:
        if not isinstance(retention_days, int) or isinstance(retention_days, bool) or retention_days < 1:
            raise ValueError("retention_days must be a positive integer")
        cutoff = ((now or datetime.now(UTC)) - timedelta(days=retention_days)).isoformat()
        terminal = "SELECT event_id FROM webhook_outbox WHERE status IN ('delivered','dead_letter') AND COALESCE(delivered_at,updated_at,created_at) < ?"
        with self._lock, self._connect() as db:
            attempts = db.execute(f"DELETE FROM webhook_delivery_attempts WHERE event_id IN ({terminal})", (cutoff,)).rowcount
            events = db.execute(f"DELETE FROM webhook_outbox WHERE event_id IN ({terminal})", (cutoff,)).rowcount
            return {"events": int(events), "attempts": int(attempts)}

    def webhook_endpoint_report(self, window_hours: int = 24,
                                now: datetime | None = None) -> list[dict[str, Any]]:
        if window_hours < 1:
            raise ValueError("window_hours must be positive")
        now = now or datetime.now(UTC)
        since = (now - timedelta(hours=window_hours)).isoformat()
        with self._lock, self._connect() as db:
            rows = db.execute("""SELECT c.webhook_url,c.paused,c.reset_at,c.updated_at,c.updated_by,
                SUM(CASE WHEN w.status='pending' THEN 1 ELSE 0 END) AS pending_count,
                SUM(CASE WHEN w.status='processing' THEN 1 ELSE 0 END) AS active_count,
                MIN(CASE WHEN w.status='pending' THEN w.created_at END) AS oldest_pending_at
                FROM webhook_endpoint_controls c LEFT JOIN webhook_outbox w ON w.webhook_url=c.webhook_url
                GROUP BY c.webhook_url,c.paused,c.reset_at,c.updated_at,c.updated_by ORDER BY c.webhook_url""").fetchall()
            attempts = db.execute("""SELECT w.webhook_url,COUNT(a.id) AS attempt_count,
                SUM(CASE WHEN a.outcome='delivered' THEN 1 ELSE 0 END) AS success_count,
                AVG(a.duration_ms) AS average_latency_ms
                FROM webhook_outbox w JOIN webhook_delivery_attempts a ON a.event_id=w.event_id
                WHERE a.attempted_at>=? GROUP BY w.webhook_url""", (since,)).fetchall()
        attempt_stats = {row["webhook_url"]: dict(row) for row in attempts}
        report = []
        for row in rows:
            stats = attempt_stats.get(row["webhook_url"], {})
            attempt_count = int(stats.get("attempt_count") or 0)
            success_count = int(stats.get("success_count") or 0)
            oldest = datetime.fromisoformat(row["oldest_pending_at"]) if row["oldest_pending_at"] else None
            report.append({"endpoint_id": webhook_endpoint_id(row["webhook_url"]),
                "webhook_url": row["webhook_url"], "paused": bool(row["paused"]),
                "pending_count": int(row["pending_count"] or 0),
                "active_deliveries": int(row["active_count"] or 0),
                "oldest_pending_age_seconds": (max(0, int((now-oldest).total_seconds())) if oldest else 0),
                "attempt_count": attempt_count, "success_count": success_count,
                "success_rate": success_count / attempt_count if attempt_count else None,
                "average_latency_ms": round(float(stats["average_latency_ms"]), 2)
                    if stats.get("average_latency_ms") is not None else None,
                "reset_at": row["reset_at"], "updated_at": row["updated_at"],
                "updated_by": row["updated_by"]})
        return report

    def set_webhook_endpoint_paused(self, endpoint_id: str, paused: bool, actor: str,
                                    now: datetime | None = None) -> dict[str, Any] | None:
        from .webhook_security import webhook_endpoint_id

        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            matches = [row[0] for row in db.execute("SELECT webhook_url FROM webhook_endpoint_controls").fetchall()
                       if webhook_endpoint_id(row[0]) == endpoint_id]
            if len(matches) != 1:
                return None
            url = matches[0]
            db.execute("""UPDATE webhook_endpoint_controls SET paused=?,updated_at=?,updated_by=?
                WHERE webhook_url=?""", (int(paused), now.isoformat(), actor, url))
            active = int(db.execute("SELECT COUNT(*) FROM webhook_outbox WHERE webhook_url=? AND status='processing'",
                                    (url,)).fetchone()[0])
            action = "pause" if paused else "resume"
            self._append_audit(db, actor, f"webhook_endpoint_{action}:{endpoint_id}", 200, now.isoformat())
            return {"endpoint_id": endpoint_id, "paused": paused, "active_deliveries": active,
                    "updated_at": now.isoformat(), "updated_by": actor}

    def reset_webhook_endpoint_circuit(self, endpoint_id: str, actor: str,
                                       now: datetime | None = None) -> dict[str, Any] | None:
        from .webhook_security import webhook_endpoint_id

        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            matches = [row[0] for row in db.execute("SELECT webhook_url FROM webhook_endpoint_controls").fetchall()
                       if webhook_endpoint_id(row[0]) == endpoint_id]
            if len(matches) != 1:
                return None
            url = matches[0]
            active = int(db.execute("SELECT COUNT(*) FROM webhook_outbox WHERE webhook_url=? AND status='processing'",
                                    (url,)).fetchone()[0])
            if active:
                raise WebhookEndpointBusy(endpoint_id)
            db.execute("UPDATE webhook_endpoint_controls SET reset_at=?,updated_at=?,updated_by=? WHERE webhook_url=?",
                       (now.isoformat(), now.isoformat(), actor, url))
            self._append_audit(db, actor, f"webhook_circuit_reset:{endpoint_id}", 200, now.isoformat())
            return {"endpoint_id": endpoint_id, "reset_at": now.isoformat(), "updated_by": actor}

    def webhook_endpoint_control(self, webhook_url: str) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT paused,reset_at FROM webhook_endpoint_controls WHERE webhook_url=?",
                             (webhook_url,)).fetchone()
            return {"paused": bool(row["paused"]), "reset_at": row["reset_at"]} if row else {
                "paused": False, "reset_at": None}

    def webhook_events(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            if status:
                rows = db.execute("SELECT event_id, application, status, attempts, created_at, delivered_at, last_error, job_id, event_type, updated_at FROM webhook_outbox WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
            else:
                rows = db.execute("SELECT event_id, application, status, attempts, created_at, delivered_at, last_error, job_id, event_type, updated_at FROM webhook_outbox ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]

    def async_job_webhook_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            events = [dict(row) for row in db.execute("""SELECT event_id, application, status, attempts,
                created_at, updated_at, delivered_at, last_error, event_type FROM webhook_outbox
                WHERE job_id=? ORDER BY created_at, event_id""", (job_id,)).fetchall()]
            for event in events:
                event["delivery_history"] = [dict(row) for row in db.execute("""SELECT attempted_at,
                    attempt, outcome, http_status, error FROM webhook_delivery_attempts
                    WHERE event_id=? ORDER BY attempt, id""", (event["event_id"],)).fetchall()]
            return events

    def record_audit(self, actor: str, action: str, status_code: int,
                     now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as db:
            self._append_audit(db, actor, action, status_code, now.isoformat())

    @staticmethod
    def _audit_chain_status(db: sqlite3.Connection) -> dict[str, Any]:
        state = db.execute("SELECT anchor_id,anchor_hash FROM admin_audit_chain_state WHERE singleton=1").fetchone()
        rows = db.execute("SELECT id,occurred_at,actor,action,status_code,previous_hash,entry_hash "
                          "FROM admin_audit_events WHERE id>? ORDER BY id", (state["anchor_id"],)).fetchall()
        previous = state["anchor_hash"]
        invalid_id = None
        for row in rows:
            expected = audit_entry_hash(row["id"], row["occurred_at"], row["actor"], row["action"],
                                        row["status_code"], previous)
            if row["previous_hash"] != previous or not row["entry_hash"] or not hmac.compare_digest(row["entry_hash"], expected):
                invalid_id = int(row["id"])
                break
            previous = row["entry_hash"]
        return {"verified": invalid_id is None, "record_count": len(rows),
                "anchor_id": int(state["anchor_id"]), "anchor_hash": state["anchor_hash"],
                "head_id": int(rows[-1]["id"]) if rows else int(state["anchor_id"]),
                "head_hash": previous, "first_invalid_id": invalid_id}

    def audit_chain_status(self) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            return self._audit_chain_status(db)

    def export_audit_events(self) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            status = self._audit_chain_status(db)
            events = [dict(row) for row in db.execute("SELECT id,occurred_at,actor,action,status_code,previous_hash,entry_hash "
                "FROM admin_audit_events ORDER BY id DESC")]
        canonical = json.dumps(events, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return {"format": "llm-gateway-admin-audit-v1", "exported_at": datetime.now(UTC).isoformat(),
                "chain": status, "export_sha256": hashlib.sha256(canonical).hexdigest(), "events": events}

    def prune_admin_audit_events(self, retention_days: int, now: datetime | None = None) -> dict[str, Any]:
        if not isinstance(retention_days, int) or isinstance(retention_days, bool) or retention_days < 1:
            raise ValueError("retention_days must be a positive integer")
        now = now or datetime.now(UTC)
        cutoff = (now - timedelta(days=retention_days)).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = self._audit_chain_status(db)
            if not status["verified"]:
                raise ValueError(f"admin audit chain integrity check failed at event {status['first_invalid_id']}")
            rows = db.execute("SELECT id,occurred_at,entry_hash FROM admin_audit_events ORDER BY id").fetchall()
            expired = []
            for row in rows:
                if row["occurred_at"] >= cutoff:
                    break
                expired.append(row)
            if not expired:
                return {"deleted": 0, "anchor_id": status["anchor_id"], "anchor_hash": status["anchor_hash"]}
            anchor = expired[-1]
            db.execute("DELETE FROM admin_audit_events WHERE id<=?", (anchor["id"],))
            db.execute("UPDATE admin_audit_chain_state SET anchor_id=?,anchor_hash=? WHERE singleton=1",
                       (anchor["id"], anchor["entry_hash"]))
            return {"deleted": len(expired), "anchor_id": int(anchor["id"]), "anchor_hash": anchor["entry_hash"]}

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, occurred_at, actor, action, status_code, previous_hash, entry_hash FROM admin_audit_events ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()]

    def record_tool_call_audits(self, request_id: str, tenant_id: str | None, application: str,
                                events: list[dict[str, Any]]) -> None:
        if not events:
            return
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.executemany("""INSERT INTO tool_call_audits
                (request_id, tenant_id, application, occurred_at, tool_call_id, tool_name, outcome, reason, actor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(request_id, tenant_id, application, now, event.get("tool_call_id"), event["tool_name"],
                  event["outcome"], event.get("reason"), event.get("actor")) for event in events])

    def tool_call_audit_events(self, tenant_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            if tenant_id is None:
                rows = db.execute("""SELECT request_id, tenant_id, application, occurred_at,
                    tool_call_id, tool_name, outcome, reason, actor, expires_at FROM tool_call_audits ORDER BY id DESC LIMIT ?""", (limit,))
            else:
                rows = db.execute("""SELECT request_id, tenant_id, application, occurred_at,
                    tool_call_id, tool_name, outcome, reason, actor, expires_at FROM tool_call_audits WHERE tenant_id = ?
                    ORDER BY id DESC LIMIT ?""", (tenant_id, limit))
            return [dict(row) for row in rows.fetchall()]

    def record_tool_approval(self, request_id: str, tool_call_id: str, decision: str,
                             actor: str, expires_at: str | None = None) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            required = db.execute("""SELECT tenant_id, application, tool_name FROM tool_call_audits
                WHERE request_id=? AND tool_call_id=? AND outcome='approval_required'
                ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
            if required is None:
                raise ValueError("No tool call requiring external approval matches this request")
            prior = db.execute("""SELECT outcome, actor, expires_at FROM tool_call_audits
                WHERE request_id=? AND tool_call_id=? AND outcome IN ('approved','denied','revoked')
                ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
            if prior:
                if decision == "revoked":
                    if prior["outcome"] == "revoked":
                        return {"request_id": request_id, "tool_call_id": tool_call_id,
                            "tool_name": required["tool_name"], "application": required["application"],
                            "decision": decision, "actor": prior["actor"], "duplicate": True}
                    if prior["outcome"] != "approved":
                        raise ValueError("Only an approved tool call can be revoked")
                    if prior["expires_at"] and prior["expires_at"] <= now:
                        raise ValueError("An expired approval cannot be revoked")
                elif prior["outcome"] == decision:
                    if decision != "approved" or not prior["expires_at"] or prior["expires_at"] > now:
                        return {"request_id": request_id, "tool_call_id": tool_call_id,
                            "tool_name": required["tool_name"], "application": required["application"],
                            "decision": decision, "actor": prior["actor"], "expires_at": prior["expires_at"],
                            "duplicate": True}
                else:
                    raise ValueError("A different approval decision is already recorded")
            db.execute("""INSERT INTO tool_call_audits
                (request_id, tenant_id, application, occurred_at, tool_call_id, tool_name,
                 outcome, reason, actor, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (request_id, required["tenant_id"], required["application"], now, tool_call_id,
                 required["tool_name"], decision, "external_approval", actor,
                 expires_at if decision == "approved" else None))
        return {"request_id": request_id, "tool_call_id": tool_call_id,
            "tool_name": required["tool_name"], "application": required["application"],
            "decision": decision, "actor": actor,
            "expires_at": expires_at if decision == "approved" else None, "duplicate": False}

    def tool_approval_status(self, request_id: str, tool_call_id: str, tenant_id: str | None,
                             application: str) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            required = db.execute("""SELECT tenant_id, application, tool_name FROM tool_call_audits
                WHERE request_id=? AND tool_call_id=? AND outcome='approval_required'
                ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
            allowed = db.execute("""SELECT tenant_id, application, tool_name FROM tool_call_audits
                WHERE request_id=? AND tool_call_id=? AND outcome='allowed'
                ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
            source = required or allowed
            if source is None or source["tenant_id"] != tenant_id or source["application"] != application:
                raise ValueError("No authorized generated tool call matches this request")
            if required:
                decision = db.execute("""SELECT outcome, actor, occurred_at, expires_at FROM tool_call_audits
                    WHERE request_id=? AND tool_call_id=? AND outcome IN ('approved','denied','revoked')
                    ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
            else:
                decision = None
        status = decision["outcome"] if decision else ("pending" if required else "not_required")
        if status == "approved" and decision["expires_at"] and decision["expires_at"] <= datetime.now(UTC).isoformat():
            status = "expired"
        return {"request_id": request_id, "tool_call_id": tool_call_id, "tool_name": source["tool_name"],
            "status": status, "actor": decision["actor"] if decision else None,
            "decided_at": decision["occurred_at"] if decision else None,
            "expires_at": decision["expires_at"] if decision else None}

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
                WHERE (a.outcome='allowed' OR (a.outcome='approval_required' AND EXISTS (
                    SELECT 1 FROM tool_call_audits approval WHERE approval.request_id=a.request_id
                    AND approval.tool_call_id=a.tool_call_id AND approval.outcome='approved'
                    AND (approval.expires_at IS NULL OR approval.expires_at > ?)
                    AND approval.id=(SELECT MAX(latest.id) FROM tool_call_audits latest
                        WHERE latest.request_id=a.request_id AND latest.tool_call_id=a.tool_call_id
                        AND latest.outcome IN ('approved','denied','revoked')))))
                    AND a.tool_call_id IS NOT NULL AND o.request_id IS NULL{tenant_clause}""",
                (cutoff, current.isoformat(), *values)).fetchone()
            rows = db.execute(f"""SELECT a.request_id, a.tool_call_id, a.tenant_id, a.application,
                a.occurred_at, a.tool_name FROM tool_call_audits a LEFT JOIN tool_execution_outcomes o
                ON o.request_id=a.request_id AND o.tool_call_id=a.tool_call_id
                WHERE (a.outcome='allowed' OR (a.outcome='approval_required' AND EXISTS (
                    SELECT 1 FROM tool_call_audits approval WHERE approval.request_id=a.request_id
                    AND approval.tool_call_id=a.tool_call_id AND approval.outcome='approved'
                    AND (approval.expires_at IS NULL OR approval.expires_at > ?)
                    AND approval.id=(SELECT MAX(latest.id) FROM tool_call_audits latest
                        WHERE latest.request_id=a.request_id AND latest.tool_call_id=a.tool_call_id
                        AND latest.outcome IN ('approved','denied','revoked')))))
                    AND a.tool_call_id IS NOT NULL AND o.request_id IS NULL{tenant_clause}
                ORDER BY a.occurred_at ASC LIMIT ?""", (current.isoformat(), *values, limit)).fetchall()
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
            audit = db.execute("""SELECT tenant_id, application, tool_name, outcome FROM tool_call_audits
                WHERE request_id=? AND tool_call_id=? AND outcome IN ('allowed','approval_required')
                ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
            if audit is None or audit["tenant_id"] != tenant_id or audit["application"] != application:
                raise ValueError("No authorized generated tool call matches this request")
            if audit["outcome"] == "approval_required":
                approval = db.execute("""SELECT outcome, expires_at FROM tool_call_audits
                    WHERE request_id=? AND tool_call_id=? AND outcome IN ('approved','denied','revoked')
                    ORDER BY id DESC LIMIT 1""", (request_id, tool_call_id)).fetchone()
                if (approval is None or approval["outcome"] != "approved"
                        or (approval["expires_at"] and approval["expires_at"] <= now)):
                    raise ValueError("Tool execution requires recorded external approval")
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

    def evaluation_comparison(self, comparison_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("""SELECT comparison_id, baseline_run_id, candidate_run_id, report, created_at
                FROM evaluation_comparisons WHERE comparison_id=?""", (comparison_id,)).fetchone()
            return {**dict(row), "report": json.loads(row["report"])} if row else None

    def record_shadow_observation(self, observation: dict[str, Any]) -> None:
        with self._lock, self._connect() as db:
            db.execute("""INSERT OR IGNORE INTO shadow_observations
                (request_id, occurred_at, application, route, baseline_provider, baseline_model,
                 candidate_provider, candidate_model, baseline_quality, candidate_quality,
                 baseline_cost_usd, candidate_cost_usd, baseline_latency_ms, candidate_latency_ms,
                 candidate_status, failure_type)
                VALUES (:request_id, :occurred_at, :application, :route, :baseline_provider, :baseline_model,
                 :candidate_provider, :candidate_model, :baseline_quality, :candidate_quality,
                 :baseline_cost_usd, :candidate_cost_usd, :baseline_latency_ms, :candidate_latency_ms,
                 :candidate_status, :failure_type)""", observation)

    def shadow_report(self, application: str, route: str, start: str, end: str) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            row = db.execute("""SELECT COUNT(*) AS samples,
                SUM(baseline_quality) AS baseline_quality_passes,
                SUM(candidate_quality) AS candidate_quality_passes,
                AVG(baseline_cost_usd) AS baseline_cost_usd, AVG(candidate_cost_usd) AS candidate_cost_usd,
                AVG(baseline_latency_ms) AS baseline_latency_ms, AVG(candidate_latency_ms) AS candidate_latency_ms,
                SUM(CASE WHEN candidate_status!='success' THEN 1 ELSE 0 END) AS failures
                FROM shadow_observations WHERE application=? AND route=? AND occurred_at>=? AND occurred_at<?""",
                (application, route, start, end)).fetchone()
        samples = int(row["samples"] or 0)
        baseline_cost, candidate_cost = float(row["baseline_cost_usd"] or 0), float(row["candidate_cost_usd"] or 0)
        baseline_latency = float(row["baseline_latency_ms"] or 0)
        candidate_latency = float(row["candidate_latency_ms"] or 0)
        return {"application": application, "route": route, "samples": samples,
            "baseline": {"quality_pass_rate": (row["baseline_quality_passes"] or 0) / samples if samples else 0,
                "average_cost_usd": baseline_cost, "average_latency_ms": baseline_latency},
            "candidate": {"quality_pass_rate": (row["candidate_quality_passes"] or 0) / samples if samples else 0,
                "average_cost_usd": candidate_cost, "average_latency_ms": candidate_latency,
                "failure_rate": float(row["failures"] or 0) / samples if samples else 0},
            "cost_regression_percent": ((candidate_cost - baseline_cost) / baseline_cost * 100
                if baseline_cost else (0.0 if candidate_cost == 0 else 100.0)),
            "latency_regression_percent": ((candidate_latency - baseline_latency) / baseline_latency * 100
                if baseline_latency else (0.0 if candidate_latency == 0 else 100.0))}

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
            by_operation = [dict(row) for row in db.execute(f"SELECT operation, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(AVG(cost_usd),0) AS avg_cost_usd, SUM(status != 'success') AS failures FROM requests{where} GROUP BY operation ORDER BY cost_usd DESC", values).fetchall()]
            by_cost_center = [dict(row) for row in db.execute(f"SELECT COALESCE(cost_center,'unassigned') AS cost_center, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens, SUM(cost_usd) AS cost_usd, COALESCE(SUM(actual_cost_usd),0) AS actual_cost_usd FROM requests{where} GROUP BY cost_center ORDER BY cost_usd DESC", values).fetchall()]
        summary["failure_rate"] = summary["failures"] / summary["requests"] if summary["requests"] else 0
        summary["fallback_rate"] = summary["fallbacks"] / summary["requests"] if summary["requests"] else 0
        summary["generated_at"] = datetime.now(UTC).isoformat()
        summary["by_application"] = by_app
        summary["by_provider"] = by_provider
        summary["by_model"] = by_model
        summary["by_pricing_catalog"] = by_pricing_catalog
        summary["by_route"] = by_route
        summary["by_operation"] = by_operation
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
