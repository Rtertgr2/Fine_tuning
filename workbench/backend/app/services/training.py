"""Training run service layer: CRUD + run lifecycle (T3.4, T3.6)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app import config

RUN_ID_RE = re.compile(r"^r(\d+)$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_run_id(conn: sqlite3.Connection) -> str:
    """Generate next sequential run ID (r001, r002, ...)."""
    top = 0
    for row in conn.execute("SELECT run_id FROM training_runs").fetchall():
        m = RUN_ID_RE.match(row["run_id"])
        if m:
            top = max(top, int(m.group(1)))
    return f"r{top + 1:03d}"


def _runs_dir() -> Path:
    """Base directory for all training run outputs."""
    runs_dir = config.DATA_DIR / "training" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "config": json.loads(row["config_json"]) if row["config_json"] else {},
        "status": row["status"],
        "dataset_version": row["dataset_version"],
        "adapter_path": row["adapter_path"],
        "metrics_path": row["metrics_path"],
        "report_path": row["report_path"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "preflight": json.loads(row["preflight_json"]) if row["preflight_json"] else None,
    }


def create_run(
    conn: sqlite3.Connection,
    config_json: dict[str, Any],
    dataset_version: str = "",
) -> dict[str, Any]:
    """Create a new training run record and set up its output directory."""
    run_id = _next_run_id(conn)
    now = _now_iso()

    # Set up output directory
    run_dir = _runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=True)

    conn.execute(
        """INSERT INTO training_runs
           (run_id, config_json, status, dataset_version, metrics_path, created_at)
           VALUES (?,?,?,?,?,?)""",
        (
            run_id,
            json.dumps(config_json, ensure_ascii=False),
            "pending",
            dataset_version,
            str(run_dir / "metrics.jsonl"),
            now,
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return _row_to_dict(row)


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_runs(
    conn: sqlite3.Connection,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    if status_filter:
        rows = conn.execute(
            "SELECT * FROM training_runs WHERE status = ? ORDER BY created_at DESC",
            (status_filter,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_run(
    conn: sqlite3.Connection,
    run_id: str,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Update training run fields."""
    row = conn.execute(
        "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None

    fields: list[str] = []
    values: list[Any] = []

    if "status" in kwargs:
        fields.append("status = ?")
        values.append(kwargs["status"])
    if "adapter_path" in kwargs:
        fields.append("adapter_path = ?")
        values.append(kwargs["adapter_path"])
    if "metrics_path" in kwargs:
        fields.append("metrics_path = ?")
        values.append(kwargs["metrics_path"])
    if "report_path" in kwargs:
        fields.append("report_path = ?")
        values.append(kwargs["report_path"])
    if "completed_at" in kwargs:
        fields.append("completed_at = ?")
        values.append(kwargs["completed_at"])
    if "preflight" in kwargs:
        fields.append("preflight_json = ?")
        values.append(kwargs["preflight"])

    if fields:
        sql = f"UPDATE training_runs SET {', '.join(fields)} WHERE run_id = ?"
        values.append(run_id)
        conn.execute(sql, values)
        conn.commit()

    return get_run(conn, run_id)


def stop_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Mark a training run as stopped."""
    return update_run(conn, run_id, status="stopped")


def resume_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Resume a stopped training run (status set to pending)."""
    record = get_run(conn, run_id)
    if record is None:
        return None
    if record["status"] not in ("stopped", "failed"):
        raise ValueError(f"cannot resume run with status {record['status']!r}")
    return update_run(conn, run_id, status="pending")


def delete_run(conn: sqlite3.Connection, run_id: str) -> bool:
    """Delete a training run record."""
    cur = conn.execute(
        "DELETE FROM training_runs WHERE run_id = ?", (run_id,)
    )
    conn.commit()
    return cur.rowcount > 0
