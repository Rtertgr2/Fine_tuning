"""Training run lifecycle: state machine, process management, checkpoint, recovery.

Consolidates the run state machine (training.py), Popen process tracking,
checkpoint persistence, and server-restart recovery into a single module.

State machine (valid transitions):
    pending → running
    pending → stopped
    running → completed
    running → failed
    running → stopped
    stopped → pending   (resume, triggers preflight first)
    failed  → pending   (resume, triggers preflight first)

On module import, any DB records still in 'running' state are recovered:
if the process is no longer alive the run is reset to 'pending' so it
can be resumed cleanly after a server restart.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.app import config
from backend.app.db import connect
from backend.app.services import training as _training
from backend.app.services.preflight import run_preflight

RUN_ID_RE = _training.RUN_ID_RE
_active_processes: dict[str, Optional[subprocess.Popen]] = {}
_bg_tasks: set[asyncio.Task] = set()

VALID_TRANSITIONS: dict[str, list[str]] = {
    "pending": ["running", "stopped"],
    "running": ["completed", "failed", "stopped"],
    "stopped": ["pending"],
    "failed": ["pending"],
    "completed": [],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runs_dir() -> Path:
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


# ---------------------------------------------------------------------------
# State machine helpers
# ---------------------------------------------------------------------------

def _validate_transition(current_status: str, new_status: str) -> bool:
    """Return True if the transition current_status → new_status is valid."""
    allowed = VALID_TRANSITIONS.get(current_status, [])
    return new_status in allowed


# ---------------------------------------------------------------------------
# Recovery: re-track running processes on server restart
# ---------------------------------------------------------------------------

def _recover_runs() -> None:
    """Recover any in-flight runs after server restart.

    Scans the DB for runs in 'running' state. If the child process
    is no longer alive (no pid file or dead PID), resets the run to
    'pending' so it can be resumed cleanly.
    """
    try:
        conn = connect()
        try:
            rows = conn.execute(
                "SELECT run_id FROM training_runs WHERE status = 'running'"
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            run_id = row["run_id"]
            run_dir = _runs_dir() / run_id
            pid_file = run_dir / "training.pid"
            if not pid_file.exists():
                _reset_to_pending(conn, run_id)
                continue
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)  # raises ProcessLookupError if dead
                # Process alive — keep it running, just note it
                _active_processes[run_id] = None
            except (ValueError, OSError, ProcessLookupError):
                _reset_to_pending(conn, run_id)
    except Exception:
        pass


def _reset_to_pending(conn: sqlite3.Connection, run_id: str) -> None:
    """Reset a running run to pending after recovery."""
    try:
        conn.execute(
            "UPDATE training_runs SET status = 'pending' WHERE run_id = ? AND status = 'running'",
            (run_id,),
        )
        conn.commit()
    except Exception:
        pass


# Recover on module import
_recover_runs()


# ---------------------------------------------------------------------------
# Public API (mirrors training.py functions for router compatibility)
# ---------------------------------------------------------------------------

def create_run_sync(
    conn: sqlite3.Connection,
    config_json: dict[str, Any],
    dataset_version: str = "",
) -> dict[str, Any]:
    """Create a new training run record (status=pending)."""
    return _training.create_run_sync(conn, config_json, dataset_version)


def create_run(
    conn: sqlite3.Connection,
    config_json: dict[str, Any],
    dataset_version: str = "",
) -> dict[str, Any]:
    """Create a new training run record (status=pending)."""
    return _training.create_run(conn, config_json, dataset_version)


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Get training run by ID."""
    return _training.get_run(conn, run_id)


def list_runs(
    conn: sqlite3.Connection,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    """List training runs, optionally filtered by status."""
    return _training.list_runs(conn, status_filter)


def update_run(
    conn: sqlite3.Connection,
    run_id: str,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Update training run fields."""
    return _training.update_run(conn, run_id, **kwargs)


def spawn_training_process(run_id: str, run_dir: Path) -> None:
    """Start ``_run_training_process`` in the background."""
    _training.spawn_training_process(run_id, run_dir)


def stop_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Stop a pending/running training run with state machine validation."""
    record = get_run(conn, run_id)
    if record is None:
        return None
    current_status = record["status"]
    if current_status not in ("pending", "running"):
        raise ValueError(
            f"cannot stop run with status {current_status!r}"
        )
    return _training.stop_run(conn, run_id)


def resume_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Resume a stopped/failed run: trigger preflight first, then start training.

    On resume:
    1. Validate the state transition (stopped/failed → pending)
    2. Run current preflight checks against the run's config
    3. If preflight passes, set status to pending and spawn training
    4. If preflight fails, raise ValueError
    """
    record = get_run(conn, run_id)
    if record is None:
        return None

    current_status = record["status"]
    if current_status not in ("stopped", "failed"):
        raise ValueError(
            f"cannot resume run with status {current_status!r}"
        )

    # Get the config from the run record and run preflight
    cfg_dict = record.get("config", {})
    if not cfg_dict:
        update_run(conn, run_id, status="pending")
        spawn_training_process(run_id, _runs_dir() / run_id)
        return get_run(conn, run_id)

    from backend.app.services.training_config import TrainingConfig
    cfg = TrainingConfig.from_dict(cfg_dict)
    errors = cfg.validate()
    if errors:
        raise ValueError(
            f"Config validation failed on resume: {errors}"
        )

    # Run current preflight checks
    report = run_preflight(cfg)
    if not report.can_proceed:
        raise ValueError(
            f"Pre-flight checks failed on resume: "
            f"{[r.message for r in report.results if not r.passed and r.critical]}"
        )

    # Preflight passed: transition to pending and spawn
    update_run(conn, run_id, status="pending", preflight=json.dumps(report.to_dict()))
    spawn_training_process(run_id, _runs_dir() / run_id)
    return get_run(conn, run_id)


def delete_run(conn: sqlite3.Connection, run_id: str) -> bool:
    """Delete a training run record."""
    return _training.delete_run(conn, run_id)