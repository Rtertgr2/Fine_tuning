"""Training run service layer: CRUD + run lifecycle (T3.4, T3.6)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app import config

RUN_ID_RE = re.compile(r"^r(\d+)$")

# Track active training processes: run_id -> Popen
_active_processes: dict[str, subprocess.Popen] = {}


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


def _generate_training_script(run_dir: Path, config_json: dict[str, Any]) -> Path:
    """Generate training script from config into run_dir."""
    from backend.app.services.training_config import TrainingConfig
    from scripts.generate_training_script import generate_script

    cfg = TrainingConfig.from_dict(config_json)
    output = run_dir / "train.py"
    generate_script(cfg=cfg, output_path=str(output))
    return output


async def _run_training_process(run_id: str, run_dir: Path) -> None:
    """Async background task: run training subprocess, update DB when done."""
    import sqlite3 as sq
    log_path = run_dir / "training.log"
    
    try:
        # Update status to running
        conn = sq.connect(str(config.DATA_DIR / "examples.db"))
        conn.execute("UPDATE training_runs SET status = ? WHERE run_id = ?", ("running", run_id))
        conn.commit()
        conn.close()
        
        # Find training script
        script_path = run_dir / "train.py"
        if not script_path.exists():
            raise FileNotFoundError(f"train.py not found at {script_path}")
        
        # Run training asynchronously — output to log file
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            cwd=str(run_dir),
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
        )
        _active_processes[run_id] = proc
        
        # Wait for completion (non-blocking for event loop)
        returncode = await proc.wait()
        log_fh.close()
        
        # Update DB with result
        conn = sq.connect(str(config.DATA_DIR / "examples.db"))
        if returncode == 0:
            conn.execute(
                "UPDATE training_runs SET status = ?, completed_at = ? WHERE run_id = ?",
                ("completed", _now_iso(), run_id),
            )
        else:
            conn.execute(
                "UPDATE training_runs SET status = ?, completed_at = ? WHERE run_id = ?",
                ("failed", _now_iso(), run_id),
            )
        conn.commit()
        conn.close()
        
    except Exception as e:
        # Write error to log
        log_path.write_text(f"Training process error: {e}", encoding="utf-8")
        # Update status to failed
        try:
            conn = sq.connect(str(config.DATA_DIR / "examples.db"))
            conn.execute(
                "UPDATE training_runs SET status = ?, completed_at = ? WHERE run_id = ?",
                ("failed", _now_iso(), run_id),
            )
            conn.commit()
            conn.close()
        except:
            pass
    finally:
        _active_processes.pop(run_id, None)


def create_run_sync(
    conn: sqlite3.Connection,
    config_json: dict[str, Any],
    dataset_version: str = "",
) -> dict[str, Any]:
    """Create a new training run record (synchronous version for async endpoints).
    
    Generates training script immediately but does NOT start training.
    The caller is responsible for starting the background process.
    """
    run_id = _next_run_id(conn)
    now = _now_iso()

    # Set up output directory
    run_dir = _runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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

    # Generate training script
    _generate_training_script(run_dir, config_json)

    row = conn.execute(
        "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return _row_to_dict(row)


def create_run(
    conn: sqlite3.Connection,
    config_json: dict[str, Any],
    dataset_version: str = "",
) -> dict[str, Any]:
    """Create a new training run record and set up its output directory.
    
    Immediately generates training script and starts training in background thread.
    """
    record = create_run_sync(conn, config_json, dataset_version)

    # Start training in background (async task)
    run_id = record["run_id"]
    runs_dir = _runs_dir() / run_id
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_training_process(run_id, runs_dir))
    except RuntimeError:
        # No running loop — use threading
        import threading
        t = threading.Thread(target=_run_training_process, args=(run_id, runs_dir), daemon=False)
        t.start()

    return record


def start_run(run_id: str) -> None:
    """Start training in background."""
    from fastapi import BackgroundTasks
    # This will be called from the router with BackgroundTasks
    pass


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
    """Stop a training run: kill process + update status."""
    proc = _active_processes.get(run_id)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    _active_processes.pop(run_id, None)
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
