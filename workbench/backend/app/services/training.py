"""Training run service layer: CRUD + run lifecycle (T3.4, T3.6)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app import config

RUN_ID_RE = re.compile(r"^r(\d+)$")

# Track active training processes: run_id -> Popen
_active_processes: dict[str, subprocess.Popen] = {}
# Strong references to background tasks so they are not garbage-collected
# mid-flight (documented asyncio.create_task caveat).
_bg_tasks: set[asyncio.Task] = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    return config.DB_PATH


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
    """Run one subprocess and update its lifecycle without overwriting stop."""
    from backend.app.db import connect

    log_path = run_dir / "training.log"

    def _mark_running() -> bool:
        conn = connect()
        try:
            cur = conn.execute(
                "UPDATE training_runs SET status = 'running' WHERE run_id = ? AND status = 'pending'",
                (run_id,),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()

    def _finish(status: str) -> None:
        conn = connect()
        try:
            adapter_path = run_dir / "adapter"
            report_path = run_dir / "report.md"
            conn.execute(
                """UPDATE training_runs
                   SET status = ?, completed_at = ?,
                       adapter_path = ?, report_path = ?
                   WHERE run_id = ? AND status = 'running'""",
                (
                    status, _now_iso(),
                    str(adapter_path) if adapter_path.exists() else None,
                    str(report_path) if report_path.exists() else None,
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    proc: subprocess.Popen | None = None
    try:
        # If a stop request won the race before this task started, do nothing.
        if not _mark_running():
            return
        script_path = run_dir / "train.py"
        if not script_path.exists():
            raise FileNotFoundError(f"train.py not found at {script_path}")

        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(run_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        _active_processes[run_id] = proc

        # Stop may have been requested after status became running but before
        # Popen completed. Re-read status before allowing the child to continue.
        conn = connect()
        try:
            row = conn.execute("SELECT status FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
            still_running = row is not None and row["status"] == "running"
        finally:
            conn.close()
        if not still_running and proc.poll() is None:
            proc.terminate()

        returncode = await asyncio.to_thread(proc.wait)
        _finish("completed" if returncode == 0 else "failed")
    except Exception as exc:  # record failures without changing a user-stopped run
        try:
            log_path.write_text(f"Training process error: {exc}\n", encoding="utf-8")
        except OSError:
            pass
        try:
            _finish("failed")
        except Exception:
            pass
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 10)
            except Exception:
                proc.kill()
        _active_processes.pop(run_id, None)


def spawn_training_process(run_id: str, run_dir: Path) -> None:
    """Start ``_run_training_process`` in the background from any context.

    - Inside a running event loop (FastAPI endpoint): schedules an asyncio
      task, keeping a strong reference so it survives GC.
    - Otherwise (tests, scripts): runs it on a daemon thread with its own
      loop — never a bare coroutine (which would be silently discarded).
    """
    coro = _run_training_process(run_id, run_dir)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        def _runner() -> None:
            try:
                asyncio.run(coro)
            except Exception:
                pass

        threading.Thread(target=_runner, name=f"train-{run_id}", daemon=True).start()
        return

    task = loop.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


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
    """Create a new training run record (status=pending) and generate its
    training script. Does NOT start training — callers that want a live
    process (router, CLI) call :func:`spawn_training_process` explicitly,
    after their own gating (pre-flight, plan 03 §6).
    """
    return create_run_sync(conn, config_json, dataset_version)


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
    """Stop a pending/running job and preserve the stopped state on completion."""
    record = get_run(conn, run_id)
    if record is None:
        return None
    if record["status"] not in ("pending", "running"):
        raise ValueError(f"cannot stop run with status {record['status']!r}")

    proc = _active_processes.get(run_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    _active_processes.pop(run_id, None)
    # Persist stop after the process exits; the async waiter only updates rows
    # still marked running, so it cannot overwrite this terminal state.
    return update_run(conn, run_id, status="stopped", completed_at=_now_iso())


def resume_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Resume a stopped/failed run: restart the training script in the
    background (the generated script picks up the latest checkpoint)."""
    record = get_run(conn, run_id)
    if record is None:
        return None
    if record["status"] not in ("stopped", "failed"):
        raise ValueError(f"cannot resume run with status {record['status']!r}")
    update_run(conn, run_id, status="pending")
    spawn_training_process(run_id, _runs_dir() / run_id)
    return get_run(conn, run_id)


def delete_run(conn: sqlite3.Connection, run_id: str) -> bool:
    """Delete a training run record."""
    cur = conn.execute(
        "DELETE FROM training_runs WHERE run_id = ?", (run_id,)
    )
    conn.commit()
    return cur.rowcount > 0
