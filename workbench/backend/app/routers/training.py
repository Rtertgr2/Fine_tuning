"""Training router API (T3.4): endpoints for managing training runs."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
import asyncio

from backend.app import config, schemas
from backend.app.db import get_db
from backend.app.services import training as svc
from backend.app.services.training_config import TrainingConfig
from backend.adapters.registry import list_adapters

router = APIRouter(tags=["training"])


@router.get(
    "/training/base-models",
    response_model=dict[str, Any],
)
def list_base_models():
    """List available base models: adapters + previously registered models."""
    adapters = list_adapters()
    # Also query model_versions table for previously registered models
    # For now, return adapters as base models
    models = []
    for a in adapters:
        models.append({
            "id": a["name"],
            "name": a["display_name"],
            "hf_repo": a["hf_repo"],
            "source": "adapter",
        })
    return {"items": models}


@router.post(
    "/training/preflight",
    response_model=dict[str, Any],
)
def run_preflight_only(payload: schemas.TrainingRunIn, conn: sqlite3.Connection = Depends(get_db)):
    """Run pre-flight checks PF1-PF6 against a config WITHOUT creating a run
    (the UI's 'ตรวจสอบ' button — plan 03 §6, plan 07 §8)."""
    cfg = TrainingConfig.from_dict(payload.config if payload.config else {})
    errors = cfg.validate()
    if errors:
        raise HTTPException(422, detail={"message": "config validation failed", "errors": errors})

    from backend.app.services.preflight import run_preflight

    return run_preflight(cfg).to_dict()


@router.post(
    "/training/runs",
    status_code=201,
    response_model=dict[str, Any],
)
async def start_training_run(
    payload: schemas.TrainingRunIn,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Create a training run and start it in the background.

    Pre-flight is mandatory before starting (plan 03 §6): if any critical
    check fails (PF1/PF3/...), the run is NOT created and the report is
    returned with 409, so the UI can show why training is locked.
    """
    cfg = TrainingConfig.from_dict(payload.config if payload.config else {})
    errors = cfg.validate()
    if errors:
        raise HTTPException(422, detail={"message": "config validation failed", "errors": errors})

    from backend.app.services.preflight import run_preflight

    report = await asyncio.to_thread(run_preflight, cfg)
    if not report.can_proceed:
        raise HTTPException(
            409,
            detail={
                "message": "pre-flight failed — training is locked until the "
                           "critical checks pass (plan 03 §6)",
                "preflight": report.to_dict(),
            },
        )

    try:
        record = svc.create_run_sync(
            conn,
            config_json=cfg.to_dict(),
            dataset_version=cfg.dataset_version,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    # Store the passing pre-flight report on the run record
    svc.update_run(conn, record["run_id"], preflight=json.dumps(report.to_dict()))

    # Start training in the background (asyncio task with a strong reference)
    svc.spawn_training_process(record["run_id"], config.DATA_DIR / "training" / "runs" / record["run_id"])

    return record


@router.get(
    "/training/runs/{run_id}",
    response_model=dict[str, Any],
)
def get_training_run(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Get training run status."""
    record = svc.get_run(conn, run_id)
    if record is None:
        raise HTTPException(404, f"training run {run_id!r} not found")
    return record


@router.get(
    "/training/runs",
    response_model=dict[str, Any],
)
def list_training_runs(
    status: str | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    """List training runs, optionally filtered by status."""
    items = svc.list_runs(conn, status_filter=status)
    return {"total": len(items), "items": items}


@router.post(
    "/training/runs/{run_id}/stop",
    response_model=dict[str, Any],
)
def stop_training_run(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Stop a training run."""
    record = svc.stop_run(conn, run_id)
    if record is None:
        raise HTTPException(404, f"training run {run_id!r} not found")
    return record


@router.post(
    "/training/runs/{run_id}/resume",
    response_model=dict[str, Any],
)
def resume_training_run(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Resume a stopped training run from its best checkpoint."""
    record = svc.resume_run(conn, run_id)
    if record is None:
        raise HTTPException(404, f"training run {run_id!r} not found")
    return record


@router.get(
    "/training/runs/{run_id}/metrics",
    response_model=dict[str, Any],
)
def get_training_metrics(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Get metrics for a training run (returns all entries).

    For SSE live streaming, the frontend can poll this endpoint periodically.
    """
    record = svc.get_run(conn, run_id)
    if record is None:
        raise HTTPException(404, f"training run {run_id!r} not found")

    from pathlib import Path
    from runner.metrics import load_metrics

    metrics_path = record.get("metrics_path")
    if metrics_path:
        entries = load_metrics(Path(metrics_path))
    else:
        entries = []

    return {"run_id": run_id, "metrics": entries}


@router.post(
    "/training/runs/{run_id}/preflight",
    response_model=dict[str, Any],
)
def run_preflight_endpoint(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    """Run pre-flight checks PF1-PF6 for a training run."""
    record = svc.get_run(conn, run_id)
    if record is None:
        raise HTTPException(404, f"training run {run_id!r} not found")

    # get_run returns "config" as parsed dict — use directly
    cfg = TrainingConfig.from_dict(record.get("config", {}))

    from backend.app.services.preflight import run_preflight
    report = run_preflight(cfg)

    # Store preflight result in the run record
    svc.update_run(conn, run_id, preflight=json.dumps(report.to_dict()))

    return report.to_dict()
