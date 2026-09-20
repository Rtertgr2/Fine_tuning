"""Active Learning case queue endpoints (T6.5, T6.7).

Endpoints:
  - GET /cases — List cases (filterable by type, priority, status)
  - GET /cases/{case_id} — Get case details with timeline
  - POST /cases/{case_id}/edit — Open in editor (pre-filled with original conversation)
  - POST /cases/{case_id}/approve — Approve edited case
  - POST /cases/{case_id}/reject — Reject with reason
  - POST /cases/scan — Run case detection on activity_log
  - GET /metrics/active-learning — Dashboard metrics
  - POST /datasets/from-active-learning — Build new dataset from approved cases
"""
from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app import schemas
from backend.app.db import get_db
from backend.app.services import cases as case_svc
from backend.app.services import metrics as metrics_svc
from backend.app.services import active_learning as al_svc
from backend.app.services import case_detector as detector_svc
from backend.app.services import activity_log as log_svc

router = APIRouter(tags=["cases"])


# ── Case Queue Endpoints ──────────────────────────────────────────────────


@router.get("/cases")
def list_cases(
    case_type: str | None = Query(default=None, description="Filter by case type"),
    priority: int | None = Query(default=None, ge=1, le=5, description="Filter by priority"),
    status: str | None = Query(default=None, description="Filter by status"),
    model_version: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    conn: sqlite3.Connection = Depends(get_db),
):
    """List cases with optional filters."""
    total, items = case_svc.list_cases(
        conn,
        case_type=case_type,
        priority=priority,
        status=status,
        model_version=model_version,
        limit=limit,
        offset=offset,
    )
    return {"total": total, "items": items}


@router.get("/cases/{case_id}")
def get_case(case_id: str, conn: sqlite3.Connection = Depends(get_db)):
    """Get case details with timeline."""
    case = case_svc.get_case(conn, case_id)
    if case is None:
        raise HTTPException(404, "case not found")
    return case


@router.post("/cases/{case_id}/edit")
def start_editing(
    case_id: str,
    reviewer: str = Query(..., description="Reviewer identifier"),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Open a case in the editor (marks it as being edited)."""
    case = case_svc.start_editing(conn, case_id, reviewer)
    if case is None:
        raise HTTPException(404, "case not found")
    return case


@router.post("/cases/{case_id}/approve")
def approve_case(
    case_id: str,
    final_example: dict[str, Any] | None = None,
    second_reviewer: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Approve a case. Sec cases require second_reviewer."""
    try:
        case = case_svc.approve_case(conn, case_id, final_example, second_reviewer)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if case is None:
        raise HTTPException(404, "case not found")
    return case


@router.post("/cases/{case_id}/reject")
def reject_case(
    case_id: str,
    reason: str = Query(..., description="Reason for rejection"),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Reject a case with reason."""
    case = case_svc.reject_case(conn, case_id, reason)
    if case is None:
        raise HTTPException(404, "case not found")
    return case


# ── Case Detection Endpoint ───────────────────────────────────────────────


@router.post("/cases/scan")
def scan_cases(
    sample_success_rate: float = Query(default=0.2, ge=0.0, le=1.0),
    since: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Run case detection on activity_log and enqueue detected cases."""
    cases = detector_svc.scan_all_sessions(
        conn, sample_success_rate=sample_success_rate, since=since
    )
    case_ids = case_svc.enqueue_batch(conn, cases)
    return {"scanned": len(case_ids), "case_ids": case_ids}


# ── Active Learning Metrics Endpoint ──────────────────────────────────────


@router.get("/metrics/active-learning")
def active_learning_metrics(conn: sqlite3.Connection = Depends(get_db)):
    """Active Learning dashboard metrics."""
    return metrics_svc.compute_metrics(conn)


# ── Active Learning Dataset Builder ───────────────────────────────────────


@router.post("/datasets/from-active-learning", response_model=schemas.DatasetOut)
def create_dataset_from_al(
    seed: int | None = Query(default=None),
    val_ratio: float | None = Query(default=None, gt=0, lt=0.5),
    categories: list[str] | None = Query(default=None),
    name: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Build a new dataset version mixing approved AL cases with original data."""
    try:
        record = al_svc.build_al_dataset(
            conn,
            seed=seed,
            val_ratio=val_ratio,
            categories=categories,
            name=name,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return record


@router.get("/datasets/threshold")
def check_threshold(conn: sqlite3.Connection = Depends(get_db)):
    """Check if enough approved cases exist to trigger a training round."""
    count, met = al_svc.check_threshold(conn)
    return {"approved_count": count, "threshold": al_svc.AL_THRESHOLD, "threshold_met": met}
