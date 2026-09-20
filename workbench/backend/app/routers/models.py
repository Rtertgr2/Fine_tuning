"""Model Registry endpoints (Phase 4): register, promote, rollback, list, current.

T5.6 / T5.7:
- POST /models/register — register a new model version
- POST /models/{version}/promote — promote (with gate validation for production)
- POST /models/{version}/rollback — rollback to this version
- GET /models — list all versions with status filter
- GET /models/current — get current production model
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app import schemas
from backend.app.db import get_db
from backend.app.services import models as svc

router = APIRouter(tags=["models"])


@router.post(
    "/models/register",
    status_code=201,
    response_model=schemas.ModelOut,
)
def register_model(
    payload: schemas.ModelRegisterIn,
    conn: sqlite3.Connection = Depends(get_db),
):
    try:
        record = svc.register_model(
            conn,
            version=payload.version,
            dataset_version=payload.dataset_version,
            train_run=payload.train_run,
            llama_cpp_commit=payload.llama_cpp_commit,
            quant=payload.quant,
            gguf_sha256=payload.gguf_sha256,
            eval_report=payload.eval_report,
            status=payload.status,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return record


@router.post(
    "/models/{version}/promote",
    response_model=schemas.PromoteResult,
)
def promote_model(
    version: str,
    payload: schemas.ModelPromoteIn | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    target = payload.target_status if payload else "production"
    try:
        result = svc.promote_model(conn, version, target_status=target)
    except svc.GateCheckError as exc:
        raise HTTPException(
            403,
            detail={
                "message": str(exc),
                "gate_details": exc.details,
                "version": version,
                "target_status": target,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    msg = f"model {version} promoted from {result['previous_status']} to {result['new_status']}"
    if result.get("previous_production"):
        msg += f" (retired {result['previous_production']})"

    return schemas.PromoteResult(
        version=result["version"],
        previous_status=result["previous_status"],
        new_status=result["new_status"],
        previous_production=result.get("previous_production"),
        gate_passed=result["gate_passed"],
        message=msg,
    )


@router.post(
    "/models/{version}/rollback",
    response_model=schemas.RollbackResult,
)
def rollback_model(
    version: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    try:
        result = svc.rollback_to(conn, version)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    msg = f"rolled back to {version}"
    if result.get("previous_production"):
        msg += f" (retired {result['previous_production']})"

    return schemas.RollbackResult(
        version=result["version"],
        previous_production=result.get("previous_production"),
        message=msg,
    )


@router.get(
    "/models",
    response_model=schemas.ModelListOut,
)
def list_models(
    status: schemas.ModelStatus | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    items = svc.list_models(conn, status_filter=status)
    return {"total": len(items), "items": items}


@router.get(
    "/models/current",
    response_model=schemas.ModelOut,
)
def get_current_production(conn: sqlite3.Connection = Depends(get_db)):
    record = svc.get_current_production(conn)
    if record is None:
        raise HTTPException(404, "no production model found")
    return record


@router.get(
    "/models/{version}",
    response_model=schemas.ModelOut,
)
def get_model(
    version: str,
    conn: sqlite3.Connection = Depends(get_db),
):
    record = svc.get_model(conn, version)
    if record is None:
        raise HTTPException(404, f"model version {version!r} not found")
    return record
