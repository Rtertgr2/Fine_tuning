"""Draft validation endpoints."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends

from backend.app import schemas
from backend.app.db import get_db
from backend.app.services import examples as svc
from backend.validators.registry import all_rules_for, report_to_dict

router = APIRouter(tags=["validation"])


@router.post("/validate", response_model=schemas.ValidationReportOut)
def validate_draft(data: schemas.ExampleIn, conn: sqlite3.Connection = Depends(get_db)):
    report = svc.validate_draft(conn, data)
    return report_to_dict(report)


@router.get("/validators")
def list_validators():
    """Rule inventory per category — used by the UI to explain codes."""
    out: dict[str, list[dict[str, str]]] = {}
    for category in ("tool", "loop", "plan", "sec"):
        out[category] = [
            {"code": r.code, "level": r.level, "scope": r.scope}
            for r in all_rules_for(category)
        ]
    return out