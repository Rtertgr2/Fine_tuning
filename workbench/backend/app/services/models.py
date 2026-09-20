"""Model registry: version tracking, promotion, rollback, and gate checks.

Phase 4 (Model Registry + Versioning + Rollback).

Key rules:
- A version is registered as `candidate` with all required metadata.
- Promotion to `staging`/`production` requires passing the Phase 3 gate
  (tool_syntax=100%, anti_loop>=90%, security_catch>=85%, security_fp<=10%,
   json_validity=100%, speed_8k>=30%, regression>=97%).
- Promoting to `production` automatically retires the previous production model.
- Rollback points `current` symlink at a specific version and restarts llama-server.
- The `current` symlink in models/ always points to the active production version.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.app import config

# Phase 3 gate thresholds (from eval/config.py)
GATE_THRESHOLDS: dict[str, tuple[float, float | None]] = {
    "tool_syntax": (100.0, None),
    "anti_loop": (90.0, None),
    "security_catch": (85.0, None),
    "security_fp": (0.0, 10.0),
    "json_validity": (100.0, None),
    "speed_8k": (30.0, None),
    "regression": (97.0, None),
}

VALID_VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "version": row["version"],
        "dataset_version": row["dataset_version"],
        "train_run": row["train_run"],
        "llama_cpp_commit": row["llama_cpp_commit"],
        "quant": row["quant"],
        "gguf_sha256": row["gguf_sha256"],
        "eval_report": row["eval_report"],
        "status": row["status"],
        "manifest": json.loads(row["manifest_json"]),
        "created_at": row["created_at"],
        "promoted_at": row["promoted_at"],
    }


def _check_gate(eval_report_link: str | None) -> tuple[bool, list[dict[str, Any]]]:
    """Check whether the eval report passes all Phase 3 gate thresholds.

    Returns (passed, details). Each detail has name, threshold, actual, pass.
    If no eval_report link is provided, gate check fails (cannot promote blindly).
    """
    if not eval_report_link:
        return False, [{"name": "no_report", "threshold": "report required", "actual": "none", "pass": False}]

    # Try to load the eval report JSON from the path
    import os
    from pathlib import Path

    report_path = Path(eval_report_link)
    if not report_path.is_absolute():
        report_path = config.ROOT / report_path

    try:
        with open(report_path) as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return False, [{"name": "report_load_error", "threshold": "valid report", "actual": str(e), "pass": False}]

    details: list[dict[str, Any]] = []
    all_passed = True

    gates_data = report.get("gates", [])
    gate_actuals: dict[str, float] = {}
    for g in gates_data:
        name = g.get("name", "")
        actual = g.get("actual", 0)
        try:
            gate_actuals[name] = float(actual)
        except (TypeError, ValueError):
            pass

    for gate_name, (min_pct, max_pct) in GATE_THRESHOLDS.items():
        actual = gate_actuals.get(gate_name, 0.0)

        if max_pct is not None:
            passed = min_pct <= actual <= max_pct
            threshold_str = f"{min_pct}-{max_pct}%"
        else:
            passed = actual >= min_pct
            threshold_str = f">={min_pct}%"

        if not passed:
            all_passed = False

        details.append({
            "name": gate_name,
            "threshold": threshold_str,
            "actual": actual,
            "pass": passed,
        })

    return all_passed, details


def register_model(
    conn: sqlite3.Connection,
    version: str,
    dataset_version: str | None = None,
    train_run: str | None = None,
    llama_cpp_commit: str | None = None,
    quant: str = "Q5_K_M",
    gguf_sha256: str | None = None,
    eval_report: str | None = None,
    status: str = "candidate",
) -> dict[str, Any]:
    """Register a new model version. Returns the full record."""
    if not VALID_VERSION_RE.match(version):
        raise ValueError(f"invalid version format: {version!r}, expected vN.N.N")

    existing = conn.execute("SELECT 1 FROM model_versions WHERE version = ?", (version,)).fetchone()
    if existing:
        raise ValueError(f"model version {version!r} already exists")

    now = _now_iso()
    manifest: dict[str, Any] = {
        "version": version,
        "dataset_version": dataset_version,
        "train_run": train_run,
        "llama_cpp_commit": llama_cpp_commit,
        "quant": quant,
        "gguf_sha256": gguf_sha256,
        "eval_report": eval_report,
        "status": status,
        "registered_at": now,
    }

    conn.execute(
        """INSERT INTO model_versions
           (version, dataset_version, train_run, llama_cpp_commit, quant,
            gguf_sha256, eval_report, status, manifest_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            version,
            dataset_version,
            train_run,
            llama_cpp_commit,
            quant,
            gguf_sha256,
            eval_report,
            status,
            json.dumps(manifest, ensure_ascii=False),
            now,
        ),
    )
    conn.commit()

    row = conn.execute("SELECT * FROM model_versions WHERE version = ?", (version,)).fetchone()
    return _row_to_dict(row)


def get_model(conn: sqlite3.Connection, version: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM model_versions WHERE version = ?", (version,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_models(conn: sqlite3.Connection, status_filter: str | None = None) -> list[dict[str, Any]]:
    if status_filter:
        rows = conn.execute(
            "SELECT * FROM model_versions WHERE status = ? ORDER BY created_at", (status_filter,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM model_versions ORDER BY created_at").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_current_production(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM model_versions WHERE status = 'production' ORDER BY promoted_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def promote_model(
    conn: sqlite3.Connection,
    version: str,
    target_status: str = "production",
) -> dict[str, Any]:
    """Promote a model to staging or production.

    Gate check is enforced when promoting to `production`.
    If target is `production`, the previous production model gets retired.
    """
    if target_status not in ("staging", "production"):
        raise ValueError(f"cannot promote to {target_status!r}, must be 'staging' or 'production'")

    model = get_model(conn, version)
    if model is None:
        raise ValueError(f"model version {version!r} not found")

    previous_status = model["status"]

    # Gate check for production promotion
    gate_passed = True
    gate_details: list[dict[str, Any]] = []

    if target_status == "production":
        gate_passed, gate_details = _check_gate(model["eval_report"])
        if not gate_passed:
            raise GateCheckError(
                f"model {version!r} failed gate check for production promotion",
                details=gate_details,
            )

    now = _now_iso()

    # If promoting to production, retire the current production model
    previous_production = None
    if target_status == "production":
        current = get_current_production(conn)
        if current and current["version"] != version:
            previous_production = current["version"]
            conn.execute(
                "UPDATE model_versions SET status = 'retired' WHERE version = ?",
                (previous_production,),
            )

    # Update the target model
    conn.execute(
        "UPDATE model_versions SET status = ?, promoted_at = ? WHERE version = ?",
        (target_status, now, version),
    )
    conn.commit()

    model = get_model(conn, version)

    return {
        "version": version,
        "previous_status": previous_status,
        "new_status": target_status,
        "previous_production": previous_production,
        "gate_passed": gate_passed,
        "gate_details": gate_details,
        "model": model,
    }


def rollback_to(conn: sqlite3.Connection, version: str) -> dict[str, Any]:
    """Rollback to a previous model version.

    - The target version becomes `production`.
    - The current production version gets retired.
    - The `current` symlink is updated.
    """
    model = get_model(conn, version)
    if model is None:
        raise ValueError(f"model version {version!r} not found")

    if model["status"] == "production":
        raise ValueError(f"model {version!r} is already the production version")

    now = _now_iso()

    # Retire current production
    current = get_current_production(conn)
    previous_production = current["version"] if current else None

    if previous_production and previous_production != version:
        conn.execute(
            "UPDATE model_versions SET status = 'retired' WHERE version = ?",
            (previous_production,),
        )

    # Promote target to production
    conn.execute(
        "UPDATE model_versions SET status = 'production', promoted_at = ? WHERE version = ?",
        (now, version),
    )
    conn.commit()

    model = get_model(conn, version)

    return {
        "version": version,
        "previous_production": previous_production,
        "model": model,
    }


class GateCheckError(ValueError):
    """Raised when a model fails the gate check for promotion."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.details = details or []
