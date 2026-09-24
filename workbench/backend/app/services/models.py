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

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app import config

# Phase 3 gate thresholds (from eval/config.py)
GATE_THRESHOLDS: dict[str, tuple[float, float | None]] = {
    "tool_syntax": (100.0, None),
    "anti_loop": (90.0, None),
    "security_catch": (85.0, None),
    "security_fp": (0.0, 10.0),
    "json_validity": (100.0, None),
    "plan_json": (100.0, None),
    "speed_8k": (30.0, None),
    "regression": (97.0, None),  # candidate/base pass@1 ratio
    "end_to_end": (100.0, None),
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


def _update_current_symlink(version: str) -> dict[str, Any]:
    """Atomically point ``models/current`` at a complete version directory."""
    import uuid

    models_dir = config.MODELS_DIR
    target_dir = models_dir / version
    link = models_dir / "current"
    temporary = models_dir / f".current-{uuid.uuid4().hex}"
    try:
        if not target_dir.is_dir():
            raise FileNotFoundError(f"model artifact directory not found: {target_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
        if link.exists() and not link.is_symlink():
            raise OSError(f"refusing to replace non-symlink current path: {link}")
        os.symlink(version, temporary)
        os.replace(temporary, link)
        return {"action": "symlink", "target": version, "status": "ok"}
    except OSError as e:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return {"action": "symlink", "target": version, "status": f"failed: {e}"}


def _restart_llama_server() -> dict[str, Any]:
    """Optionally restart llama-server after a symlink switch (plan 05 §5).

    Only runs when WORKBENCH_LLAMA_SERVER_CMD is configured. The command must
    itself launch the server in the background and return promptly (e.g.
    ``pkill -f llama-server; nohup llama-server -m models/current/... &``) —
    the workbench will wait at most 120 s and then give up.
    """
    import subprocess

    cmd = config.LLAMA_SERVER_CMD
    if not cmd:
        return {"action": "llama_server_restart", "status": "skipped (not configured)"}
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        tail = (proc.stderr or proc.stdout or "")[-300:]
        return {"action": "llama_server_restart", "status": status, "output": tail}
    except Exception as e:  # noqa: BLE001
        return {"action": "llama_server_restart", "status": f"failed: {e}"}


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(model: dict[str, Any]):
    """Resolve and verify the registered GGUF (path, size, magic, SHA256)."""
    manifest = model.get("manifest") or {}
    stored_path = manifest.get("gguf_path")
    if not stored_path:
        raise ValueError("model manifest is missing gguf_path")
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = config.MODELS_DIR / candidate
    try:
        path = candidate.resolve(strict=True)
        models_root = config.MODELS_DIR.resolve()
        if not path.is_relative_to(models_root):
            raise ValueError("GGUF path must be inside WORKBENCH_MODELS_DIR")
    except FileNotFoundError as exc:
        raise ValueError(f"GGUF artifact does not exist: {candidate}") from exc
    if not path.is_file() or path.stat().st_size < 4:
        raise ValueError("GGUF artifact is missing or empty")
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("registered artifact is not a GGUF file (bad magic)")
    actual = _sha256_file(path)
    expected = str(model.get("gguf_sha256") or "").lower()
    if not expected or actual != expected:
        raise ValueError(f"GGUF SHA256 mismatch (expected {expected or 'missing'}, got {actual})")
    return path


def _check_gate(eval_report_link: str | None) -> tuple[bool, list[dict[str, Any]]]:
    """Load a report from the report store and verify every production gate.

    Missing, malformed, stale/incomplete or failed measurements all fail
    closed. Paths are confined to eval/reports so an API caller cannot use the
    report field to read arbitrary files from the server.
    """
    if not eval_report_link:
        return False, [{"name": "no_report", "threshold": "report required", "actual": None, "pass": False}]

    from pathlib import Path

    raw_path = Path(eval_report_link)
    if not raw_path.is_absolute():
        raw_path = config.ROOT / raw_path
    report_root = (config.EVAL_DIR / "reports").resolve()
    try:
        report_path = raw_path.resolve(strict=True)
        if not report_path.is_relative_to(report_root):
            return False, [{"name": "report_path", "threshold": "inside eval/reports", "actual": str(report_path), "pass": False}]
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return False, [{"name": "report_load_error", "threshold": "valid report", "actual": str(exc), "pass": False}]

    if not isinstance(report, dict):
        return False, [{"name": "report_shape", "threshold": "JSON object", "actual": type(report).__name__, "pass": False}]

    raw_gates = report.get("gates")
    gate_by_name = {
        str(g.get("name")): g for g in raw_gates
        if isinstance(g, dict) and g.get("name")
    } if isinstance(raw_gates, list) else {}
    details: list[dict[str, Any]] = []
    all_passed = report.get("overall_pass") is True

    for gate_name, (min_value, max_value) in GATE_THRESHOLDS.items():
        source = gate_by_name.get(gate_name)
        threshold = f"{min_value:g}-{max_value:g}%" if max_value is not None else f">={min_value:g}"
        if gate_name == "speed_8k":
            threshold += " tok/s"
        else:
            threshold += "%"

        actual = None
        try:
            if source is not None and source.get("actual") is not None:
                actual = float(source["actual"])
        except (TypeError, ValueError):
            actual = None

        within_threshold = actual is not None and (
            min_value <= actual <= max_value if max_value is not None else actual >= min_value
        )
        passed = bool(source and source.get("pass") is True and within_threshold)
        if not passed:
            all_passed = False
        details.append({
            "name": gate_name,
            "threshold": threshold,
            "actual": actual,
            "pass": passed,
            "note": None if source else "required gate missing from report",
        })

    if report.get("overall_pass") is not True:
        details.append({
            "name": "report_overall",
            "threshold": "overall_pass=true",
            "actual": report.get("overall_pass"),
            "pass": False,
        })
    return all_passed, details

def register_model(
    conn: sqlite3.Connection,
    version: str,
    dataset_version: str | None = None,
    train_run: str | None = None,
    llama_cpp_commit: str | None = None,
    quant: str = "Q5_K_M",
    gguf_path: str | None = None,
    gguf_sha256: str | None = None,
    eval_report: str | None = None,
    status: str = "candidate",
) -> dict[str, Any]:
    """Register an immutable candidate only after verifying its GGUF artifact."""
    if not VALID_VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid version format: {version!r}, expected vN.N.N")
    if status != "candidate":
        raise ValueError("new model versions must be registered as 'candidate'; use the gate-checked promote endpoint")
    if quant not in {"Q4_K_M", "Q5_K_M", "Q8_0", "Q2_K", "Q3_K_M", "Q6_K"}:
        raise ValueError(f"unsupported quantization level: {quant!r}")
    if not gguf_path:
        raise ValueError("gguf_path is required; register a real, hashed GGUF artifact")

    existing = conn.execute("SELECT 1 FROM model_versions WHERE version = ?", (version,)).fetchone()
    if existing:
        raise ValueError(f"model version {version!r} already exists")

    model_root = config.MODELS_DIR.resolve()
    raw_path = Path(gguf_path)
    if not raw_path.is_absolute():
        raw_path = config.MODELS_DIR / raw_path
    try:
        artifact = raw_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"GGUF artifact not found: {raw_path}") from exc
    if not artifact.is_relative_to(model_root):
        raise ValueError("gguf_path must be inside WORKBENCH_MODELS_DIR")
    version_dir = (model_root / version).resolve()
    if artifact.parent != version_dir:
        raise ValueError(f"GGUF must be stored directly inside {version_dir}")
    if not artifact.is_file() or artifact.stat().st_size < 4:
        raise ValueError("GGUF artifact is missing or empty")
    with artifact.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("gguf_path does not point to a GGUF file (bad magic)")
    actual_sha256 = _sha256_file(artifact)
    if gguf_sha256 and gguf_sha256.lower() != actual_sha256:
        raise ValueError(f"GGUF SHA256 mismatch: expected {gguf_sha256.lower()}, got {actual_sha256}")

    now = _now_iso()
    relative_artifact = str(artifact.relative_to(model_root))
    manifest: dict[str, Any] = {
        "version": version,
        "dataset_version": dataset_version,
        "train_run": train_run,
        "llama_cpp_commit": llama_cpp_commit,
        "quant": quant,
        "gguf_path": relative_artifact,
        "gguf_sha256": actual_sha256,
        "gguf_size_bytes": artifact.stat().st_size,
        "eval_report": eval_report,
        "status": "candidate",
        "registered_at": now,
    }

    # Persist the same manifest beside the GGUF for portability/auditing.
    manifest_path = artifact.parent / "manifest.json"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + os.linesep, encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)

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
            actual_sha256,
            eval_report,
            "candidate",
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


def attach_eval_report(
    conn: sqlite3.Connection,
    version: str,
    eval_report_link: str,
) -> dict[str, Any]:
    """Attach a completed eval report to a candidate version (including failures)."""
    model = get_model(conn, version)
    if model is None:
        raise ValueError(f"model version {version!r} not found")
    if model["status"] not in ("candidate", "staging"):
        raise ValueError("eval reports may only be attached before production")
    raw_path = Path(eval_report_link)
    if not raw_path.is_absolute():
        raw_path = config.ROOT / raw_path
    report_root = (config.EVAL_DIR / "reports").resolve()
    try:
        report_path = raw_path.resolve(strict=True)
        if not report_path.is_relative_to(report_root):
            raise ValueError("eval report must be inside workbench/eval/reports")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load eval report: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("gates"), list):
        raise ValueError("eval report must be a JSON object with a gates array")

    # Verify report provenance: model_id, revision, and artifact_hash must match.
    report_model_id = str(report.get("model_id", ""))
    report_revision = str(report.get("revision", ""))
    report_artifact_hash = str(report.get("artifact_hash", ""))
    manifest = model["manifest"]
    expected_artifact_hash = str(model.get("gguf_sha256") or manifest.get("gguf_sha256", "")).lower()

    if report_revision and report_revision != version:
        raise ValueError(
            f"report revision mismatch: report says {report_revision!r}, model version is {version!r}"
        )
    if report_artifact_hash and report_artifact_hash.lower() != expected_artifact_hash:
        raise ValueError(
            f"report artifact_hash mismatch: report says {report_artifact_hash!r}, expected {expected_artifact_hash or 'missing'}"
        )

    stored_path = str(report_path.relative_to(config.ROOT))
    manifest = model["manifest"]
    manifest["eval_report"] = stored_path
    manifest["eval_report_run_id"] = report.get("run_id")
    artifact = _resolve_artifact(model)
    manifest_path = artifact.parent / "manifest.json"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + os.linesep, encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    conn.execute(
        "UPDATE model_versions SET eval_report = ?, manifest_json = ? WHERE version = ?",
        (stored_path, json.dumps(manifest, ensure_ascii=False), version),
    )
    conn.commit()
    return get_model(conn, version)


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


def current_symlink_target() -> str | None:
    """Version that models/current points at on disk (None if absent)."""
    import os

    link = config.MODELS_DIR / "current"
    if link.is_symlink():
        return os.readlink(link)
    return None


def promote_model(
    conn: sqlite3.Connection,
    version: str,
    target_status: str = "production",
) -> dict[str, Any]:
    """Promote a candidate to staging or production with safe deployment."""
    if target_status not in ("staging", "production"):
        raise ValueError(f"cannot promote to {target_status!r}, must be 'staging' or 'production'")

    model = get_model(conn, version)
    if model is None:
        raise ValueError(f"model version {version!r} not found")
    previous_status = model["status"]

    gate_details: list[dict[str, Any]] = []
    if target_status == "production":
        try:
            _resolve_artifact(model)
        except ValueError as exc:
            raise GateCheckError(
                f"model {version!r} artifact check failed",
                details=[{"name": "gguf_artifact", "threshold": "exists and matches SHA256", "actual": str(exc), "pass": False}],
            ) from exc
        gate_passed, gate_details = _check_gate(model["eval_report"])
        if not gate_passed:
            raise GateCheckError(
                f"model {version!r} failed gate check for production promotion",
                details=gate_details,
            )
    else:
        gate_passed = True

    now = _now_iso()
    previous_production = None
    actions: list[dict[str, Any]] = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        if target_status == "production":
            current = get_current_production(conn)
            if current and current["version"] != version:
                previous_production = current["version"]
                conn.execute("UPDATE model_versions SET status = 'retired' WHERE version = ?", (previous_production,))

        conn.execute(
            "UPDATE model_versions SET status = ?, promoted_at = ? WHERE version = ?",
            (target_status, now, version),
        )
        if target_status == "production":
            link_action = _update_current_symlink(version)
            if link_action["status"] != "ok":
                raise OSError(link_action["status"])
            actions.append(link_action)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if target_status == "production":
        actions.append(_restart_llama_server())

    return {
        "version": version,
        "previous_status": previous_status,
        "new_status": target_status,
        "previous_production": previous_production,
        "gate_passed": gate_passed,
        "gate_details": gate_details,
        "actions": actions,
        "model": get_model(conn, version),
    }


def rollback_to(conn: sqlite3.Connection, version: str) -> dict[str, Any]:
    """Rollback to a retained, hash-verified model and update current atomically."""
    model = get_model(conn, version)
    if model is None:
        raise ValueError(f"model version {version!r} not found")
    if model["status"] == "production":
        raise ValueError(f"model {version!r} is already the production version")
    try:
        _resolve_artifact(model)
    except ValueError as exc:
        raise ValueError(f"cannot rollback to {version}: {exc}") from exc

    now = _now_iso()
    current = get_current_production(conn)
    previous_production = current["version"] if current else None
    conn.execute("BEGIN IMMEDIATE")
    try:
        if previous_production and previous_production != version:
            conn.execute("UPDATE model_versions SET status = 'retired' WHERE version = ?", (previous_production,))
        conn.execute(
            "UPDATE model_versions SET status = 'production', promoted_at = ? WHERE version = ?",
            (now, version),
        )
        link_action = _update_current_symlink(version)
        if link_action["status"] != "ok":
            raise OSError(link_action["status"])
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    actions = [link_action, _restart_llama_server()]
    return {
        "version": version,
        "previous_production": previous_production,
        "actions": actions,
        "model": get_model(conn, version),
    }


class GateCheckError(ValueError):
    """Raised when a model fails the gate check for promotion."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.details = details or []
