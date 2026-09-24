"""Model registry (Phase 4): register, promote, rollback, list."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app import schemas
from backend.app.db import init_db
from backend.app.services import models as svc
from backend.app import config


@pytest.fixture()
def client(tmp_workbench):
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _model_payload(version: str = "v0.1.0", **overrides) -> dict:
    import hashlib

    artifact_dir = config.MODELS_DIR / version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model-Q5_K_M.gguf"
    if not artifact_path.exists():
        artifact_path.write_bytes(b"GGUF" + version.encode("ascii") + b"-test-artifact")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    base = {
        "version": version,
        "dataset_version": "v0001",
        "train_run": "r001",
        "llama_cpp_commit": "abc123def",
        "quant": "Q5_K_M",
        "gguf_path": str(artifact_path),
        "gguf_sha256": digest,
        "eval_report": None,
        "status": "candidate",
    }
    base.update(overrides)
    return base


def _write_eval_report(tmp_path, gates: list | None = None) -> str:
    """Write a passing eval report to disk; return its path (relative to ROOT)."""
    if gates is None:
        gates = [
            {"name": "tool_syntax", "threshold": ">=100%", "actual": 100.0, "pass": True},
            {"name": "anti_loop", "threshold": ">=90%", "actual": 95.0, "pass": True},
            {"name": "security_catch", "threshold": ">=85%", "actual": 90.0, "pass": True},
            {"name": "security_fp", "threshold": "0-10%", "actual": 5.0, "pass": True},
            {"name": "json_validity", "threshold": ">=100%", "actual": 100.0, "pass": True},
            {"name": "speed_8k", "threshold": ">=30 tok/s", "actual": 45.0, "pass": True},
            {"name": "regression", "threshold": ">=97%", "actual": 98.0, "pass": True},
            {"name": "plan_json", "threshold": ">=100%", "actual": 100.0, "pass": True},
            {"name": "end_to_end", "threshold": ">=100%", "actual": 100.0, "pass": True},
        ]
    report = {
        "run_id": "test_run",
        "model": "test-model",
        "created_at": "2026-01-01T00:00:00Z",
        "config": {},
        "suites": [],
        "gates": gates,
        "overall_pass": all(g["pass"] for g in gates),
    }
    # Use absolute path so it works regardless of config.ROOT
    # The service handles both absolute and relative paths.
    import uuid
    unique_name = f"test_report_{uuid.uuid4().hex[:8]}.json"
    path = config.ROOT / "eval" / "reports" / unique_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return str(path)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_model_version(client):
    r = client.post("/models/register", json=_model_payload())
    assert r.status_code == 201
    body = r.json()
    assert body["version"] == "v0.1.0"
    assert body["status"] == "candidate"
    assert body["quant"] == "Q5_K_M"
    assert body["manifest"]["gguf_sha256"]


def test_register_rejects_invalid_version_format(client):
    r = client.post("/models/register", json=_model_payload(version="1.0"))
    assert r.status_code == 422


def test_register_cannot_bypass_candidate_gate(client):
    r = client.post("/models/register", json=_model_payload(status="production"))
    assert r.status_code == 422


def test_register_rejects_tampered_artifact_hash(client):
    r = client.post("/models/register", json=_model_payload(gguf_sha256="0" * 64))
    assert r.status_code == 422


def test_register_rejects_duplicate_version(client):
    client.post("/models/register", json=_model_payload())
    r = client.post("/models/register", json=_model_payload())
    assert r.status_code == 422


def test_register_default_quant(client):
    payload = _model_payload()
    del payload["quant"]
    r = client.post("/models/register", json=payload)
    assert r.status_code == 201
    assert r.json()["quant"] == "Q5_K_M"


def test_attach_eval_report_after_registration(client, tmp_workbench):
    client.post("/models/register", json=_model_payload())
    report_path = _write_eval_report(tmp_workbench)
    response = client.post("/models/v0.1.0/eval-report", json={"eval_report": report_path})
    assert response.status_code == 200
    assert response.json()["eval_report"] == response.json()["manifest"]["eval_report"]
    assert response.json()["manifest"]["eval_report_run_id"] == "test_run"


# ---------------------------------------------------------------------------
# list & get
# ---------------------------------------------------------------------------


def test_list_models_empty(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json() == {"total": 0, "items": []}


def test_list_models_by_status(client):
    client.post("/models/register", json=_model_payload("v0.1.0"))
    client.post("/models/register", json=_model_payload("v0.2.0"))

    r = client.get("/models", params={"status": "candidate"})
    assert r.status_code == 200
    assert r.json()["total"] == 2

    r = client.get("/models", params={"status": "production"})
    assert r.json()["total"] == 0


def test_get_current_production_none(client):
    r = client.get("/models/current")
    assert r.status_code == 404


def test_get_specific_version(client):
    client.post("/models/register", json=_model_payload("v0.1.0"))
    r = client.get("/models/v0.1.0")
    assert r.status_code == 200
    assert r.json()["version"] == "v0.1.0"


def test_get_specific_version_not_found(client):
    r = client.get("/models/v9.9.9")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


def test_promote_to_staging(client):
    client.post("/models/register", json=_model_payload("v0.1.0"))
    r = client.post("/models/v0.1.0/promote", json={"target_status": "staging"})
    assert r.status_code == 200
    body = r.json()
    assert body["new_status"] == "staging"
    assert body["previous_status"] == "candidate"
    assert body["gate_passed"] is True  # staging skips gate


def test_promote_to_production_passes_with_report(client, tmp_workbench):
    report_path = _write_eval_report(tmp_workbench)
    client.post(
        "/models/register",
        json=_model_payload("v0.1.0", eval_report=report_path),
    )
    r = client.post("/models/v0.1.0/promote", json={"target_status": "production"})
    assert r.status_code == 200
    body = r.json()
    assert body["new_status"] == "production"
    assert body["gate_passed"] is True
    assert body["actions"][0]["status"] == "ok"
    assert body["actions"][0]["target"] == "v0.1.0"

    # Verify current production
    r = client.get("/models/current")
    assert r.status_code == 200
    assert r.json()["version"] == "v0.1.0"


def test_promote_to_production_blocked_without_report(client):
    client.post("/models/register", json=_model_payload("v0.1.0"))
    r = client.post("/models/v0.1.0/promote", json={"target_status": "production"})
    assert r.status_code == 403
    body = r.json()["detail"]
    assert "gate" in body["message"].lower() or "report" in body["message"].lower()


def test_promote_to_production_blocked_by_failed_gate(client, tmp_workbench):
    failing_gates = [
        {"name": "tool_syntax", "threshold": ">=100%", "actual": 95.0, "pass": False},
        {"name": "anti_loop", "threshold": ">=90%", "actual": 95.0, "pass": True},
        {"name": "security_catch", "threshold": ">=85%", "actual": 90.0, "pass": True},
        {"name": "security_fp", "threshold": "0-10%", "actual": 5.0, "pass": True},
        {"name": "json_validity", "threshold": ">=100%", "actual": 100.0, "pass": True},
        {"name": "speed_8k", "threshold": ">=30%", "actual": 45.0, "pass": True},
        {"name": "regression", "threshold": ">=97%", "actual": 98.0, "pass": True},
    ]
    report_path = _write_eval_report(tmp_workbench, gates=failing_gates)
    client.post(
        "/models/register",
        json=_model_payload("v0.1.0", eval_report=report_path),
    )
    r = client.post("/models/v0.1.0/promote", json={"target_status": "production"})
    assert r.status_code == 403
    body = r.json()["detail"]
    assert body["gate_details"][0]["pass"] is False


def test_promote_retires_previous_production(client, tmp_workbench):
    report_path = _write_eval_report(tmp_workbench)

    client.post(
        "/models/register",
        json=_model_payload("v0.1.0", eval_report=report_path),
    )
    client.post("/models/v0.1.0/promote", json={"target_status": "production"})

    client.post(
        "/models/register",
        json=_model_payload("v0.2.0", eval_report=report_path),
    )
    r = client.post("/models/v0.2.0/promote", json={"target_status": "production"})
    body = r.json()
    assert body["previous_production"] == "v0.1.0"

    # v0.1.0 should be retired
    r = client.get("/models/v0.1.0")
    assert r.json()["status"] == "retired"

    # current should be v0.2.0
    r = client.get("/models/current")
    assert r.json()["version"] == "v0.2.0"


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def test_rollback_to_previous_version(client, tmp_workbench):
    report_path = _write_eval_report(tmp_workbench)

    client.post(
        "/models/register",
        json=_model_payload("v0.1.0", eval_report=report_path),
    )
    client.post("/models/v0.1.0/promote", json={"target_status": "production"})

    client.post(
        "/models/register",
        json=_model_payload("v0.2.0", eval_report=report_path),
    )
    client.post("/models/v0.2.0/promote", json={"target_status": "production"})

    # Rollback to v0.1.0
    r = client.post("/models/v0.1.0/rollback")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "v0.1.0"
    assert body["previous_production"] == "v0.2.0"
    assert body["actions"][0]["status"] == "ok"
    assert body["actions"][0]["target"] == "v0.1.0"

    # v0.1.0 is now production
    r = client.get("/models/current")
    assert r.json()["version"] == "v0.1.0"

    # v0.2.0 is retired
    r = client.get("/models/v0.2.0")
    assert r.json()["status"] == "retired"


def test_rollback_fails_if_already_production(client, tmp_workbench):
    report_path = _write_eval_report(tmp_workbench)
    client.post(
        "/models/register",
        json=_model_payload("v0.1.0", eval_report=report_path),
    )
    client.post("/models/v0.1.0/promote", json={"target_status": "production"})

    r = client.post("/models/v0.1.0/rollback")
    assert r.status_code == 400


def test_rollback_not_found(client):
    r = client.post("/models/v9.9.9/rollback")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# service-level tests
# ---------------------------------------------------------------------------


def test_service_register_and_get(conn):
    payload = _model_payload("v0.1.0")
    record = svc.register_model(conn, **payload)
    assert record["version"] == "v0.1.0"
    assert record["status"] == "candidate"
    assert record["manifest"]["gguf_path"].endswith("model-Q5_K_M.gguf")

    fetched = svc.get_model(conn, "v0.1.0")
    assert fetched is not None
    assert fetched["gguf_sha256"] == payload["gguf_sha256"]


def test_service_list_models_filter(conn):
    svc.register_model(conn, **_model_payload("v0.1.0"))
    svc.register_model(conn, **_model_payload("v0.2.0"))

    all_models = svc.list_models(conn)
    assert len(all_models) == 2

    candidates = svc.list_models(conn, status_filter="candidate")
    assert len(candidates) == 2

    prod = svc.list_models(conn, status_filter="production")
    assert len(prod) == 0


def test_service_promote_gate_check_blocks(conn):
    svc.register_model(conn, **_model_payload("v0.1.0", eval_report=None))
    with pytest.raises(svc.GateCheckError):
        svc.promote_model(conn, "v0.1.0", target_status="production")


def test_service_rollback(conn):
    svc.register_model(conn, **_model_payload("v0.1.0"))
    svc.register_model(conn, **_model_payload("v0.2.0"))

    # Manually set v0.1.0 to production for this test
    conn.execute("UPDATE model_versions SET status='production' WHERE version='v0.1.0'")
    conn.commit()

    result = svc.rollback_to(conn, "v0.2.0")
    assert result["version"] == "v0.2.0"
    assert result["previous_production"] == "v0.1.0"

    current = svc.get_current_production(conn)
    assert current["version"] == "v0.2.0"
