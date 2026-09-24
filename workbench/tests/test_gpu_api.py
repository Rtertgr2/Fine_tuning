"""GPU endpoint contract tests without requiring local GPU hardware."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_gpu_endpoints_expose_detected_device_and_runtime_catalog(monkeypatch, tmp_workbench):
    from backend.app.main import app
    from backend.app.routers import gpu as gpu_router

    device = {
        "id": "gpu-0",
        "vendor": "intel",
        "name": "Intel Arc B580",
        "driver_name": "xe",
        "driver_version": "6.18.51",
        "memory_total_mb": 11444,
        "memory_free_mb": 10000,
        "runtime": "xpu",
        "status": "ready",
        "error": None,
    }
    monkeypatch.setattr(gpu_router, "get_gpu_devices", lambda: [device])

    with TestClient(app) as client:
        devices_response = client.get("/gpu/devices")
        runtimes_response = client.get("/gpu/runtimes")

    assert devices_response.status_code == 200
    assert devices_response.json() == {"devices": [device]}
    assert runtimes_response.status_code == 200
    runtimes = {item["id"]: item for item in runtimes_response.json()["runtimes"]}
    assert runtimes["intel-xpu"]["ready"] is True
    assert runtimes["intel-xpu"]["image"] == "intel/pytorch:xpu-2.13.0-ubuntu24.04"
    assert runtimes["nvidia-cuda"]["status"] == "planned"
    assert runtimes["amd-rocm"]["status"] == "planned"


def test_gpu_runtime_endpoint_keeps_cpu_available_without_gpu(monkeypatch, tmp_workbench):
    from backend.app.main import app
    from backend.app.routers import gpu as gpu_router

    monkeypatch.setattr(gpu_router, "get_gpu_devices", lambda: [])

    with TestClient(app) as client:
        response = client.get("/gpu/runtimes")

    assert response.status_code == 200
    runtimes = {item["id"]: item for item in response.json()["runtimes"]}
    assert runtimes["cpu"]["supported"] is False
    assert runtimes["cpu"]["ready"] is False
    assert runtimes["intel-xpu"]["ready"] is False
