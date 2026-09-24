"""GPU-specific preflight gates are device-agnostic and fail closed."""
from __future__ import annotations

from backend.app.services.preflight import (
    pf0_gpu_detection,
    pf7_gpu_device_access,
    pf8_framework_compatibility,
)
from backend.app.services.training_config import TrainingConfig


def test_pf0_passes_when_a_pytorch_gpu_runtime_is_ready():
    devices = [{
        "id": "gpu-0",
        "vendor": "intel",
        "name": "Intel Arc B580",
        "runtime": "xpu",
        "status": "ready",
        "memory_total_mb": 11444,
        "memory_free_mb": 10000,
    }]

    result = pf0_gpu_detection(TrainingConfig(), devices=devices).results[0]

    assert result.code == "PF0"
    assert result.passed is True
    assert result.critical is True
    assert result.details["runtime"] == "xpu"


def test_pf0_rejects_physical_gpu_without_usable_runtime():
    devices = [{
        "id": "gpu-0",
        "vendor": "intel",
        "name": "Intel Arc B580",
        "runtime": "xpu",
        "status": "runtime_unavailable",
        "memory_total_mb": None,
        "memory_free_mb": None,
    }]

    result = pf0_gpu_detection(TrainingConfig(), devices=devices).results[0]

    assert result.code == "PF0"
    assert result.passed is False
    assert result.critical is True
    assert "PyTorch" in result.message or "runtime" in result.message.lower()


def test_pf0_rejects_when_no_gpu_is_detected():
    result = pf0_gpu_detection(TrainingConfig(), devices=[]).results[0]

    assert result.code == "PF0"
    assert result.passed is False
    assert result.critical is True
    assert result.details["devices"] == []


def test_pf7_requires_accessible_dri_render_node(tmp_path):
    device = {
        "id": "gpu-0", "vendor": "intel", "name": "Intel Arc B580",
        "runtime": "xpu", "status": "ready",
    }
    render_node = tmp_path / "dri" / "renderD128"
    render_node.parent.mkdir()
    render_node.touch()

    result = pf7_gpu_device_access(
        TrainingConfig(), devices=[device], device_root=tmp_path,
    ).results[0]

    assert result.code == "PF7"
    assert result.passed is True
    assert result.details["accessible_nodes"] == [str(render_node)]


def test_pf7_fails_when_no_render_node_is_mapped(tmp_path):
    device = {
        "id": "gpu-0", "vendor": "intel", "name": "Intel Arc B580",
        "runtime": "xpu", "status": "ready",
    }

    result = pf7_gpu_device_access(
        TrainingConfig(), devices=[device], device_root=tmp_path,
    ).results[0]

    assert result.code == "PF7"
    assert result.passed is False
    assert result.critical is True


def test_pf8_fails_closed_when_gpu_is_not_ready():
    result = pf8_framework_compatibility(TrainingConfig(), devices=[]).results[0]

    assert result.code == "PF8"
    assert result.passed is False
    assert result.critical is True
    assert "GPU" in result.message
