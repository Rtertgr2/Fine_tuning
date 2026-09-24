"""GPU Runtime read-only API for device discovery and runtime selection."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.app.services.gpu_runtime import get_gpu_devices, get_gpu_runtimes

router = APIRouter(tags=["gpu"])


@router.get("/gpu/devices", response_model=dict[str, Any])
def list_gpu_devices() -> dict[str, Any]:
    """Return host GPUs and whether the active PyTorch runtime can use them."""
    return {"devices": get_gpu_devices()}


@router.get("/gpu/runtimes", response_model=dict[str, Any])
def list_gpu_runtimes() -> dict[str, Any]:
    """Return supported runtime profiles and current host readiness."""
    devices = get_gpu_devices()
    return {"runtimes": get_gpu_runtimes(devices=devices)}
