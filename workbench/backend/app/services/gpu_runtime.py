"""Host GPU detection and the supported runtime catalog.

The first executable container profile targets Intel Arc/Battlemage XPU.
NVIDIA and AMD entries are intentionally marked as planned until their images
and device-specific smoke tests are maintained in this repository.
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any

DRM_SYSFS_ROOT = Path("/sys/class/drm")
IMPLEMENTED_RUNTIMES = {"intel": "xpu"}
INTEL_XPU_IMAGE = os.environ.get(
    "WORKBENCH_INTEL_XPU_IMAGE",
    "intel/pytorch:xpu-2.13.0-ubuntu24.04",
)

_VENDOR_BY_PCI_ID = {
    "0x8086": ("intel", "xpu"),
    "0x10de": ("nvidia", "cuda"),
    "0x1002": ("amd", "rocm"),
}
_VENDOR_NAMES = {"intel": "Intel", "nvidia": "NVIDIA", "amd": "AMD"}


def _read_text(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _module_driver_name(device_dir: Path) -> str | None:
    explicit = _read_text(device_dir / "driver_name")
    if explicit:
        return explicit
    driver_path = device_dir / "driver"
    try:
        return driver_path.resolve(strict=True).name
    except OSError:
        return None


def _physical_devices(drm_sysfs_root: Path) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    try:
        cards = sorted(
            (entry for entry in drm_sysfs_root.glob("card*") if re.fullmatch(r"card\d+", entry.name)),
            key=lambda entry: int(entry.name[4:]),
        )
    except OSError:
        return devices

    for card in cards:
        device_dir = card / "device"
        pci_vendor = (_read_text(device_dir / "vendor") or "").lower()
        pci_device = (_read_text(device_dir / "device") or "").lower()
        vendor_runtime = _VENDOR_BY_PCI_ID.get(pci_vendor)
        if vendor_runtime is None:
            continue
        vendor, runtime = vendor_runtime
        name = _read_text(device_dir / "product_name")
        if not name:
            name = f"{_VENDOR_NAMES[vendor]} GPU ({pci_device or 'unknown PCI device'})"
        devices.append({
            "vendor": vendor,
            "runtime": runtime,
            "name": name,
            "driver_name": _module_driver_name(device_dir),
            "driver_version": _read_text(device_dir / "driver" / "module" / "version"),
        })
    return devices


def _memory_mb(backend: Any, index: int) -> tuple[int | None, int | None]:
    try:
        free_bytes, total_bytes = backend.mem_get_info(index)
    except (AttributeError, OSError, RuntimeError, TypeError):
        return None, None
    mib = 1024 * 1024
    return int(total_bytes // mib), int(free_bytes // mib)


def _torch_devices(torch_module: Any | None) -> tuple[list[dict[str, Any]], str | None]:
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except ImportError:
            return [], "PyTorch is not installed in this runtime"
        except Exception as exc:
            return [], f"PyTorch could not initialize its GPU runtime: {exc}"

    detected: list[dict[str, Any]] = []
    for backend_name in ("xpu", "cuda"):
        backend = getattr(torch_module, backend_name, None)
        if backend is None:
            continue
        try:
            available = bool(backend.is_available())
        except Exception:
            available = False
        if not available:
            continue
        try:
            count = max(1, int(backend.device_count()))
        except Exception:
            count = 1

        hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
        vendor, runtime = (("amd", "rocm") if backend_name == "cuda" and hip_version else
                           ("nvidia", "cuda") if backend_name == "cuda" else
                           ("intel", "xpu"))
        for index in range(count):
            try:
                properties = backend.get_device_properties(index)
            except Exception:
                properties = None
            try:
                name = str(getattr(properties, "name", None) or backend.get_device_name(index))
            except Exception:
                name = f"{_VENDOR_NAMES[vendor]} GPU {index}"
            total_mb, free_mb = _memory_mb(backend, index)
            if total_mb is None:
                total_bytes = getattr(properties, "total_memory", None)
                if isinstance(total_bytes, int) and total_bytes > 0:
                    total_mb = total_bytes // (1024 * 1024)
            detected.append({
                "vendor": vendor,
                "runtime": runtime,
                "_index": index,
                "name": name,
                "runtime_driver": getattr(properties, "platform_name", None),
                "driver_version": getattr(properties, "driver_version", None),
                "memory_total_mb": total_mb,
                "memory_free_mb": free_mb,
            })

    return detected, None


def get_gpu_devices(
    *,
    drm_sysfs_root: Path = DRM_SYSFS_ROOT,
    torch_module: Any | None = None,
) -> list[dict[str, Any]]:
    """Return physical GPU devices and whether this process can use them.

    ``drm_sysfs_root`` and ``torch_module`` are injectable system seams for
    tests. In production they default to Linux DRM sysfs and installed PyTorch.
    A GPU is ``ready`` only when the matching PyTorch backend sees it.
    """
    physical = _physical_devices(Path(drm_sysfs_root))
    runtime_devices, torch_error = _torch_devices(torch_module)
    runtime_indices: dict[str, int] = {}
    result: list[dict[str, Any]] = []

    def append_device(
        vendor: str,
        runtime: str,
        name: str,
        driver_name: str | None,
        driver_version: str | None,
        runtime_device: dict[str, Any] | None,
    ) -> None:
        torch_visible = runtime_device is not None
        ready = torch_visible and IMPLEMENTED_RUNTIMES.get(vendor) == runtime
        if ready:
            status = "ready"
            error = None
        elif torch_visible:
            status = "unsupported_runtime"
            error = f"The {vendor}/{runtime} runtime profile is not implemented for training yet"
        else:
            status = "runtime_unavailable"
            error = torch_error or f"PyTorch {runtime} device is unavailable"
        result.append({
            "id": f"gpu-{len(result)}",
            "vendor": vendor,
            "name": runtime_device["name"] if torch_visible else name,
            "driver_name": driver_name,
            "driver_version": (runtime_device.get("driver_version") or driver_version) if torch_visible else driver_version,
            "runtime_driver": runtime_device.get("runtime_driver") if torch_visible else None,
            "memory_total_mb": runtime_device.get("memory_total_mb") if torch_visible else None,
            "memory_free_mb": runtime_device.get("memory_free_mb") if torch_visible else None,
            "runtime": runtime,
            "status": status,
            "error": error,
        })

    for device in physical:
        vendor = device["vendor"]
        index = runtime_indices.get(vendor, 0)
        runtime_indices[vendor] = index + 1
        match = next(
            (item for item in runtime_devices
             if item["vendor"] == vendor and item.get("_index", 0) == index),
            None,
        )
        # The runtime enumerator is already ordered per vendor; attach indices
        # here so multiple cards cannot accidentally reuse device zero.
        if match is None:
            matching = [item for item in runtime_devices if item["vendor"] == vendor]
            match = matching[index] if index < len(matching) else None
        append_device(
            vendor, device["runtime"], device["name"], device["driver_name"],
            device["driver_version"], match,
        )

    # Some containers expose /dev/dri and PyTorch XPU but not the corresponding
    # /sys/class/drm entry. Keep runtime-visible devices in the response.
    for runtime_device in runtime_devices:
        vendor = runtime_device["vendor"]
        seen = sum(1 for item in result if item["vendor"] == vendor)
        vendor_runtime_count = sum(1 for item in runtime_devices if item["vendor"] == vendor)
        if seen >= vendor_runtime_count:
            continue
        append_device(
            vendor, runtime_device["runtime"], runtime_device["name"], None, None,
            runtime_device,
        )

    return result


def get_gpu_runtimes(*, devices: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return the single-source runtime catalog and host readiness per profile."""
    if devices is None:
        devices = get_gpu_devices()

    registry = [
        {
            "id": "intel-xpu",
            "vendor": "intel",
            "runtime": "xpu",
            "supported": True,
            "image": INTEL_XPU_IMAGE,
            "device_nodes": ["/dev/dri/renderD*"],
        },
        {
            "id": "nvidia-cuda",
            "vendor": "nvidia",
            "runtime": "cuda",
            "supported": False,
            "image": None,
            "device_nodes": ["/dev/nvidia*"],
        },
        {
            "id": "amd-rocm",
            "vendor": "amd",
            "runtime": "rocm",
            "supported": False,
            "image": None,
            "device_nodes": ["/dev/kfd", "/dev/dri/renderD*"],
        },
        {
            "id": "cpu",
            "vendor": "cpu",
            "runtime": "cpu",
            "supported": False,
            "image": None,
            "device_nodes": [],
        },
    ]

    for runtime in registry:
        vendor_devices = [device for device in devices if device.get("vendor") == runtime["vendor"]]
        runtime["detected"] = bool(vendor_devices)
        runtime["ready"] = any(
            device.get("status") == "ready" and device.get("runtime") == runtime["runtime"]
            for device in vendor_devices
        )
        if runtime["ready"]:
            runtime["status"] = "ready"
        elif not runtime["supported"]:
            runtime["status"] = "detected_unsupported" if runtime["detected"] else "planned"
        elif runtime["detected"]:
            runtime["status"] = "detected_runtime_unavailable"
        else:
            runtime["status"] = "not_detected"
    return registry
