"""GPU Runtime API contract: device detection and runtime catalog."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend.app.services.gpu_runtime import get_gpu_devices, get_gpu_runtimes


class FakeBackend:
    def __init__(
        self,
        available: bool,
        name: str = "",
        memory: tuple[int, int] | None = None,
        driver_version: str | None = None,
        runtime_driver: str | None = None,
    ):
        self.available = available
        self.name = name
        self.memory = memory
        self.driver_version = driver_version
        self.runtime_driver = runtime_driver

    def get_device_properties(self, index: int):
        return SimpleNamespace(
            name=self.name,
            driver_version=self.driver_version,
            platform_name=self.runtime_driver,
        )

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return int(self.available)

    def get_device_name(self, index: int) -> str:
        return self.name

    def mem_get_info(self, index: int) -> tuple[int, int]:
        if self.memory is None:
            raise RuntimeError("memory query unavailable")
        return self.memory


class FakeTorch:
    def __init__(self, *, xpu: FakeBackend, cuda: FakeBackend, hip: str | None = None):
        self.xpu = xpu
        self.cuda = cuda
        self.version = SimpleNamespace(hip=hip)


def _make_drm_card(
    root: Path,
    *,
    vendor: str = "0x8086",
    device: str = "0xe20b",
    name: str = "Intel Arc B580",
    driver: str = "xe",
    driver_version: str = "6.18.51",
) -> None:
    device_dir = root / "card0" / "device"
    (device_dir / "driver" / "module").mkdir(parents=True)
    (device_dir / "vendor").write_text(vendor, encoding="utf-8")
    (device_dir / "device").write_text(device, encoding="utf-8")
    (device_dir / "product_name").write_text(name, encoding="utf-8")
    (device_dir / "driver_name").write_text(driver, encoding="utf-8")
    (device_dir / "driver" / "module" / "version").write_text(driver_version, encoding="utf-8")


def test_intel_arc_device_is_ready_when_xpu_torch_sees_it(tmp_path):
    _make_drm_card(tmp_path)
    torch = FakeTorch(
        xpu=FakeBackend(
            True, "Intel(R) Arc(TM) B580 Graphics", (10_000_000_000, 12_000_000_000),
            driver_version="1.17.39395+13", runtime_driver="Intel oneAPI Level Zero",
        ),
        cuda=FakeBackend(False),
    )

    devices = get_gpu_devices(drm_sysfs_root=tmp_path, torch_module=torch)

    assert devices == [{
        "id": "gpu-0",
        "vendor": "intel",
        "name": "Intel(R) Arc(TM) B580 Graphics",
        "driver_name": "xe",
        "driver_version": "1.17.39395+13",
        "runtime_driver": "Intel oneAPI Level Zero",
        "memory_total_mb": 11444,
        "memory_free_mb": 9536,
        "runtime": "xpu",
        "status": "ready",
        "error": None,
    }]


def test_physical_gpu_without_torch_runtime_is_not_reported_ready(tmp_path):
    _make_drm_card(tmp_path)
    torch = FakeTorch(xpu=FakeBackend(False), cuda=FakeBackend(False))

    devices = get_gpu_devices(drm_sysfs_root=tmp_path, torch_module=torch)

    assert len(devices) == 1
    assert devices[0]["vendor"] == "intel"
    assert devices[0]["runtime"] == "xpu"
    assert devices[0]["status"] == "runtime_unavailable"
    assert devices[0]["memory_total_mb"] is None


def test_gpu_runtime_catalog_marks_only_intel_xpu_as_enabled():
    devices = [{"vendor": "intel", "runtime": "xpu", "status": "ready"}]

    runtimes = get_gpu_runtimes(devices=devices)
    by_id = {runtime["id"]: runtime for runtime in runtimes}

    assert by_id["intel-xpu"]["supported"] is True
    assert by_id["intel-xpu"]["ready"] is True
    assert by_id["intel-xpu"]["image"] == "intel/pytorch:xpu-2.13.0-ubuntu24.04"
    assert by_id["nvidia-cuda"]["supported"] is False
    assert by_id["amd-rocm"]["supported"] is False
    assert by_id["cpu"]["supported"] is False
    assert by_id["cpu"]["ready"] is False


def test_runtime_device_is_returned_when_drm_sysfs_is_unavailable(tmp_path):
    torch = FakeTorch(
        xpu=FakeBackend(True, "Intel XPU 0", (2_000_000_000, 4_000_000_000)),
        cuda=FakeBackend(False),
    )

    devices = get_gpu_devices(drm_sysfs_root=tmp_path, torch_module=torch)

    assert len(devices) == 1
    assert devices[0]["vendor"] == "intel"
    assert devices[0]["runtime"] == "xpu"
    assert devices[0]["status"] == "ready"
    assert devices[0]["driver_name"] is None


def test_torch_visible_nvidia_device_is_not_training_ready_until_profile_exists(tmp_path):
    _make_drm_card(
        tmp_path, vendor="0x10de", device="0x2684", name="NVIDIA GPU",
        driver="nvidia", driver_version="555.1",
    )
    torch = FakeTorch(
        xpu=FakeBackend(False),
        cuda=FakeBackend(True, "NVIDIA Test GPU", (8_000_000_000, 10_000_000_000)),
    )

    devices = get_gpu_devices(drm_sysfs_root=tmp_path, torch_module=torch)
    runtimes = {runtime["id"]: runtime for runtime in get_gpu_runtimes(devices=devices)}

    assert devices[0]["runtime"] == "cuda"
    assert devices[0]["status"] == "unsupported_runtime"
    assert runtimes["nvidia-cuda"]["supported"] is False
    assert runtimes["nvidia-cuda"]["ready"] is False
    assert runtimes["nvidia-cuda"]["status"] == "detected_unsupported"
