# Use Intel's maintained PyTorch/XPU image; host supplies the kernel driver.
ARG INTEL_PYTORCH_IMAGE=intel/pytorch:xpu-2.13.0-ubuntu24.04
FROM ${INTEL_PYTORCH_IMAGE}

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY containers/intel/requirements-gpu-xpu.txt /tmp/requirements-gpu-xpu.txt
# torch is intentionally not listed here: keep Intel's XPU-enabled build.
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-gpu-xpu.txt \
    && python3 -c "import torch; assert hasattr(torch, 'xpu'), 'Intel XPU-enabled PyTorch is required'"

COPY . /app

EXPOSE 8300
CMD ["python3", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8300"]
