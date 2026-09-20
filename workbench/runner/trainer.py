"""Training loop with metrics logging and overfit detection (T3.4, T3.5)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.services.training_config import TrainingConfig
from runner.metrics import MetricsTracker


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl_entry(path: Path, entry: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _capture_env(output_dir: Path) -> str:
    """PF6: Capture pip freeze, GPU info, driver version."""
    env_path = output_dir / "env.txt"
    lines = []

    # pip freeze
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            lines.append("=== pip freeze ===")
            lines.append(result.stdout.strip())
    except Exception:
        lines.append("=== pip freeze: unavailable ===")

    # GPU info
    try:
        import torch
        lines.append("\n=== GPU ===")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"Device: {torch.cuda.get_device_name(0)}")
            lines.append(f"VRAM total: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    except ImportError:
        lines.append("\n=== GPU: torch not installed ===")

    content = "\n".join(lines)
    env_path.write_text(content, encoding="utf-8")
    return content


def _copy_dataset_manifest(cfg: TrainingConfig, output_dir: Path) -> None:
    """Copy dataset manifest to run output."""
    from backend.app.services import datasets as ds_service
    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    manifest_path = ds_dir / "manifest.json"
    if manifest_path.exists():
        import shutil
        shutil.copy2(manifest_path, output_dir / "dataset_manifest.json")


def _detect_overfit(metrics: MetricsTracker, patience: int = 3) -> tuple[bool, str]:
    """Detect overfitting: eval loss increases N times while train loss decreases.

    Returns (is_overfitting, message).
    """
    eval_entries = [e for e in metrics.entries if e.eval_loss is not None]
    if len(eval_entries) < patience + 1:
        return False, ""

    # Check last `patience` eval points
    recent = eval_entries[-(patience + 1):]
    eval_increases = 0
    train_decreases = 0

    for i in range(1, len(recent)):
        prev_eval = recent[i - 1].eval_loss
        curr_eval = recent[i].eval_loss
        if prev_eval is not None and curr_eval is not None and curr_eval > prev_eval:
            eval_increases += 1

        # Find closest train loss entries
        prev_step = recent[i - 1].step
        curr_step = recent[i].step
        prev_train = [e for e in metrics.entries if e.step == prev_step and e.train_loss is not None]
        curr_train = [e for e in metrics.entries if e.step == curr_step and e.train_loss is not None]

        if prev_train and curr_train:
            if curr_train[0].train_loss < prev_train[0].train_loss:
                train_decreases += 1

    if eval_increases >= patience and train_decreases >= patience:
        return True, (
            f"Overfit detected: eval loss increased {eval_increases} times "
            f"while train loss decreased {train_decreases} times in last {patience} evaluations"
        )
    return False, ""


def _generate_report(cfg: TrainingConfig, output_dir: Path, metrics: MetricsTracker,
                     status: str, message: str = "") -> str:
    """Generate report.md for the run."""
    report_path = output_dir / "report.md"
    best_eval = metrics.best_eval_loss
    best_step = metrics.best_step
    latest = metrics.get_latest()

    lines = [
        f"# Training Report: {cfg.run_name}",
        "",
        f"**Status:** {status}",
        f"**Base Model:** {cfg.base_model}",
        f"**Dataset:** {cfg.dataset_version}",
        f"**Seed:** {cfg.seed}",
        "",
        "## Configuration",
        "",
        "```yaml",
        f"run_name: {cfg.run_name}",
        f"base_model: {cfg.base_model}",
        f"dataset_version: {cfg.dataset_version}",
        f"seed: {cfg.seed}",
        f"loss_masking: {cfg.loss_masking}",
        "",
        "quantization:",
        f"  load_in_4bit: {cfg.quantization.load_in_4bit}",
        f"  quant_type: {cfg.quantization.quant_type}",
        f"  double_quant: {cfg.quantization.double_quant}",
        "",
        "lora:",
        f"  r: {cfg.lora.r}",
        f"  alpha: {cfg.lora.alpha}",
        f"  dropout: {cfg.lora.dropout}",
        f"  target_modules: {cfg.lora.target_modules}",
        "",
        "train:",
        f"  epochs: {cfg.train.epochs}",
        f"  learning_rate: {cfg.train.learning_rate}",
        f"  lr_scheduler: {cfg.train.lr_scheduler}",
        f"  warmup_ratio: {cfg.train.warmup_ratio}",
        f"  per_device_batch_size: {cfg.train.per_device_batch_size}",
        f"  grad_accum: {cfg.train.grad_accum}",
        f"  max_seq_length: {cfg.train.max_seq_length}",
        f"  eval_steps: {cfg.train.eval_steps}",
        f"  save_steps: {cfg.train.save_steps}",
        f"  early_stopping_patience: {cfg.train.early_stopping_patience}",
        "```",
        "",
        "## Results",
        "",
    ]

    if latest:
        lines.append(f"- **Final step:** {latest.step}")
        if latest.train_loss is not None:
            lines.append(f"- **Final train loss:** {latest.train_loss:.4f}")
        if latest.eval_loss is not None:
            lines.append(f"- **Final eval loss:** {latest.eval_loss:.4f}")
    if best_eval is not None:
        lines.append(f"- **Best eval loss:** {best_eval:.4f} (step {best_step})")
    if latest and latest.elapsed_seconds:
        lines.append(f"- **Total time:** {latest.elapsed_seconds:.0f}s")

    if message:
        lines.extend(["", f"**Note:** {message}"])

    lines.extend([
        "",
        "## Files",
        "",
        "- `config.yaml` — training configuration",
        "- `dataset_manifest.json` — dataset version and hashes",
        "- `env.txt` — pip freeze, GPU info",
        "- `metrics.jsonl` — training metrics",
        "- `adapter/` — LoRA adapter weights",
        "- `best_checkpoint.txt` — path to best checkpoint",
    ])

    content = "\n".join(lines)
    report_path.write_text(content, encoding="utf-8")
    return content


class TrainingRunner:
    """Orchestrates a training run with metrics tracking and overfit detection."""

    def __init__(self, cfg: TrainingConfig, output_dir: str | Path):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsTracker(self.output_dir / "metrics.jsonl")
        self.status: str = "pending"
        self.message: str = ""
        self._process: subprocess.Popen | None = None

    def setup(self) -> None:
        """Prepare the run directory with config, manifest, env."""
        # Save config
        self.cfg.save_yaml(self.output_dir / "config.yaml")

        # Copy dataset manifest
        _copy_dataset_manifest(self.cfg, self.output_dir)

        # Capture environment
        _capture_env(self.output_dir)

    def start(self) -> None:
        """Start training (runs the generated script)."""
        self.status = "running"
        script_path = self.output_dir / "train.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"Training script not found at {script_path}. "
                "Generate it first with scripts/generate_training_script.py"
            )

        self._process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(self.output_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        """Stop the training process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self.status = "stopped"

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def check_overfit(self) -> tuple[bool, str]:
        """Check for overfitting based on current metrics."""
        return _detect_overfit(self.metrics, patience=self.cfg.train.early_stopping_patience)

    def finalize(self, status: str = "completed", message: str = "") -> None:
        """Finalize the run: write best_checkpoint.txt and report.md."""
        self.status = status
        self.message = message

        best_step = self.metrics.best_step
        if best_step is not None:
            checkpoint_dir = self.output_dir / f"checkpoint-{best_step}"
            if checkpoint_dir.exists():
                (self.output_dir / "best_checkpoint.txt").write_text(
                    str(checkpoint_dir), encoding="utf-8"
                )

        _generate_report(self.cfg, self.output_dir, self.metrics, status, message)

    def to_dict(self) -> dict[str, Any]:
        latest = self.metrics.get_latest()
        return {
            "run_name": self.cfg.run_name,
            "status": self.status,
            "message": self.message,
            "latest_step": latest.step if latest else None,
            "best_eval_loss": self.metrics.best_eval_loss,
            "best_step": self.metrics.best_step,
            "config": self.cfg.to_dict(),
        }
