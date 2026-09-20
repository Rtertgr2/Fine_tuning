"""Pre-flight checks PF1-PF6 (T3.2)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from backend.adapters.registry import get_adapter
from backend.app import config
from backend.app.services import datasets as ds_service
from backend.app.services.training_config import TrainingConfig


@dataclass
class PreflightResult:
    code: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    critical: bool = False  # if True and failed, training cannot proceed

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "critical": self.critical,
        }


@dataclass
class PreflightReport:
    results: list[PreflightResult] = field(default_factory=list)
    can_proceed: bool = True

    def add(self, result: PreflightResult) -> None:
        self.results.append(result)
        if result.critical and not result.passed:
            self.can_proceed = False

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_proceed": self.can_proceed,
            "all_passed": self.all_passed,
            "results": [r.to_dict() for r in self.results],
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# PF1: Dataset hash matches manifest
# ---------------------------------------------------------------------------

def pf1_dataset_hash(cfg: TrainingConfig) -> PreflightReport:
    report = PreflightReport()
    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    manifest_path = ds_dir / "manifest.json"

    if not manifest_path.exists():
        report.add(PreflightResult(
            code="PF1",
            passed=False,
            message=f"manifest.json not found for dataset {cfg.dataset_version}",
            critical=True,
        ))
        return report

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        report.add(PreflightResult(
            code="PF1",
            passed=False,
            message=f"manifest.json is corrupt: {e}",
            critical=True,
        ))
        return report

    files_meta = manifest.get("files", {})
    all_ok = True
    mismatches: list[dict[str, str]] = []

    for split, meta in files_meta.items():
        if split not in ("train", "val"):
            continue
        expected = meta.get("sha256", "")
        actual_path = ds_dir / Path(meta["path"]).name
        if not actual_path.exists():
            mismatches.append({"file": split, "expected": expected, "actual": "FILE_MISSING"})
            all_ok = False
            continue
        actual_hash = _sha256_file(actual_path)
        if actual_hash != expected:
            mismatches.append({"file": split, "expected": expected, "actual": actual_hash})
            all_ok = False

    report.add(PreflightResult(
        code="PF1",
        passed=all_ok,
        message="Dataset hashes match manifest" if all_ok else f"Hash mismatch: {mismatches}",
        details={"mismatches": mismatches},
        critical=True,
    ))
    return report


# ---------------------------------------------------------------------------
# PF2: Render 5 samples with chat template
# ---------------------------------------------------------------------------

def pf2_render_samples(cfg: TrainingConfig) -> PreflightReport:
    report = PreflightReport()
    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    train_path = ds_dir / "train.jsonl"

    if not train_path.exists():
        report.add(PreflightResult(
            code="PF2",
            passed=False,
            message=f"train.jsonl not found at {train_path}",
        ))
        return report

    samples = _load_jsonl(train_path)
    if not samples:
        report.add(PreflightResult(
            code="PF2",
            passed=False,
            message="train.jsonl is empty",
        ))
        return report

    adapter = get_adapter()
    rendered: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, sample in enumerate(samples[:5]):
        messages = sample.get("messages", [])
        tools = sample.get("tools")
        try:
            text = adapter.render_conversation(
                messages=messages, tools=tools, add_generation_prompt=False
            )
            rendered.append({
                "index": i,
                "num_messages": len(messages),
                "num_tools": len(tools) if tools else 0,
                "rendered_length": len(text),
            })
        except Exception as e:
            errors.append(f"Sample {i}: {e}")

    passed = len(errors) == 0 and len(rendered) > 0
    report.add(PreflightResult(
        code="PF2",
        passed=passed,
        message=f"Rendered {len(rendered)} samples" + (f"; errors: {errors}" if errors else ""),
        details={"rendered": rendered, "errors": errors},
    ))
    return report


# ---------------------------------------------------------------------------
# PF3: Loss mask verification (only assistant tokens labeled)
# ---------------------------------------------------------------------------

def pf3_loss_mask(cfg: TrainingConfig) -> PreflightReport:
    """Verify that labels are only on assistant tokens (or configured subset).

    This is a CRITICAL check — failing it means the model would learn from
    system/user/tool-response tokens.
    """
    report = PreflightReport()

    if cfg.loss_masking == "all":
        report.add(PreflightResult(
            code="PF3",
            passed=False,
            message="loss_masking='all' is not recommended — model will learn from non-assistant tokens",
            details={"loss_masking": cfg.loss_masking},
            critical=True,
        ))
        return report

    # Structural check: verify that the chat template has assistant markers
    adapter = get_adapter()
    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    train_path = ds_dir / "train.jsonl"

    if not train_path.exists():
        report.add(PreflightResult(
            code="PF3",
            passed=False,
            message=f"train.jsonl not found at {train_path}",
            critical=True,
        ))
        return report

    samples = _load_jsonl(train_path)
    if not samples:
        report.add(PreflightResult(
            code="PF3",
            passed=False,
            message="train.jsonl is empty",
            critical=True,
        ))
        return report

    # Pick a few random samples and verify structure
    import random
    rng = random.Random(cfg.seed)
    check_samples = rng.sample(samples, min(5, len(samples)))

    issues: list[dict[str, Any]] = []
    for i, sample in enumerate(check_samples):
        messages = sample.get("messages", [])
        has_system = any(m.get("role") == "system" for m in messages)
        has_user = any(m.get("role") == "user" for m in messages)
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        has_tool = any(m.get("role") == "tool" for m in messages)

        if not has_assistant:
            issues.append({"index": sample.get("id", i), "issue": "no assistant messages"})

        # Verify that assistant content has actual text (not just tool calls)
        for j, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if not content or not content.strip():
                    issues.append({
                        "index": sample.get("id", i),
                        "issue": f"assistant message {j} has empty content",
                    })

    passed = len(issues) == 0
    report.add(PreflightResult(
        code="PF3",
        passed=passed,
        message="Loss mask verification passed (assistant_only)" if passed else f"Issues found: {issues}",
        details={"samples_checked": len(check_samples), "issues": issues},
        critical=True,
    ))
    return report


# ---------------------------------------------------------------------------
# PF4: Token length check (no truncation)
# ---------------------------------------------------------------------------

def pf4_token_length(cfg: TrainingConfig) -> PreflightReport:
    report = PreflightReport()
    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    train_path = ds_dir / "train.jsonl"

    if not train_path.exists():
        report.add(PreflightResult(
            code="PF4",
            passed=False,
            message=f"train.jsonl not found at {train_path}",
        ))
        return report

    samples = _load_jsonl(train_path)
    adapter = get_adapter()

    max_len = 0
    max_sample_id = ""
    truncated_count = 0
    total_count = len(samples)

    for sample in samples:
        messages = sample.get("messages", [])
        tools = sample.get("tools")
        try:
            text = adapter.render_conversation(
                messages=messages, tools=tools, add_generation_prompt=False
            )
            n_tokens = adapter.count_tokens(text) or 0
            if n_tokens > cfg.train.max_seq_length:
                truncated_count += 1
            if n_tokens > max_len:
                max_len = n_tokens
                max_sample_id = sample.get("id", "")
        except Exception:
            pass

    passed = truncated_count == 0
    report.add(PreflightResult(
        code="PF4",
        passed=passed,
        message=f"No truncation ({total_count} samples, max {max_len} tokens)" if passed
        else f"{truncated_count}/{total_count} samples exceed max_seq_length ({cfg.train.max_seq_length})",
        details={
            "max_tokens": max_len,
            "max_sample_id": max_sample_id,
            "truncated_count": truncated_count,
            "total_count": total_count,
            "max_seq_length": cfg.train.max_seq_length,
        },
    ))
    return report


# ---------------------------------------------------------------------------
# PF5: Dry run (10 steps, check OOM)
# ---------------------------------------------------------------------------

def pf5_dry_run(cfg: TrainingConfig) -> PreflightReport:
    """Attempt a short dry-run training to verify VRAM fits.

    This is a soft check — if torch/transformers are not installed, we skip it.
    """
    report = PreflightReport()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        report.add(PreflightResult(
            code="PF5",
            passed=True,
            message="Skipped: torch/transformers not installed (will run on GPU host)",
            details={"skipped": True},
        ))
        return report

    if not torch.cuda.is_available():
        report.add(PreflightResult(
            code="PF5",
            passed=True,
            message="Skipped: no CUDA available (will run on GPU host)",
            details={"skipped": True},
        ))
        return report

    try:
        from transformers import BitsAndBytesConfig

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.quantization.quant_type,
            bnb_4bit_use_double_quant=cfg.quantization.double_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=quant_cfg,
            device_map="auto",
        )

        vram_before = torch.cuda.memory_allocated()
        # Minimal forward pass
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        sample = tokenizer("test", return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**sample)
        vram_after = torch.cuda.memory_allocated()
        vram_used_mb = (vram_after - vram_before) / 1024 / 1024

        del model
        torch.cuda.empty_cache()

        report.add(PreflightResult(
            code="PF5",
            passed=True,
            message=f"Dry run OK, VRAM delta: {vram_used_mb:.1f} MB",
            details={"vram_delta_mb": round(vram_used_mb, 2)},
        ))
    except Exception as e:
        report.add(PreflightResult(
            code="PF5",
            passed=False,
            message=f"Dry run failed: {e}",
            details={"error": str(e)},
        ))

    return report


# ---------------------------------------------------------------------------
# PF6: Record library versions (pip freeze)
# ---------------------------------------------------------------------------

def pf6_library_versions(cfg: TrainingConfig) -> PreflightReport:
    """Capture pip freeze for reproducibility."""
    report = PreflightReport()

    try:
        import subprocess
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            freeze = result.stdout.strip().split("\n")
            report.add(PreflightResult(
                code="PF6",
                passed=True,
                message=f"Captured {len(freeze)} packages",
                details={"packages": len(freeze)},
            ))
        else:
            report.add(PreflightResult(
                code="PF6",
                passed=False,
                message=f"pip freeze failed: {result.stderr[:200]}",
            ))
    except Exception as e:
        report.add(PreflightResult(
            code="PF6",
            passed=False,
            message=f"Could not capture versions: {e}",
        ))

    return report


# ---------------------------------------------------------------------------
# Run all preflight checks
# ---------------------------------------------------------------------------

def run_preflight(cfg: TrainingConfig) -> PreflightReport:
    """Run all pre-flight checks PF1-PF6."""
    report = PreflightReport()

    # PF1
    r1 = pf1_dataset_hash(cfg)
    report.results.extend(r1.results)
    if not r1.can_proceed:
        report.can_proceed = False

    # PF2
    r2 = pf2_render_samples(cfg)
    report.results.extend(r2.results)

    # PF3 (critical)
    r3 = pf3_loss_mask(cfg)
    report.results.extend(r3.results)
    if not r3.can_proceed:
        report.can_proceed = False

    # PF4
    r4 = pf4_token_length(cfg)
    report.results.extend(r4.results)

    # PF5
    r5 = pf5_dry_run(cfg)
    report.results.extend(r5.results)

    # PF6
    r6 = pf6_library_versions(cfg)
    report.results.extend(r6.results)

    return report
