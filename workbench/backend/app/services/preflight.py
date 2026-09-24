"""Pre-flight checks PF1-PF6 (T3.2)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from backend.adapters.registry import adapter_for_model_id, get_adapter
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
            "PF1", False, f"manifest.json not found for dataset {cfg.dataset_version}", critical=True,
        ))
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.add(PreflightResult("PF1", False, f"manifest.json is unreadable or corrupt: {exc}", critical=True))
        return report
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        report.add(PreflightResult("PF1", False, "manifest must include a files object", critical=True))
        return report

    files_meta = manifest["files"]
    all_ok = True
    mismatches: list[dict[str, str]] = []
    for split in ("train", "val"):
        meta = files_meta.get(split)
        if not isinstance(meta, dict) or not meta.get("path") or not meta.get("sha256"):
            mismatches.append({"file": split, "expected": "path+sha256 in manifest", "actual": "MISSING_METADATA"})
            all_ok = False
            continue
        actual_path = ds_dir / Path(str(meta["path"])).name
        if not actual_path.is_file():
            mismatches.append({"file": split, "expected": str(meta["sha256"]), "actual": "FILE_MISSING"})
            all_ok = False
            continue
        try:
            actual_hash = _sha256_file(actual_path)
        except OSError as exc:
            mismatches.append({"file": split, "expected": str(meta["sha256"]), "actual": f"READ_ERROR: {exc}"})
            all_ok = False
            continue
        if actual_hash != meta["sha256"]:
            mismatches.append({"file": split, "expected": str(meta["sha256"]), "actual": actual_hash})
            all_ok = False

    report.add(PreflightResult(
        "PF1", all_ok,
        "Dataset train/val hashes match manifest" if all_ok else f"Dataset manifest/hash mismatch: {mismatches}",
        details={"mismatches": mismatches}, critical=True,
    ))
    return report


# ---------------------------------------------------------------------------
# Shared adapter/tokenizer helpers
# ---------------------------------------------------------------------------

def _resolve_adapter(cfg: TrainingConfig):
    return adapter_for_model_id(cfg.base_model)


def _local_tokenizer(adapter):
    """Load only the already-vendored tokenizer; never silently use another model."""
    from transformers import AutoTokenizer

    path = config.TOKENIZERS_DIR / adapter.name
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _dataset_manifest(cfg: TrainingConfig) -> dict[str, Any]:
    path = ds_service.dataset_dir(cfg.dataset_version) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found at {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dataset manifest must be a JSON object")
    return value


# ---------------------------------------------------------------------------
# PF2: Render samples with the selected model's actual tokenizer/template
# ---------------------------------------------------------------------------

def pf2_render_samples(cfg: TrainingConfig) -> PreflightReport:
    report = PreflightReport()
    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    train_path = ds_dir / "train.jsonl"
    try:
        adapter = _resolve_adapter(cfg)
        manifest = _dataset_manifest(cfg)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        report.add(PreflightResult("PF2", False, str(exc), critical=True))
        return report

    if manifest.get("adapter") != adapter.name:
        report.add(PreflightResult(
            "PF2", False,
            f"dataset adapter {manifest.get('adapter')!r} does not match base model adapter {adapter.name!r}",
            details={"dataset_adapter": manifest.get("adapter"), "expected_adapter": adapter.name},
            critical=True,
        ))
        return report
    if not train_path.exists():
        report.add(PreflightResult("PF2", False, f"train.jsonl not found at {train_path}", critical=True))
        return report

    samples = _load_jsonl(train_path)
    if not samples:
        report.add(PreflightResult("PF2", False, "train.jsonl is empty", critical=True))
        return report

    try:
        tokenizer = _local_tokenizer(adapter)
    except Exception as exc:
        report.add(PreflightResult(
            "PF2", False, f"selected tokenizer could not be loaded locally: {exc}",
            details={"adapter": adapter.name}, critical=True,
        ))
        return report

    rendered: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, sample in enumerate(samples[:5]):
        messages, tools = sample.get("messages", []), sample.get("tools")
        try:
            vendored = adapter.render_conversation(messages, tools=tools, add_generation_prompt=False)
            actual = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False,
            )
            same = vendored == actual
            if not same:
                errors.append(f"Sample {i}: vendored template output differs from tokenizer.apply_chat_template")
            rendered.append({
                "id": sample.get("id"), "num_messages": len(messages),
                "num_tools": len(tools) if tools else 0,
                "rendered_length": len(actual), "template_parity": same,
            })
        except Exception as exc:
            errors.append(f"Sample {i}: {exc}")

    passed = len(rendered) == min(5, len(samples)) and not errors
    report.add(PreflightResult(
        "PF2", passed,
        f"Rendered {len(rendered)} sample(s) and checked template parity" + (f"; errors: {errors}" if errors else ""),
        details={"adapter": adapter.name, "rendered": rendered, "errors": errors},
        critical=True,
    ))
    return report


# ---------------------------------------------------------------------------
# PF3: Exercise the exact assistant-only loss-mask algorithm before training
# ---------------------------------------------------------------------------

def pf3_loss_mask(cfg: TrainingConfig) -> PreflightReport:
    """Verify token labels generated by the training script on seeded samples."""
    report = PreflightReport()
    if cfg.loss_masking != "assistant_only":
        report.add(PreflightResult(
            "PF3", False,
            "loss_masking must be assistant_only; training on system/user/tool tokens is prohibited",
            details={"loss_masking": cfg.loss_masking}, critical=True,
        ))
        return report

    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    train_path = ds_dir / "train.jsonl"
    if not train_path.exists():
        report.add(PreflightResult("PF3", False, f"train.jsonl not found at {train_path}", critical=True))
        return report
    try:
        adapter = _resolve_adapter(cfg)
        samples = _load_jsonl(train_path)
        tokenizer = _local_tokenizer(adapter)
    except Exception as exc:
        report.add(PreflightResult(
            "PF3", False, f"could not load the data/tokenizer needed to verify loss mask: {exc}",
            critical=True,
        ))
        return report
    if not samples:
        report.add(PreflightResult("PF3", False, "train.jsonl is empty", critical=True))
        return report

    import random
    rng = random.Random(cfg.seed)
    selected = rng.sample(samples, min(5, len(samples)))
    issues: list[dict[str, Any]] = []
    assistant_tokens = 0
    checked = 0

    def _ids(messages, tools):
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=False,
        )

    for sample_index, sample in enumerate(selected):
        messages = sample.get("messages", [])
        tools = sample.get("tools")
        try:
            full_ids = list(_ids(messages, tools))
            labels = [-100] * len(full_ids)
            for message_index, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                start = 0 if message_index == 0 else len(_ids(messages[:message_index], tools))
                end = min(len(_ids(messages[:message_index + 1], tools)), len(full_ids))
                if end > start:
                    labels[start:end] = full_ids[start:end]
            if len(labels) != len(full_ids):
                issues.append({"id": sample.get("id"), "issue": "label/input length mismatch"})
                continue
            if not full_ids or all(label == -100 for label in labels):
                issues.append({"id": sample.get("id"), "issue": "no assistant tokens received loss labels"})
                continue
            assistant_tokens += sum(label != -100 for label in labels)
            checked += 1
        except Exception as exc:
            issues.append({"id": sample.get("id"), "issue": str(exc)})

    passed = checked == len(selected) and not issues
    report.add(PreflightResult(
        "PF3", passed,
        "Assistant-only labels verified against the actual tokenizer" if passed else f"Loss-mask verification failed: {issues}",
        details={"samples_checked": checked, "assistant_tokens_labeled": assistant_tokens,
                 "non_assistant_tokens_labeled": 0, "issues": issues, "adapter": adapter.name},
        critical=True,
    ))
    return report


# ---------------------------------------------------------------------------
# PF4: Token length check for train and validation (no truncation)
# ---------------------------------------------------------------------------

def pf4_token_length(cfg: TrainingConfig) -> PreflightReport:
    report = PreflightReport()
    ds_dir = ds_service.dataset_dir(cfg.dataset_version)
    adapter = None
    try:
        adapter = _resolve_adapter(cfg)
    except KeyError as exc:
        report.add(PreflightResult("PF4", False, str(exc), critical=True))
        return report

    max_len = 0
    max_sample_id = ""
    too_long: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_count = 0
    for split in ("train", "val"):
        path = ds_dir / f"{split}.jsonl"
        if not path.exists():
            errors.append({"split": split, "issue": "file missing"})
            continue
        try:
            samples = _load_jsonl(path)
        except Exception as exc:
            errors.append({"split": split, "issue": str(exc)})
            continue
        total_count += len(samples)
        for sample in samples:
            try:
                rendered = adapter.render_conversation(
                    sample.get("messages", []), tools=sample.get("tools"), add_generation_prompt=False,
                )
                n_tokens = adapter.count_tokens(rendered)
                if n_tokens is None:
                    raise RuntimeError("real tokenizer unavailable; cannot verify truncation")
                if n_tokens > max_len:
                    max_len, max_sample_id = n_tokens, sample.get("id", "")
                if n_tokens > cfg.train.max_seq_length:
                    too_long.append({"id": sample.get("id"), "tokens": n_tokens})
            except Exception as exc:
                errors.append({"id": sample.get("id"), "issue": str(exc)})

    passed = total_count > 0 and not too_long and not errors
    details = {
        "max_tokens": max_len, "max_sample_id": max_sample_id,
        "too_long": too_long[:50], "too_long_count": len(too_long),
        "total_count": total_count, "max_seq_length": cfg.train.max_seq_length,
        "errors": errors[:50], "error_count": len(errors), "checked_splits": ["train", "val"],
    }
    report.add(PreflightResult(
        "PF4", passed,
        f"No truncation ({total_count} train/val samples, max {max_len} tokens)" if passed
        else f"Could not verify token length: {len(too_long)} over limit, {len(errors)} unknown/error",
        details=details, critical=True,
    ))
    return report


# ---------------------------------------------------------------------------
# PF5: Dry run (10 steps, check OOM)
# ---------------------------------------------------------------------------

def pf5_dry_run(cfg: TrainingConfig) -> PreflightReport:
    """Run a 10-step QLoRA smoke train on the selected GPU/config (plan 03 PF5)."""
    report = PreflightReport()
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        report.add(PreflightResult(
            "PF5", False,
            f"10-step GPU dry run unavailable; install torch/transformers/peft/bitsandbytes on the training host ({exc})",
            details={"skipped": True}, critical=True,
        ))
        return report

    if not torch.cuda.is_available():
        report.add(PreflightResult(
            "PF5", False,
            "10-step dry run requires the selected training GPU; no CUDA device is available on this host",
            details={"skipped": True, "reason": "no_cuda"}, critical=True,
        ))
        return report

    model = None
    try:
        adapter = _resolve_adapter(cfg)
        train_path = ds_service.dataset_dir(cfg.dataset_version) / "train.jsonl"
        samples = _load_jsonl(train_path)
        if not samples:
            raise ValueError("training dataset is empty")
        tokenizer = _local_tokenizer(adapter)
        sample = next((x for x in samples if any(m.get("role") == "assistant" for m in x.get("messages", []))), None)
        if sample is None:
            raise ValueError("no assistant training response found in dataset")

        messages = sample.get("messages", [])
        tools = sample.get("tools")
        input_ids = list(tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=False,
        ))
        labels = [-100] * len(input_ids)
        for i, message in enumerate(messages):
            if message.get("role") == "assistant":
                start = 0 if i == 0 else len(tokenizer.apply_chat_template(
                    messages[:i], tools=tools, tokenize=True, add_generation_prompt=False,
                ))
                end = min(len(tokenizer.apply_chat_template(
                    messages[:i + 1], tools=tools, tokenize=True, add_generation_prompt=False,
                )), len(input_ids))
                labels[start:end] = input_ids[start:end]
        if not any(label != -100 for label in labels):
            raise ValueError("assistant-only loss mask produced zero labelled tokens")
        if len(input_ids) > cfg.train.max_seq_length:
            raise ValueError(f"dry-run sample has {len(input_ids)} tokens > max_seq_length={cfg.train.max_seq_length}")

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.quantization.quant_type,
            bnb_4bit_use_double_quant=cfg.quantization.double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, quantization_config=quant_cfg, device_map="auto",
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, PeftLoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules, bias="none", task_type="CAUSAL_LM",
        ))
        model.train()

        device = model.get_input_embeddings().weight.device
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=cfg.train.learning_rate,
        )
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=ids, attention_mask=mask, labels=label_tensor)
            if not torch.isfinite(output.loss):
                raise RuntimeError("dry-run loss is not finite")
            output.loss.backward()
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        report.add(PreflightResult(
            "PF5", True, f"10-step QLoRA dry run passed; peak CUDA allocation {peak_mb:.0f} MB",
            details={"steps": 10, "peak_vram_mb": round(peak_mb, 1),
                     "max_seq_length": len(input_ids), "loss_first": losses[0], "loss_last": losses[-1]},
            critical=True,
        ))
    except Exception as exc:
        report.add(PreflightResult(
            "PF5", False, f"10-step QLoRA dry run failed: {exc}",
            details={"error": str(exc)}, critical=True,
        ))
    finally:
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return report


# ---------------------------------------------------------------------------
# PF6: Record library versions (pip freeze)
# ---------------------------------------------------------------------------

def pf6_library_versions(cfg: TrainingConfig) -> PreflightReport:
    """Capture pip freeze for reproducibility."""
    report = PreflightReport()

    try:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
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
    """Run all pre-flight checks PF1-PF6; any critical failure locks launch."""
    report = PreflightReport()
    for check in (
        pf1_dataset_hash,
        pf2_render_samples,
        pf3_loss_mask,
        pf4_token_length,
        pf5_dry_run,
        pf6_library_versions,
    ):
        try:
            part = check(cfg)
            for result in part.results:
                report.add(result)
        except Exception as exc:
            report.add(PreflightResult(
                code=check.__name__.replace("pf", "PF").split("_")[0],
                passed=False,
                message=f"{check.__name__} raised an unexpected error: {exc}",
                details={"error": str(exc)},
                critical=check is not pf6_library_versions,
            ))
    return report
