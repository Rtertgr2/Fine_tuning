#!/usr/bin/env python3
"""Merge LoRA adapter into base model (16-bit).

Usage:
    python -m workbench.scripts.merge_adapter --run-id run_001 --output /path/to/merged
    python -m workbench.scripts.merge_adapter run_001 /path/to/merged

Uses Unsloth if installed, otherwise falls back to peft merge_and_unload.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_adapter")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_adapter(run_id: str) -> Path:
    """Locate adapter directory from a training run ID."""
    candidates = [
        BASE_DIR / "models" / "adapters" / run_id,
        BASE_DIR / "models" / "runs" / run_id / "adapter",
        BASE_DIR / "models" / "runs" / run_id,
        BASE_DIR / "models" / "adapters",
    ]
    # Direct adapter path
    for p in candidates:
        if (p / "adapter_config.json").exists():
            return p
    # Check models/runs/<run_id>/adapter
    run_dir = BASE_DIR / "models" / "runs" / run_id
    if run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_dir() and (child / "adapter_config.json").exists():
                return child
    raise FileNotFoundError(f"adapter not found for run_id={run_id}")


def merge_unsloth(adapter_path: Path, output_path: Path, base_model: str | None = None) -> dict:
    """Merge using Unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=int(os.environ.get("WORKBENCH_MAX_SEQ_LEN", "8192")),
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(model, adapter_path)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )
    logger.info(f"Unsloth merged model saved to {output_path}")
    return {"method": "unsloth", "base_model": base_model or "from_adapter_config"}


def merge_peft(adapter_path: Path, output_path: Path, base_model: str | None = None) -> dict:
    """Merge using peft merge_and_unload."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Read base model from adapter config
    config_path = adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    base_model_name = base_model or config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("Cannot determine base model. Pass --base-model")

    logger.info(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    logger.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info(f"PEFT merged model saved to {output_path}")
    return {"method": "peft", "base_model": base_model_name}


def write_metadata(output_path: Path, run_id: str, method: str, base_model: str, manifest: dict) -> None:
    """Write metadata alongside the merged model."""
    metadata_path = output_path / "merge_manifest.json"
    metadata = {
        "run_id": run_id,
        "method": method,
        "base_model": base_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    for f in sorted(output_path.rglob("*")):
        if f.is_file() and f.name != "merge_manifest.json":
            metadata["files"][str(f.relative_to(output_path))] = sha256_file(f)
    metadata["total_size_mb"] = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info(f"Manifest written to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model (16-bit)")
    parser.add_argument("run_id", help="Training run ID to locate adapter")
    parser.add_argument("output", help="Output directory for merged model")
    parser.add_argument("--base-model", help="Base model name/path (auto-detected from adapter config if omitted)")
    parser.add_argument("--method", choices=["unsloth", "peft", "auto"], default="auto",
                        help="Merge method (default: auto)")
    parser.add_argument("--no-manifest", action="store_true", help="Skip writing merge_manifest.json")
    args = parser.parse_args()

    adapter_path = find_adapter(args.run_id)
    logger.info(f"Found adapter: {adapter_path}")

    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    method = args.method
    if method == "auto":
        try:
            import unsloth  # noqa: F401
            method = "unsloth"
        except ImportError:
            method = "peft"

    logger.info(f"Using merge method: {method}")
    if method == "unsloth":
        result = merge_unsloth(adapter_path, output_path, args.base_model)
    else:
        result = merge_peft(adapter_path, output_path, args.base_model)

    if not args.no_manifest:
        write_metadata(output_path, args.run_id, result["method"], result["base_model"], result)

    logger.info("Merge complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
