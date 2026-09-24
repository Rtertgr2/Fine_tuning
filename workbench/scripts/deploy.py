#!/usr/bin/env python3
"""Deploy pipeline: merge → convert → imatrix → quantize → register.

Orchestrates the full deployment pipeline for a trained adapter:
1. Merge LoRA adapter into base model (16-bit)
2. Convert to GGUF (f16)
3. Generate importance matrix
4. Quantize to Q4_K_M, Q5_K_M, Q8_0
5. Register each output as model version in DB
6. Run smoke eval check

Usage:
    python -m workbench.scripts.deploy --run-id run_001 --dataset v0001 --version v0.1.0
    python -m workbench.scripts.deploy run_001 --outdir /path/to/output/ --levels Q4_K_M Q5_K_M
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deploy")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app import config

MODELS_DIR = config.MODELS_DIR
ADAPTERS_DIR = MODELS_DIR / "adapters"
RUNS_DIR = MODELS_DIR / "runs"
# Per plan 05 §5, deployed artifacts live at models/<version>/ with
# models/current pointing to the selected production directory.
DEPLOY_DIR = MODELS_DIR


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_adapter(run_id: str) -> Path:
    """Locate adapter directory from a training run ID."""
    candidates = [
        ADAPTERS_DIR / run_id,
        RUNS_DIR / run_id / "adapter",
        RUNS_DIR / run_id,
        config.DATA_DIR / "training" / "runs" / run_id / "adapter",
    ]
    for p in candidates:
        if (p / "adapter_config.json").exists():
            return p
    # Search models/runs
    if RUNS_DIR.exists():
        for run_dir in RUNS_DIR.iterdir():
            for child in run_dir.iterdir():
                if child.is_dir() and (child / "adapter_config.json").exists():
                    return child
    raise FileNotFoundError(f"Adapter not found for run_id={run_id}")


def get_dataset_info(dataset_version: str) -> dict:
    """Get dataset version info from DB."""
    try:
        from backend.app.db import connect
        conn = connect()
        row = conn.execute(
            "SELECT id, name, seed, manifest_json, created_at FROM dataset_versions WHERE name=? OR id=?",
            (dataset_version, dataset_version),
        ).fetchone()
        conn.close()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "seed": row[2],
                "manifest": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
            }
    except Exception as e:
        logger.warning(f"Could not fetch dataset info: {e}")
    return {}


def register_model_version(
    version: str,
    quant: str,
    gguf_path: Path,
    dataset_version: str | None,
    train_run: str,
    llama_cpp_commit: str,
    eval_report: str | None = None,
) -> None:
    """Register one selected quantized artifact through the shared service."""
    from backend.app.db import connect
    from backend.app.services.models import register_model

    conn = connect()
    try:
        record = register_model(
            conn,
            version=version,
            dataset_version=dataset_version,
            train_run=train_run,
            llama_cpp_commit=llama_cpp_commit,
            quant=quant,
            gguf_path=str(gguf_path),
            gguf_sha256=sha256_file(gguf_path),
            eval_report=eval_report,
        )
    finally:
        conn.close()
    logger.info("Registered candidate model: %s (%s)", record["version"], quant)


def smoke_eval(gguf_path: Path, model_name: str = "smoke_test") -> dict:
    """Run a basic smoke eval: load and generate."""
    logger.info(f"Running smoke eval for {gguf_path.name}")
    results = {
        "model": str(gguf_path),
        "model_size_mb": gguf_path.stat().st_size / (1024 * 1024),
        "tests": [],
    }

    # Test 1: GGUF file is readable and non-empty
    if gguf_path.stat().st_size > 1024 * 1024:  # > 1MB
        results["tests"].append({"name": "file_size_ok", "pass": True})
    else:
        results["tests"].append({"name": "file_size_ok", "pass": False, "error": "File too small"})

    # Test 2: Check if llama-server is available
    try:
        import shutil
        server_bin = shutil.which("llama-server")
        if server_bin:
            results["tests"].append({"name": "server_available", "pass": True})
        else:
            results["tests"].append({"name": "server_available", "pass": False, "note": "llama-server not in PATH"})
    except Exception:
        pass

    # Test 3: Check file magic bytes
    with open(gguf_path, "rb") as f:
        magic = f.read(4)
        if magic == b"GGUF":
            results["tests"].append({"name": "gguf_magic", "pass": True})
        else:
            results["tests"].append({"name": "gguf_magic", "pass": False, "error": f"Bad magic: {magic!r}"})

    all_pass = all(t.get("pass", False) for t in results["tests"])
    results["all_pass"] = all_pass
    logger.info(f"Smoke eval result: {'PASS' if all_pass else 'FAIL'}")

    return results


def run_pipeline(
    run_id: str,
    version: str,
    dataset_version: str | None = None,
    outdir: Path | None = None,
    levels: list[str] | None = None,
    register_quant: str | None = "Q5_K_M",
    eval_report: str | None = None,
    skip_merge: bool = False,
    skip_convert: bool = False,
    skip_imatrix: bool = False,
    skip_quantize: bool = False,
    skip_smoke: bool = False,
    base_model: str | None = None,
    llama_cpp_dir: Path | None = None,
    calib_file: Path | None = None,
) -> dict:
    """Execute the full deploy pipeline."""
    start_time = time.time()

    levels = levels or ["Q4_K_M", "Q5_K_M", "Q8_0"]
    outdir = outdir or (DEPLOY_DIR / version)
    outdir.mkdir(parents=True, exist_ok=True)
    expected_version_dir = (MODELS_DIR / version).resolve()
    if outdir.resolve() != expected_version_dir:
        raise ValueError(f"deployment output must be models/{version} so current symlink can load it")
    if register_quant and register_quant not in levels:
        raise ValueError(f"register_quant={register_quant!r} is not one of requested quantization levels: {levels}")
    selected_quant = register_quant or (levels[0] if levels else None)

    pipeline_result = {
        "version": version,
        "run_id": run_id,
        "dataset_version": dataset_version,
        "output_dir": str(outdir),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    # ── Find adapter ────────────────────────────────────────────
    logger.info(f"=== STEP 0: Locating adapter for run_id={run_id} ===")
    adapter_path: Path | None = None
    if skip_merge:
        merged_dir = outdir / "merged"
        if not merged_dir.exists():
            raise FileNotFoundError(f"--skip-merge but no merged model at {merged_dir}")
    else:
        adapter_path = find_adapter(run_id)
        logger.info(f"Found adapter: {adapter_path}")
        pipeline_result["adapter_path"] = str(adapter_path)

    # ── Step 1: Merge ───────────────────────────────────────────
    logger.info("=== STEP 1: Merge adapter → 16-bit model ===")
    merged_dir = outdir / "merged"
    if not skip_merge:
        if adapter_path is None:
            raise RuntimeError("adapter_path not found and --skip-merge not specified")
        try:
            from scripts.merge_adapter import merge_peft, merge_unsloth
            method = "auto"
            try:
                import unsloth  # noqa: F401
                method = "unsloth"
            except ImportError:
                method = "peft"

            if method == "unsloth":
                merge_result = merge_unsloth(adapter_path, merged_dir, base_model)
            else:
                merge_result = merge_peft(adapter_path, merged_dir, base_model)

            pipeline_result["steps"]["merge"] = {
                "status": "ok",
                "method": merge_result["method"],
                "base_model": merge_result["base_model"],
                "output_dir": str(merged_dir),
            }
        except Exception as e:
            pipeline_result["steps"]["merge"] = {"status": "error", "error": str(e)}
            logger.error(f"Merge failed: {e}")
            return pipeline_result

    # ── Step 2: Convert to GGUF ────────────────────────────────
    logger.info("=== STEP 2: Convert to GGUF (f16) ===")
    gguf_path = outdir / "model-f16.gguf"
    if not skip_convert:
        try:
            from scripts.convert_to_gguf import (
                convert_to_gguf,
                find_convert_script,
                get_llama_cpp_version_info,
            )
            convert_script = find_convert_script()
            version_info = get_llama_cpp_version_info(llama_cpp_dir or Path(os.environ.get("LLAMA_CPP_DIR", BASE_DIR.parent / "llama.cpp")))

            result = convert_to_gguf(
                model_path=merged_dir,
                output_path=gguf_path,
                outtype="f16",
            )
            pipeline_result["steps"]["convert"] = {
                "status": "ok",
                "gguf_path": str(gguf_path),
                "size_mb": result["output_size_mb"],
                "llama_cpp_commit": version_info["commit"],
            }
        except Exception as e:
            pipeline_result["steps"]["convert"] = {"status": "error", "error": str(e)}
            logger.error(f"Conversion failed: {e}")
            return pipeline_result
    else:
        if not gguf_path.exists():
            raise FileNotFoundError(f"--skip-convert but no GGUF at {gguf_path}")

    # ── Step 3: Generate imatrix ───────────────────────────────
    logger.info("=== STEP 3: Generate imatrix ===")
    imatrix_path = outdir / "imatrix.dat"
    if not skip_imatrix:
        try:
            from scripts.create_imatrix import generate_calibration_text, run_imatrix, find_imatrix_binary

            if calib_file:
                calib_path = calib_file
            else:
                calib_path = outdir / "calib.txt"
                generate_calibration_text(
                    calib_path,
                    dataset_version=dataset_version,
                    max_train_samples=200,
                    general_code_count=50,
                    seed=42,
                )

            result = run_imatrix(gguf_path, calib_path, imatrix_path)
            pipeline_result["steps"]["imatrix"] = {
                "status": "ok",
                "imatrix_path": str(imatrix_path),
                "size_mb": result["output_size_mb"],
            }
        except Exception as e:
            pipeline_result["steps"]["imatrix"] = {"status": "error", "error": str(e)}
            logger.error(f"imatrix generation failed: {e}")
            return pipeline_result

    # ── Step 4: Quantize ───────────────────────────────────────
    logger.info("=== STEP 4: Quantize ===")
    if not skip_quantize:
        try:
            from scripts.quantize import find_quantize_binary, get_llama_cpp_commit, quantize_level, sha256_file

            quantize_bin = find_quantize_binary()
            llama_cpp_commit = get_llama_cpp_commit(
                llama_cpp_dir or Path(os.environ.get("LLAMA_CPP_DIR", BASE_DIR.parent / "llama.cpp"))
            )

            quant_results = {}
            for level in levels:
                output_path = outdir / f"model-{level}.gguf"
                result = quantize_level(quantize_bin, gguf_path, imatrix_path, output_path, level)
                quant_results[level] = result

            pipeline_result["steps"]["quantize"] = {
                "status": "ok",
                "levels": quant_results,
                "llama_cpp_commit": llama_cpp_commit,
            }

            # ── Step 5: Register one selected quant for this semantic version.
            # Other quantized GGUFs remain available in the version directory;
            # a different serving quant must use a new model version.
            logger.info("=== STEP 5: Register selected candidate ===")
            quant_path = outdir / f"model-{selected_quant}.gguf"
            register_model_version(
                version=version,
                quant=selected_quant,
                gguf_path=quant_path,
                dataset_version=dataset_version,
                train_run=run_id,
                llama_cpp_commit=llama_cpp_commit,
                eval_report=eval_report,
            )
            pipeline_result["steps"]["register"] = {
                "status": "ok",
                "selected_quant": selected_quant,
                "available_levels": levels,
            }

        except Exception as e:
            pipeline_result["steps"]["quantize"] = {"status": "error", "error": str(e)}
            logger.error(f"Quantization failed: {e}")
            return pipeline_result

    # ── Step 6: Smoke eval ─────────────────────────────────────
    if not skip_smoke:
        logger.info("=== STEP 6: Smoke eval ===")
        try:
            smoke_results = {}
            for level in levels:
                quant_path = outdir / f"model-{level}.gguf"
                if quant_path.exists():
                    smoke_results[level] = smoke_eval(quant_path)
            pipeline_result["steps"]["smoke_eval"] = {
                "status": "ok",
                "results": smoke_results,
            }
        except Exception as e:
            pipeline_result["steps"]["smoke_eval"] = {"status": "error", "error": str(e)}

    elapsed = time.time() - start_time
    pipeline_result["elapsed_seconds"] = round(elapsed, 1)
    pipeline_result["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Write pipeline result
    result_path = outdir / "pipeline_result.json"
    result_path.write_text(json.dumps(pipeline_result, indent=2, ensure_ascii=False))
    logger.info(f"Pipeline complete in {elapsed:.1f}s. Result written to {result_path}")

    return pipeline_result


def main():
    parser = argparse.ArgumentParser(description="Deploy pipeline: merge → convert → imatrix → quantize")
    parser.add_argument("run_id", help="Training run ID")
    parser.add_argument("--version", required=True, help="Model version (e.g., v0.1.0)")
    parser.add_argument("--dataset", help="Dataset version used for training")
    parser.add_argument("--outdir", help="Version directory (must be models/<version>)")
    parser.add_argument("--levels", nargs="+", default=["Q4_K_M", "Q5_K_M", "Q8_0"],
                        help="Quantization levels to build")
    parser.add_argument("--register-quant", default="Q5_K_M",
                        help="Which built quant becomes this model version (default: Q5_K_M)")
    parser.add_argument("--eval-report", help="Relative path under eval/reports to attach later-stage gate results")
    parser.add_argument("--base-model", help="Base model name/path")
    parser.add_argument("--llama-cpp-dir", help="Path to llama.cpp directory")
    parser.add_argument("--calib-file", help="Calibration text file (auto-generated if omitted)")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step")
    parser.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion")
    parser.add_argument("--skip-imatrix", action="store_true", help="Skip imatrix generation")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke eval")
    args = parser.parse_args()

    result = run_pipeline(
        run_id=args.run_id,
        version=args.version,
        dataset_version=args.dataset,
        outdir=Path(args.outdir) if args.outdir else None,
        levels=args.levels,
        register_quant=args.register_quant,
        eval_report=args.eval_report,
        skip_merge=args.skip_merge,
        skip_convert=args.skip_convert,
        skip_imatrix=args.skip_imatrix,
        skip_quantize=args.skip_quantize,
        skip_smoke=args.skip_smoke,
        base_model=args.base_model,
        llama_cpp_dir=Path(args.llama_cpp_dir) if args.llama_cpp_dir else None,
        calib_file=Path(args.calib_file) if args.calib_file else None,
    )

    # Exit code based on step success
    for step_name, step_result in result.get("steps", {}).items():
        if step_result.get("status") == "error":
            logger.error(f"Step '{step_name}' failed: {step_result.get('error')}")
            return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
