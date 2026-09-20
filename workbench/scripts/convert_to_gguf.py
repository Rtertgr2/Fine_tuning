#!/usr/bin/env python3
"""Convert HuggingFace model to GGUF format.

Usage:
    python -m workbench.scripts.convert_to_gguf /path/to/merged /path/to/output.gguf
    python -m workbench.scripts.convert_to_gguf /path/to/merged --outdir /path/to/models/

Calls convert_hf_to_gguf.py from llama.cpp and records the llama.cpp commit hash.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_to_gguf")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Environment defaults
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", BASE_DIR.parent / "llama.cpp"))
CONVERT_SCRIPT = os.environ.get("LLAMA_CPP_CONVERT", "convert_hf_to_gguf.py")


def find_convert_script() -> Path:
    """Locate convert_hf_to_gguf.py in the llama.cpp directory."""
    # Check direct path first
    direct = LLAMA_CPP_DIR / CONVERT_SCRIPT
    if direct.exists():
        return direct
    # Check build directory
    for candidate in [
        LLAMA_CPP_DIR / "convert_hf_to_gguf.py",
        LLAMA_CPP_DIR / "scripts" / "convert_hf_to_gguf.py",
        Path(CONVERT_SCRIPT),
    ]:
        if candidate.exists():
            return candidate
    # Try finding via which
    import shutil
    which_result = shutil.which(CONVERT_SCRIPT)
    if which_result:
        return Path(which_result)
    raise FileNotFoundError(
        f"convert_hf_to_gguf.py not found in {LLAMA_CPP_DIR}. "
        f"Set LLAMA_CPP_DIR environment variable or pass --llama-cpp-dir"
    )


def get_llama_cpp_commit(llama_cpp_dir: Path) -> str:
    """Get the current git commit hash of llama.cpp."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=llama_cpp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # Check for a commit hash file if git isn't available
    commit_file = llama_cpp_dir / ".git" / "HEAD"
    if commit_file.exists():
        content = commit_file.read_text().strip()
        if not content.startswith("ref:"):
            return content
    return "unknown"


def get_llama_cpp_version_info(llama_cpp_dir: Path) -> dict:
    """Collect version info for reproducibility."""
    info = {
        "commit": get_llama_cpp_commit(llama_cpp_dir),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    try:
        # Try to get tag as well
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=llama_cpp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            info["describe"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return info


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    outtype: str = "f16",
    vocab_only: bool = False,
    extra_args: list[str] | None = None,
) -> dict:
    """Run convert_hf_to_gguf.py."""
    convert_script = find_convert_script()
    llama_cpp_dir = convert_script.parent if convert_script.parent.name == "scripts" else LLAMA_CPP_DIR

    logger.info(f"Using convert script: {convert_script}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Output type: {outtype}")

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]
    if vocab_only:
        cmd.append("--vocab-only")
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running: {' '.join(cmd)}")

    # Collect version info before conversion
    version_info = get_llama_cpp_version_info(llama_cpp_dir)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.stdout:
        logger.info(f"stdout: {result.stdout[-2000:]}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr[-2000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py failed with return code {result.returncode}")

    if not output_path.exists():
        raise RuntimeError(f"Conversion completed but output file not found: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"GGUF conversion complete. Output size: {output_size_mb:.1f} MB")

    return {
        "version_info": version_info,
        "output_size_mb": output_size_mb,
        "outtype": outtype,
    }


def write_metadata(output_path: Path, model_path: Path, result: dict, extra: dict | None = None) -> None:
    """Write conversion metadata."""
    metadata_path = output_path.parent / f"{output_path.stem}_conversion.json"
    import hashlib
    sha256 = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    metadata = {
        "source_model": str(model_path),
        "output_file": str(output_path),
        "output_sha256": sha256.hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "convert_script": str(find_convert_script()),
        **result,
    }
    if extra:
        metadata.update(extra)

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info(f"Conversion metadata written to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to GGUF")
    parser.add_argument("model_path", help="Path to HuggingFace model directory")
    parser.add_argument("output", help="Output GGUF file path (or output directory with --outdir)")
    parser.add_argument("--outtype", default="f16", choices=["f16", "f32", "bf16", "q8_0", "auto"],
                        help="GGUF output type (default: f16)")
    parser.add_argument("--vocab-only", action="store_true", help="Only output vocab")
    parser.add_argument("--llama-cpp-dir", help="Path to llama.cpp directory")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Extra arguments to pass to convert_hf_to_gguf.py")
    parser.add_argument("--no-metadata", action="store_true", help="Skip writing metadata JSON")
    args = parser.parse_args()

    if args.llama_cpp_dir:
        global LLAMA_CPP_DIR
        LLAMA_CPP_DIR = Path(args.llama_cpp_dir)

    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output).resolve()

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Not a valid model directory (no config.json): {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = convert_to_gguf(
        model_path=model_path,
        output_path=output_path,
        outtype=args.outtype,
        vocab_only=args.vocab_only,
        extra_args=args.extra_arg or None,
    )

    if not args.no_metadata:
        write_metadata(output_path, model_path, result)

    logger.info("Conversion complete.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)
