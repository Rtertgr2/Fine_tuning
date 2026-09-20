#!/usr/bin/env python3
"""Quantize GGUF model to multiple levels with imatrix.

Usage:
    python -m workbench.scripts.quantize --gguf /path/to/model-f16.gguf --imatrix imatrix.dat --outdir /path/to/output/
    python -m workbench.scripts.quantize model-f16.gguf imatrix.dat -o /path/to/output/ --levels Q4_K_M Q5_K_M Q8_0

Records SHA256 hashes and llama.cpp commit for each quantization level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quantize")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", BASE_DIR.parent / "llama.cpp"))

DEFAULT_LEVELS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def find_quantize_binary() -> Path:
    """Locate llama-quantize binary."""
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "bin" / "quantize",
        LLAMA_CPP_DIR / "llama-quantize",
        LLAMA_CPP_DIR / "quantize",
    ]
    for c in candidates:
        if c.exists():
            return c
    import shutil
    for name in ["llama-quantize", "quantize"]:
        result = shutil.which(name)
        if result:
            return Path(result)
    raise FileNotFoundError(
        f"llama-quantize not found in {LLAMA_CPP_DIR}. "
        f"Build llama.cpp first or set LLAMA_CPP_DIR"
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
    return "unknown"


def sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_level(
    quantize_bin: Path,
    gguf_path: Path,
    imatrix_path: Path | None,
    output_path: Path,
    level: str,
) -> dict:
    """Quantize a single level."""
    cmd = [str(quantize_bin)]
    if imatrix_path:
        cmd.extend(["--imatrix", str(imatrix_path)])
    cmd.extend([str(gguf_path), str(output_path), level])

    logger.info(f"Quantizing to {level}: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.stdout:
        logger.info(f"stdout: {result.stdout[-1000:]}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr[-1000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"Quantization to {level} failed (rc={result.returncode})")

    if not output_path.exists():
        raise RuntimeError(f"Quantization completed but output not found: {output_path}")

    file_hash = sha256_file(output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(f"{level}: {file_size_mb:.1f} MB, sha256={file_hash[:16]}...")

    return {
        "level": level,
        "output_path": str(output_path),
        "sha256": file_hash,
        "size_mb": file_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Quantize GGUF to multiple levels")
    parser.add_argument("gguf", help="Path to f16 GGUF model")
    parser.add_argument("imatrix", nargs="?", help="Path to imatrix.dat (optional)")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS,
                        help=f"Quantization levels (default: {' '.join(DEFAULT_LEVELS)})")
    parser.add_argument("--llama-cpp-dir", help="Path to llama.cpp directory")
    parser.add_argument("--prefix", help="Output filename prefix (default: from input gguf)")
    parser.add_argument("--keep-f16", action="store_true", help="Keep f16 output name as-is")
    args = parser.parse_args()

    if args.llama_cpp_dir:
        global LLAMA_CPP_DIR
        LLAMA_CPP_DIR = Path(args.llama_cpp_dir)

    gguf_path = Path(args.gguf).resolve()
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    imatrix_path = Path(args.imatrix).resolve() if args.imatrix else None
    if imatrix_path and not imatrix_path.exists():
        raise FileNotFoundError(f"imatrix file not found: {imatrix_path}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    quantize_bin = find_quantize_binary()
    logger.info(f"Using quantize binary: {quantize_bin}")

    llama_cpp_commit = get_llama_cpp_commit(LLAMA_CPP_DIR)
    logger.info(f"llama.cpp commit: {llama_cpp_commit}")

    prefix = args.prefix or gguf_path.stem.replace("-f16", "").replace("_f16", "")

    results = []
    manifest = {
        "source_gguf": str(gguf_path),
        "source_gguf_sha256": sha256_file(gguf_path),
        "imatrix": str(imatrix_path) if imatrix_path else None,
        "llama_cpp_commit": llama_cpp_commit,
        "llama_cpp_dir": str(LLAMA_CPP_DIR),
        "quantize_binary": str(quantize_bin),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "levels": {},
    }

    for level in args.levels:
        output_name = f"{prefix}-{level}.gguf"
        output_path = outdir / output_name
        result = quantize_level(quantize_bin, gguf_path, imatrix_path, output_path, level)
        results.append(result)
        manifest["levels"][level] = result

    # Write manifest
    manifest_path = outdir / f"{prefix}_quantize_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Quantization manifest written to {manifest_path}")

    logger.info(f"Quantization complete. {len(results)} levels produced.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        sys.exit(1)
