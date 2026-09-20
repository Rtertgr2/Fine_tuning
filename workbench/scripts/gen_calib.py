#!/usr/bin/env python3
"""Generate calibration text for imatrix quantization.

Samples from training dataset and mixes with general Python code.
Can be used standalone or called from create_imatrix.py.

Usage:
    python -m workbench.scripts.gen_calib --dataset v0001 -o calib.txt
    python -m workbench.scripts.gen_calib --dataset v0001 -o calib.txt --max-samples 300 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gen_calib")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.create_imatrix import (
    load_training_samples,
    generate_calibration_text,
)


def main():
    parser = argparse.ArgumentParser(description="Generate calibration text for imatrix")
    parser.add_argument("-o", "--output", required=True, help="Output calibration file path")
    parser.add_argument("--dataset", help="Dataset version to sample from (e.g., v0001)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max training samples (default: 200)")
    parser.add_argument("--general-code-count", type=int, default=50,
                        help="Number of general code samples (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = generate_calibration_text(
        output_path=output_path,
        dataset_version=args.dataset,
        max_train_samples=args.max_samples,
        general_code_count=args.general_code_count,
        seed=args.seed,
    )

    logger.info(f"Calibration text generated: {result['total_samples']} samples, "
                f"{result['total_chars']} chars")
    logger.info(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
