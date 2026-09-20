#!/usr/bin/env python3
"""Generate importance matrix for GGUF quantization.

Usage:
    python -m workbench.scripts.create_imatrix --gguf /path/to/model-f16.gguf --calib /path/to/calib.txt -o imatrix.dat
    python -m workbench.scripts.create_imatrix --gguf /path/to/model-f16.gguf --dataset v0001 -o imatrix.dat

Mixes training samples with general Python code for calibration.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_imatrix")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", BASE_DIR.parent / "llama.cpp"))

# General Python code samples for calibration
GENERAL_PYTHON_CODE = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n    def is_empty(self):\n        return len(self.items) == 0\n",
    "import json\n\ndef parse_config(path):\n    with open(path) as f:\n        return json.load(f)\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "from pathlib import Path\n\ndef find_files(directory, pattern='*.py'):\n    return list(Path(directory).rglob(pattern))\n",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n",
    "import re\n\ndef extract_emails(text):\n    return re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', text)\n",
    "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return None\n    def put(self, key, value):\n        if key in self.cache:\n            self.order.remove(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.cache[oldest]\n        self.cache[key] = value\n        self.order.append(key)\n",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
    "import hashlib\n\ndef sha256_file(path):\n    h = hashlib.sha256()\n    with open(path, 'rb') as f:\n        for chunk in iter(lambda: f.read(8192), b''):\n            h.update(chunk)\n    return h.hexdigest()\n",
    "def bfs(graph, start):\n    visited = set()\n    queue = [start]\n    result = []\n    while queue:\n        node = queue.pop(0)\n        if node not in visited:\n            visited.add(node)\n            result.append(node)\n            queue.extend(graph.get(node, []))\n    return result\n",
    "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef inorder(root):\n    return inorder(root.left) + [root.val] + inorder(root.right) if root else []\n",
    "import sqlite3\n\ndef query_all(conn, table):\n    cur = conn.execute(f'SELECT * FROM {table}')\n    return cur.fetchall()\n",
    "def retry(func, max_attempts=3, delay=1):\n    import time\n    for attempt in range(max_attempts):\n        try:\n            return func()\n        except Exception as e:\n            if attempt == max_attempts - 1:\n                raise\n            time.sleep(delay)\n",
    "from dataclasses import dataclass\nfrom typing import Optional\n\n@dataclass\nclass Config:\n    host: str = 'localhost'\n    port: int = 8080\n    debug: bool = False\n    workers: int = 4\n",
    "def levenshtein(s1, s2):\n    if len(s1) < len(s2):\n        return levenshtein(s2, s1)\n    if len(s2) == 0:\n        return len(s1)\n    prev = range(len(s2) + 1)\n    for i, c1 in enumerate(s1):\n        curr = [i + 1]\n        for j, c2 in enumerate(s2):\n            insertions = prev[j + 1] + 1\n            deletions = curr[j] + 1\n            subs = prev[j] + (c1 != c2)\n            curr.append(min(insertions, deletions, subs))\n        prev = curr\n    return prev[-1]\n",
    "import asyncio\n\nasync def fetch_all(urls):\n    import aiohttp\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_one(session, url) for url in urls]\n        return await asyncio.gather(*tasks)\n",
    "def flatten(nested):\n    for item in nested:\n        if isinstance(item, (list, tuple)):\n            yield from flatten(item)\n        else:\n            yield item\n",
    "class Observer:\n    def __init__(self):\n        self._listeners = {}\n    def on(self, event, callback):\n        self._listeners.setdefault(event, []).append(callback)\n    def emit(self, event, *args, **kwargs):\n        for cb in self._listeners.get(event, []):\n            cb(*args, **kwargs)\n",
]


def find_imatrix_binary() -> Path:
    """Locate llama-imatrix binary."""
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-imatrix",
        LLAMA_CPP_DIR / "build" / "bin" / "imatrix",
        LLAMA_CPP_DIR / "llama-imatrix",
        LLAMA_CPP_DIR / "imatrix",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Try PATH
    import shutil
    for name in ["llama-imatrix", "imatrix"]:
        result = shutil.which(name)
        if result:
            return Path(result)
    raise FileNotFoundError(
        f"llama-imatrix not found in {LLAMA_CPP_DIR}. "
        f"Build llama.cpp first or set LLAMA_CPP_DIR"
    )


def load_training_samples(dataset_version: str | None = None, max_samples: int = 200) -> list[str]:
    """Load training samples from dataset."""
    samples = []
    datasets_dir = BASE_DIR / "data" / "datasets"

    if dataset_version:
        dataset_dir = datasets_dir / dataset_version
        if not dataset_dir.exists():
            logger.warning(f"Dataset {dataset_version} not found at {dataset_dir}")
            return []
        train_file = dataset_dir / "train.jsonl"
        if train_file.exists():
            with open(train_file) as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    try:
                        data = json.loads(line)
                        # Extract text content from messages
                        if "messages" in data:
                            for msg in data["messages"]:
                                if msg.get("content"):
                                    samples.append(msg["content"])
                        elif "text" in data:
                            samples.append(data["text"])
                    except json.JSONDecodeError:
                        continue
    else:
        # Try to find any available dataset
        if datasets_dir.exists():
            for ds_dir in sorted(datasets_dir.iterdir()):
                train_file = ds_dir / "train.jsonl"
                if train_file.exists():
                    with open(train_file) as f:
                        for i, line in enumerate(f):
                            if i >= max_samples:
                                break
                            try:
                                data = json.loads(line)
                                if "messages" in data:
                                    for msg in data["messages"]:
                                        if msg.get("content"):
                                            samples.append(msg["content"])
                                elif "text" in data:
                                    samples.append(data["text"])
                            except json.JSONDecodeError:
                                continue
                    break

    return samples


def generate_calibration_text(
    output_path: Path,
    dataset_version: str | None = None,
    max_train_samples: int = 200,
    general_code_count: int = 50,
    seed: int = 42,
) -> dict:
    """Generate calibration text file mixing training samples with general code."""
    rng = random.Random(seed)

    # Load training samples
    train_samples = load_training_samples(dataset_version, max_train_samples)
    logger.info(f"Loaded {len(train_samples)} training samples")

    # Select general code samples
    general_samples = rng.sample(GENERAL_PYTHON_CODE, min(general_code_count, len(GENERAL_PYTHON_CODE)))
    logger.info(f"Selected {len(general_samples)} general code samples")

    # Mix them: interleave training and general code
    all_samples = []
    t_idx, g_idx = 0, 0
    while t_idx < len(train_samples) or g_idx < len(general_samples):
        if t_idx < len(train_samples):
            all_samples.append(train_samples[t_idx])
            t_idx += 1
        if g_idx < len(general_samples):
            all_samples.append(general_samples[g_idx])
            g_idx += 1

    # Shuffle
    rng.shuffle(all_samples)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(sample.replace("\n", " ") + "\n")

    total_chars = sum(len(s) for s in all_samples)
    logger.info(f"Calibration text written: {len(all_samples)} samples, {total_chars} chars")

    return {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "general_samples": len(general_samples),
        "total_chars": total_chars,
        "seed": seed,
    }


def run_imatrix(
    gguf_path: Path,
    calib_path: Path,
    output_path: Path,
    extra_args: list[str] | None = None,
) -> dict:
    """Run llama-imatrix."""
    imatrix_bin = find_imatrix_binary()
    logger.info(f"Using imatrix binary: {imatrix_bin}")

    cmd = [
        str(imatrix_bin),
        "-m", str(gguf_path),
        "-f", str(calib_path),
        "-o", str(output_path),
        "--output-frequency", "10",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200,
    )

    if result.stdout:
        logger.info(f"stdout: {result.stdout[-2000:]}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr[-2000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"llama-imatrix failed with return code {result.returncode}")

    if not output_path.exists():
        raise RuntimeError(f"imatrix completed but output file not found: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"imatrix generation complete. Output size: {output_size_mb:.1f} MB")

    return {
        "output_size_mb": output_size_mb,
        "binary": str(imatrix_bin),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate importance matrix for GGUF quantization")
    parser.add_argument("--gguf", required=True, help="Path to f16 GGUF model")
    parser.add_argument("-o", "--output", required=True, help="Output imatrix.dat path")
    parser.add_argument("--calib", help="Calibration text file (generated if not provided)")
    parser.add_argument("--dataset", help="Dataset version to sample from (e.g., v0001)")
    parser.add_argument("--max-train-samples", type=int, default=200,
                        help="Max training samples to include (default: 200)")
    parser.add_argument("--general-code-count", type=int, default=50,
                        help="Number of general code samples (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--llama-cpp-dir", help="Path to llama.cpp directory")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Extra arguments to pass to llama-imatrix")
    args = parser.parse_args()

    if args.llama_cpp_dir:
        global LLAMA_CPP_DIR
        LLAMA_CPP_DIR = Path(args.llama_cpp_dir)

    gguf_path = Path(args.gguf).resolve()
    output_path = Path(args.output).resolve()

    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    # Generate or use calibration file
    if args.calib:
        calib_path = Path(args.calib).resolve()
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        calib_info = {"source": "user_provided", "path": str(calib_path)}
    else:
        calib_path = output_path.parent / f"{gguf_path.stem}_calib.txt"
        calib_info = generate_calibration_text(
            calib_path,
            dataset_version=args.dataset,
            max_train_samples=args.max_train_samples,
            general_code_count=args.general_code_count,
            seed=args.seed,
        )

    result = run_imatrix(gguf_path, calib_path, output_path, args.extra_arg)

    # Write metadata
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    metadata = {
        "gguf_model": str(gguf_path),
        "calibration": calib_info,
        "imatrix": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info(f"Metadata written to {metadata_path}")

    logger.info("imatrix generation complete.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"imatrix generation failed: {e}")
        sys.exit(1)
