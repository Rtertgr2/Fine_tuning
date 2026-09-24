"""Workbench configuration — paths and the active adapter."""

from __future__ import annotations

import os
from pathlib import Path

# workbench/ root (backend/app/config.py -> backend/app -> backend -> workbench)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("WORKBENCH_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("WORKBENCH_DB", DATA_DIR / "examples.db"))
DATASETS_DIR = Path(os.environ.get("WORKBENCH_DATASETS_DIR", DATA_DIR / "datasets"))
GOLD_DIR = DATA_DIR / "gold"
TOKENIZERS_DIR = Path(os.environ.get("WORKBENCH_TOKENIZERS_DIR", ROOT / "models" / "tokenizers"))
EVAL_DIR = ROOT / "eval"
# Model registry on-disk layout (plan 05 §5): models/<version>/ + models/current symlink
MODELS_DIR = Path(os.environ.get("WORKBENCH_MODELS_DIR", ROOT / "models"))
# Optional llama-server restart on promote/rollback (plan 05 §5).
# Set to a shell command, e.g. "pkill -f 'llama-server -m models/current' && llama-server ..."
LLAMA_SERVER_CMD = os.environ.get("WORKBENCH_LLAMA_SERVER_CMD", "")

ACTIVE_ADAPTER = os.environ.get("WORKBENCH_ADAPTER", "hermes2pro-llama3-8b")

# C4: rendered conversations must fit this many tokens
MAX_SEQ_LEN = int(os.environ.get("WORKBENCH_MAX_SEQ_LEN", "8192"))

# dataset split
VAL_RATIO = float(os.environ.get("WORKBENCH_VAL_RATIO", "0.1"))

# thresholds for warn-level checks (adjust from gold-set statistics, plan 01 §11)
NEAR_DUP_THRESHOLD = float(os.environ.get("WORKBENCH_NEAR_DUP", "0.85"))
# overlap with frozen eval suites: examples at/above this n-gram Jaccard are
# dropped when a dataset is built (plan 00 ข้อ 3 / plan 04 §4)
EVAL_OVERLAP_THRESHOLD = float(os.environ.get("WORKBENCH_EVAL_OVERLAP", "0.3"))
L4_GIVEUP_RATIO = 0.15  # at least 15% of loop examples end with a give-up report
S6_PASS_WINDOW = (0.35, 0.65)  # sec PASS share should be near 50%
S7_MIN_PER_TYPE = int(os.environ.get("WORKBENCH_S7_MIN", "3"))

# plan-category allowed status values (decision recorded in docs/decisions.md)
PLAN_STATUS_VALUES = ("APPROVED", "NEEDS_REVISION")
