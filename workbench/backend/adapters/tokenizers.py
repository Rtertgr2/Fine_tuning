"""Real-tokenizer access for the active adapter (T1.3).

Downloads only the tokenizer files from the HF repo into
`models/tokenizers/<adapter-name>/`, then counts with the real tokenizer.
If the download or load fails (offline), counting returns None and callers
must report "token count unavailable" instead of guessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.app.config import TOKENIZERS_DIR

logger = logging.getLogger(__name__)

_CACHE: dict[str, Any] = {}
_FAILED: set[str] = set()

# tokenizer files only — no weights, no configs beyond what the tokenizer needs
_ALLOW_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]


def _local_dir(adapter_name: str) -> Path:
    return TOKENIZERS_DIR / adapter_name


def _load(adapter: Any) -> Any | None:
    name = adapter.name
    if name in _CACHE:
        return _CACHE[name]
    if name in _FAILED:
        return None

    local = _local_dir(name)
    try:
        if not (local / "tokenizer.json").exists() and not (local / "vocab.json").exists():
            from huggingface_hub import snapshot_download

            local.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=adapter.hf_repo,
                local_dir=str(local),
                allow_patterns=_ALLOW_PATTERNS,
            )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(local))
        _CACHE[name] = tok
        return tok
    except Exception as exc:  # noqa: BLE001 — offline is a supported state
        logger.warning("tokenizer for %s unavailable: %s", name, exc)
        _FAILED.add(name)
        return None


def is_available(adapter: Any) -> bool:
    return _load(adapter) is not None


def count_tokens(adapter: Any, text: str) -> int | None:
    tok = _load(adapter)
    if tok is None:
        return None
    # the rendered text already carries special tokens as literal text
    return len(tok(text, add_special_tokens=False)["input_ids"])


def reset_cache() -> None:
    """For tests."""
    _CACHE.clear()
    _FAILED.clear()
