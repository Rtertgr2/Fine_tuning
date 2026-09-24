"""Adapter registry — the one place that knows which models are supported.

Switching base models = picking a different adapter name (config / env var);
validators, API and pipelines stay untouched.
"""

from __future__ import annotations

from backend.adapters.base import ModelAdapter
from backend.adapters.hermes2pro import Hermes2ProLlama3Adapter
from backend.adapters.qwen25 import Qwen25Adapter

ADAPTERS: dict[str, ModelAdapter] = {
    a.name: a for a in (Hermes2ProLlama3Adapter(), Qwen25Adapter())
}

# Exact model identity allowlist: model_id -> adapter name.
# Only exact matches on these IDs are accepted. Substring matching
# is intentionally removed to prevent false positives on model names
# that happen to contain a substring of a supported identifier.
MODEL_ID_ALLOWLIST: dict[str, str] = {
    "hermes2pro-llama3-8b": "hermes2pro-llama3-8b",
    "nousresearch/hermes-2-pro-llama-3-8b": "hermes2pro-llama3-8b",
    "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "qwen/qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "qwen/qwen2.5-coder-7b-instruct": "qwen2.5-7b-instruct",
}

DEFAULT_ADAPTER = "hermes2pro-llama3-8b"


def get_adapter(name: str | None = None) -> ModelAdapter:
    key = name or DEFAULT_ADAPTER
    try:
        return ADAPTERS[key]
    except KeyError:
        raise KeyError(
            f"unknown adapter {key!r}; available: {', '.join(sorted(ADAPTERS))}"
        ) from None


def adapter_for_model_id(model_id: str | None) -> ModelAdapter:
    """Resolve a HF model name to its locked chat-template/tokenizer adapter.

    Uses exact allowlist matching only — the full model id, adapter name,
    or hf_repo must match a known entry. Substring matching was removed
    because it allowed false positives on model names containing only a
    portion of a supported identifier. Unknown families fail closed.
    """
    if not model_id:
        raise KeyError("model_id must be provided")
    key = model_id.strip().lower()
    adapter_name = MODEL_ID_ALLOWLIST.get(key)
    if adapter_name is None:
        raise KeyError(
            f"no registered chat-template adapter for base model {model_id!r}; "
            f"allowed: {', '.join(sorted(MODEL_ID_ALLOWLIST))}"
        )
    return ADAPTERS[adapter_name]


def list_adapters() -> list[dict[str, str]]:
    return [
        {"name": a.name, "hf_repo": a.hf_repo, "display_name": a.display_name}
        for a in sorted(ADAPTERS.values(), key=lambda x: x.name)
    ]
