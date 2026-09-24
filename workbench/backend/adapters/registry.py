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

    Model names from the same supported family (for example Qwen Coder and
    Qwen Instruct) share the Qwen2.5 template. Unknown families fail closed
    instead of silently rendering with the active default adapter.
    """
    name = (model_id or "").strip().lower()
    hermes = ADAPTERS["hermes2pro-llama3-8b"]
    qwen = ADAPTERS["qwen2.5-7b-instruct"]
    if name in {hermes.name.lower(), hermes.hf_repo.lower()} or "hermes-2-pro-llama-3-8b" in name:
        return hermes
    if name in {qwen.name.lower(), qwen.hf_repo.lower()} or "qwen2.5" in name or "qwen-2.5" in name:
        return qwen
    raise KeyError(f"no registered chat-template adapter for base model {model_id!r}")


def list_adapters() -> list[dict[str, str]]:
    return [
        {"name": a.name, "hf_repo": a.hf_repo, "display_name": a.display_name}
        for a in sorted(ADAPTERS.values(), key=lambda x: x.name)
    ]
