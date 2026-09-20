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


def list_adapters() -> list[dict[str, str]]:
    return [
        {"name": a.name, "hf_repo": a.hf_repo, "display_name": a.display_name}
        for a in sorted(ADAPTERS.values(), key=lambda x: x.name)
    ]
