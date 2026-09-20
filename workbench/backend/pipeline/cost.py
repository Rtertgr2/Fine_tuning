"""Cost control for generation runs (T2.10).

Budget cap, token logging, and result caching.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Track spending for a generation run."""

    max_usd: float = 0.0  # 0 = unlimited
    spent_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def can_spend(self, estimated_usd: float) -> bool:
        if self.max_usd <= 0:
            return True
        return self.spent_usd + estimated_usd <= self.max_usd

    def record(self, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.spent_usd += cost_usd

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_usd": self.max_usd,
            "spent_usd": round(self.spent_usd, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------


# Approximate costs per 1M tokens (USD)
TOKEN_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "claude-sonnet-4": (3.0, 15.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a model call."""
    rates = TOKEN_COSTS.get(model, (1.0, 3.0))  # default conservative
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


# ---------------------------------------------------------------------------
# Simple file-based cache
# ---------------------------------------------------------------------------


class ResultCache:
    """File-based cache for generation results (keyed by prompt hash)."""

    def __init__(self, cache_dir: str | None = None):
        from pathlib import Path
        from backend.app import config

        self._dir = Path(cache_dir) if cache_dir else config.DATA_DIR / "pipeline_cache"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str) -> str:
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def get(self, prompt: str) -> dict[str, Any] | None:
        """Get cached result for a prompt."""
        from pathlib import Path
        path = self._dir / f"{self._key(prompt)}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
        return None

    def put(self, prompt: str, result: dict[str, Any]) -> None:
        """Cache a result for a prompt."""
        from pathlib import Path
        path = self._dir / f"{self._key(prompt)}.json"
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> int:
        """Clear all cached results. Returns count cleared."""
        count = 0
        for path in self._dir.glob("*.json"):
            path.unlink()
            count += 1
        return count
