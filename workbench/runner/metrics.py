"""Metrics tracking for training runs (T3.4)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MetricEntry:
    step: int
    train_loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    epoch: float | None = None
    elapsed_seconds: float | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"step": self.step, "timestamp": self.timestamp}
        if self.train_loss is not None:
            d["train_loss"] = self.train_loss
        if self.eval_loss is not None:
            d["eval_loss"] = self.eval_loss
        if self.learning_rate is not None:
            d["learning_rate"] = self.learning_rate
        if self.grad_norm is not None:
            d["grad_norm"] = self.grad_norm
        if self.epoch is not None:
            d["epoch"] = self.epoch
        if self.elapsed_seconds is not None:
            d["elapsed_seconds"] = self.elapsed_seconds
        return d


class MetricsTracker:
    """Tracks training metrics and writes to JSONL file."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[MetricEntry] = []
        self._start_time = time.time()

    def log(self, step: int, **kwargs) -> MetricEntry:
        elapsed = time.time() - self._start_time
        entry = MetricEntry(step=step, elapsed_seconds=round(elapsed, 2), **kwargs)
        self.entries.append(entry)
        # Append to JSONL
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def get_latest(self) -> MetricEntry | None:
        return self.entries[-1] if self.entries else None

    def get_all(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @property
    def best_eval_loss(self) -> float | None:
        eval_losses = [e.eval_loss for e in self.entries if e.eval_loss is not None]
        return min(eval_losses) if eval_losses else None

    @property
    def best_step(self) -> int | None:
        if not self.entries:
            return None
        best_eval = self.best_eval_loss
        if best_eval is None:
            return self.entries[-1].step
        for e in self.entries:
            if e.eval_loss == best_eval:
                return e.step
        return None


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Load metrics from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries
