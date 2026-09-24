"""Validation pipeline (T2.8): secret scanning, dedup, contamination check.

Runs after generation and before review-queue insertion:
- Secret scanning (detect-secrets patterns)
- Deduplication (hash + near-duplicate)
- Contamination check against eval suites (Phase 4)
- Distribution tracking
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from backend.adapters.base import content_hash, jaccard, shingles
from backend.validators.common import SECRET_PATTERNS

# ---------------------------------------------------------------------------
# Secret scanning (uses canonical patterns from backend.validators.common)
# ---------------------------------------------------------------------------

def scan_secrets(text: str) -> list[tuple[str, str]]:
    """Return list of (pattern_name, matched_text) found in text."""
    findings: list[tuple[str, str]] = []
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append((name, match.group(0)[:20] + "…"))
    return findings


# ---------------------------------------------------------------------------
# Contamination checker
# ---------------------------------------------------------------------------


@dataclass
class ContaminationResult:
    has_contamination: bool
    overlaps: list[dict[str, Any]] = field(default_factory=list)


class ContaminationChecker:
    """Checks generated examples against eval suites for n-gram overlap."""

    def __init__(self, eval_examples: Sequence[dict[str, Any]] = (), threshold: float = 0.3):
        self._eval_shingles: dict[str, set[str]] = {}
        self._threshold = threshold
        for ex in eval_examples:
            text = self._extract_text(ex)
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            self._eval_shingles[h] = shingles(text)

    def _extract_text(self, example: dict[str, Any]) -> str:
        """Concatenate the example's text content.

        Supports stored examples (messages[]) and frozen eval-suite cases
        (single `prompt` field).
        """
        if isinstance(example.get("messages"), list) and example["messages"]:
            return " ".join(m.get("content", "") or "" for m in example["messages"])
        if example.get("prompt"):
            return str(example["prompt"])
        return json.dumps(example, ensure_ascii=False, sort_keys=True)

    def check(self, candidate: dict[str, Any]) -> ContaminationResult:
        """Check if a candidate overlaps significantly with eval data."""
        cand_text = self._extract_text(candidate)
        cand_shingles = shingles(cand_text)
        if not cand_shingles:
            return ContaminationResult(has_contamination=False)

        overlaps: list[dict[str, Any]] = []
        for eval_h, eval_sh in self._eval_shingles.items():
            score = jaccard(cand_shingles, eval_sh)
            if score >= self._threshold:
                overlaps.append({"eval_hash": eval_h, "score": score})

        return ContaminationResult(
            has_contamination=len(overlaps) > 0,
            overlaps=overlaps,
        )

    @classmethod
    def from_eval_dir(cls, eval_dir: str | None = None, **kwargs) -> "ContaminationChecker":
        """Load eval suites from the eval/ directory (Phase 4)."""
        from pathlib import Path
        from backend.app import config

        eval_path = Path(eval_dir) if eval_dir else config.EVAL_DIR / "suites"
        eval_examples: list[dict[str, Any]] = []

        if eval_path.exists():
            files = sorted(set(eval_path.glob("**/*.jsonl")))
            files += sorted(set(eval_path.glob("**/*.json")))
            for suite_file in files:
                try:
                    text = suite_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                if suite_file.suffix == ".jsonl":
                    for line in text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            eval_examples.append(obj)
                else:
                    try:
                        suite = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(suite, list):
                        eval_examples.extend(e for e in suite if isinstance(e, dict))
                    elif isinstance(suite, dict) and isinstance(suite.get("examples"), list):
                        eval_examples.extend(e for e in suite["examples"] if isinstance(e, dict))

        return cls(eval_examples=eval_examples, **kwargs)


# ---------------------------------------------------------------------------
# Distribution tracker
# ---------------------------------------------------------------------------


@dataclass
class DistributionStats:
    by_category: Counter[str] = field(default_factory=Counter)
    by_source: Counter[str] = field(default_factory=Counter)
    by_language: Counter[str] = field(default_factory=Counter)
    by_vuln_type: Counter[str] = field(default_factory=Counter)
    total: int = 0

    def record(self, example: dict[str, Any]) -> None:
        self.by_category[example.get("category", "unknown")] += 1
        self.by_source[example.get("source", "unknown")] += 1
        meta = example.get("meta", {})
        lang = meta.get("language") or meta.get("lang")
        if lang:
            self.by_language[lang] += 1
        vuln_type = meta.get("vulnerability_type") or meta.get("injected_flaw")
        if vuln_type:
            self.by_vuln_type[vuln_type] += 1
        self.total += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_category": dict(self.by_category),
            "by_source": dict(self.by_source),
            "by_language": dict(self.by_language),
            "by_vuln_type": dict(self.by_vuln_type),
        }


# ---------------------------------------------------------------------------
# Pipeline validator
# ---------------------------------------------------------------------------


class PipelineValidator:
    """Combined validation for the data generation pipeline."""

    def __init__(
        self,
        contamination_checker: ContaminationChecker | None = None,
        existing_hashes: set[str] | None = None,
        near_dup_threshold: float = 0.85,
    ):
        self._contamination = contamination_checker
        self._existing_hashes: set[str] = existing_hashes or set()
        self._near_dup_threshold = near_dup_threshold
        self._distribution = DistributionStats()

    @property
    def distribution(self) -> DistributionStats:
        return self._distribution

    def validate(self, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a candidate. Returns (ok, reasons)."""
        reasons: list[str] = []

        # 1. Secret scan every persisted surface, including optional tool schema.
        text = self._extract_text(candidate)
        if candidate.get("tools") is not None:
            text += " " + json.dumps(candidate.get("tools"), ensure_ascii=False)
        secrets = scan_secrets(text)
        if secrets:
            reasons.append(f"secret_detected: {[s[0] for s in secrets]}")

        # 2. Hash-based dedup
        h = content_hash(candidate.get("messages", []), candidate.get("tools"))
        if h in self._existing_hashes:
            reasons.append("exact_duplicate")
        else:
            self._existing_hashes.add(h)

        # 3. Contamination check
        if self._contamination is not None:
            result = self._contamination.check(candidate)
            if result.has_contamination:
                reasons.append(f"contamination: {result.overlaps}")

        # 4. Record for distribution tracking
        self._distribution.record(candidate)

        return (len(reasons) == 0, reasons)

    def _extract_text(self, example: dict[str, Any]) -> str:
        messages = example.get("messages", [])
        return " ".join(m.get("content", "") or "" for m in messages)
