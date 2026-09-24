"""Base class for data generators (T2.4-T2.7).

Each generator:
- Takes seed tasks (repos, prompts, or gold-set examples)
- Produces candidate examples
- Runs Phase 1 validators to filter
- Outputs approved candidates to the review queue
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from backend.adapters.base import content_hash
from backend.app import ids
from backend.app.db import connect
from backend.validators.base import Example as FwExample
from backend.validators.registry import Report, make_context, validate_example


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class GenerationResult:
    """Result of a generation run."""

    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    token_count: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


class BaseGenerator(abc.ABC):
    """Abstract base for all data generators.

    Subclasses implement `_generate_candidates` to produce raw examples.
    The base class handles validation, dedup, and review-queue insertion.
    """

    category: str = ""
    source: str = "generated"

    def __init__(self, db_path: str | None = None, budget_usd: float | None = None):
        self._db_path = db_path
        self._budget_usd = budget_usd

    @abc.abstractmethod
    def _generate_candidates(
        self,
        seeds: Sequence[dict[str, Any]],
        trajectory_factory: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Produce candidate examples from seeds.

        Each candidate is a dict with keys:
          - category: str
          - messages: list[dict]
          - tools: list[dict] | None
          - group_id: str | None
          - meta: dict (generator-specific metadata)
        """
        ...

    def generate(
        self,
        seeds: Sequence[dict[str, Any]],
        trajectory_factory: Any | None = None,
    ) -> GenerationResult:
        """Generate candidates, validate, and enqueue valid ones."""
        import time

        t0 = time.monotonic()
        result = GenerationResult()

        candidates = self._generate_candidates(seeds, trajectory_factory)
        result.candidates = list(candidates)

        conn = connect(self._db_path)
        try:
            existing = self._load_existing(conn)
            from backend.app import config
            from backend.pipeline.validator import ContaminationChecker, PipelineValidator

            contamination = ContaminationChecker.from_eval_dir(
                threshold=config.EVAL_OVERLAP_THRESHOLD
            )
            pipeline_validator = PipelineValidator(
                contamination_checker=contamination,
                existing_hashes={example.content_hash for example in existing if example.content_hash},
                near_dup_threshold=config.NEAR_DUP_THRESHOLD,
            )

            for cand in candidates:
                duplicate = self._is_duplicate(cand, existing)
                pipeline_ok, pipeline_reasons = pipeline_validator.validate(cand)
                if duplicate or not pipeline_ok:
                    reasons = (["exact_duplicate"] if duplicate else []) + pipeline_reasons
                    result.rejected.append(
                        {**cand, "reason": "; ".join(sorted(set(reasons))), "meta": cand.get("meta", {})}
                    )
                    continue

                report = self._validate(cand, peers=existing)
                if not report.ok:
                    result.rejected.append(
                        {
                            **cand,
                            "reason": "validation_failed",
                            "violations": [v.code for v in report.errors],
                            "meta": cand.get("meta", {}),
                        }
                    )
                    continue

                self._enqueue(conn, cand)
                # Make accepted candidates visible to later candidates in the
                # same batch so exact and near duplicates are filtered too.
                h = content_hash(cand["messages"], cand.get("tools"))
                existing.append(FwExample(
                    id=f"batch:{len(existing)}", category=cand["category"],
                    messages=cand["messages"], tools=cand.get("tools"),
                    source=self.source, group_id=cand.get("group_id"),
                    status="draft", content_hash=h,
                ))
        finally:
            conn.close()

        result.duration_seconds = time.monotonic() - t0
        return result

    def _load_existing(self, conn) -> list[FwExample]:
        """Load existing framework-level examples for dedup."""
        from backend.app.services.examples import all_framework

        return all_framework(conn)

    def _is_duplicate(self, cand: dict[str, Any], existing: Sequence[FwExample]) -> bool:
        """Hash-based dedup against existing examples."""
        h = content_hash(cand["messages"], cand.get("tools"))
        for ex in existing:
            if ex.content_hash == h:
                return True
        return False

    def _validate(
        self,
        cand: dict[str, Any],
        peers: Sequence[FwExample] | None = None,
    ) -> Report:
        """Run the shared Phase 1 validators against stored and batch peers."""
        fw = FwExample(
            id=f"candidate:{content_hash(cand['messages'], cand.get('tools'))[:12]}",
            category=cand["category"],
            messages=cand["messages"],
            tools=cand.get("tools"),
            source=self.source,
            group_id=cand.get("group_id"),
            status="draft",
            content_hash=content_hash(cand["messages"], cand.get("tools")),
        )
        if peers is None:
            conn = connect(self._db_path)
            try:
                peers = self._load_existing(conn)
            finally:
                conn.close()
        ctx = make_context(peers=peers)
        return validate_example(fw, ctx)

    def _enqueue(self, conn, cand: dict[str, Any]) -> str:
        """Insert into review_queue with status=pending."""
        ex_id = ids.example_id()
        ts = now_iso()
        example_data = {
            "category": cand["category"],
            "messages": cand["messages"],
            "tools": cand.get("tools"),
            "source": self.source,
            "group_id": cand.get("group_id"),
            "meta": cand.get("meta", {}),
        }
        conn.execute(
            """INSERT INTO review_queue
               (id, category, example_data_json, status, reason, reviewer,
                group_id, created_at, updated_at)
               VALUES (?, ?, ?, 'pending', NULL, NULL, ?, ?, ?)""",
            (
                ex_id,
                cand["category"],
                json.dumps(example_data, ensure_ascii=False),
                cand.get("group_id"),
                ts,
                ts,
            ),
        )
        conn.commit()
        return ex_id
