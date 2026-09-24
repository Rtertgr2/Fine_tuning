"""Tests for the Phase 2 data pipeline."""

from __future__ import annotations

import json
import sqlite3
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Sandbox tests
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_create_workspace(self):
        from backend.tools.sandbox import Sandbox

        with Sandbox() as sb:
            assert sb.root().exists()
            assert sb.root().is_dir()

    def test_git_checkout(self):
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids

        with Sandbox() as sb:
            traj = Trajectory(run_id=ids.new_ulid())
            result = sb.run("git_checkout", {"repo": "test-repo", "branch": "main"}, traj=traj)
            assert "checked out" in result.lower()
            assert len(traj.records) == 1
            assert traj.records[0].tool == "git_checkout"

    def test_read_write_file(self):
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids

        with Sandbox() as sb:
            traj = Trajectory(run_id=ids.new_ulid())
            sb.run("write_file", {"path": "test.txt", "content": "hello world"}, traj=traj)
            result = sb.run("read_file", {"path": "test.txt"}, traj=traj)
            assert "hello world" in result

    def test_path_traversal_blocked(self):
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids

        with Sandbox() as sb:
            traj = Trajectory(run_id=ids.new_ulid())
            result = sb.run("read_file", {"path": "../etc/passwd"}, traj=traj)
            assert "Blocked by policy" in result

    def test_env_file_blocked(self):
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids

        with Sandbox() as sb:
            traj = Trajectory(run_id=ids.new_ulid())
            result = sb.run("write_file", {"path": ".env", "content": "SECRET=123"}, traj=traj)
            assert "Blocked by policy" in result

    def test_symlink_escape_blocked(self, tmp_path):
        from backend.tools.sandbox import Sandbox

        workspace = tmp_path / "workspace"
        outside = tmp_path / "outside.txt"
        workspace.mkdir()
        outside.write_text("do not expose", encoding="utf-8")
        try:
            (workspace / "escape.txt").symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable on this platform")
        with Sandbox(workspace_root=workspace) as sb:
            result = sb.run("read_file", {"path": "escape.txt"})
            assert "Blocked by policy" in result
            write_result = sb.run("write_file", {"path": "escape.txt", "content": "overwrite"})
            assert "Blocked by policy" in write_result
        assert outside.read_text(encoding="utf-8") == "do not expose"

    def test_unknown_tool_blocked(self):
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids

        with Sandbox() as sb:
            traj = Trajectory(run_id=ids.new_ulid())
            result = sb.run("delete_everything", {}, traj=traj)
            assert "Blocked by policy" in result
            assert traj.records[0].policy_blocked

    def test_trajectory_logging(self):
        from backend.tools.sandbox import Sandbox, Trajectory
        from backend.app import ids

        with Sandbox() as sb:
            traj = Trajectory(run_id=ids.new_ulid())
            sb.run("write_file", {"path": "a.txt", "content": "aaa"}, traj=traj)
            sb.run("read_file", {"path": "a.txt"}, traj=traj)
            sb.run("read_file", {"path": "b.txt"}, traj=traj)

            d = traj.to_dict()
            assert d["run_id"] == traj.run_id
            assert len(d["records"]) == 3
            assert d["records"][0]["tool"] == "write_file"
            assert d["records"][1]["tool"] == "read_file"
            assert d["records"][2]["tool"] == "read_file"


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestSecretScanning:
    def test_aws_key_detected(self):
        from backend.pipeline.validator import scan_secrets

        text = "AKIAIOSFODNN7EXAMPLE"
        findings = scan_secrets(text)
        assert len(findings) > 0
        assert findings[0][0] == "aws_access_key"

    def test_private_key_detected(self):
        from backend.pipeline.validator import scan_secrets

        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        findings = scan_secrets(text)
        assert any(f[0] == "private_key" for f in findings)

    def test_clean_text_no_findings(self):
        from backend.pipeline.validator import scan_secrets

        text = "Hello world, this is normal text with no secrets."
        findings = scan_secrets(text)
        assert len(findings) == 0


class TestContamination:
    def test_no_contamination(self):
        from backend.pipeline.validator import ContaminationChecker

        checker = ContaminationChecker(eval_examples=[])
        candidate = {
            "messages": [
                {"role": "user", "content": "unique text xyz"},
                {"role": "assistant", "content": "response abc"},
            ]
        }
        result = checker.check(candidate)
        assert not result.has_contamination

    def test_contamination_detected(self):
        from backend.pipeline.validator import ContaminationChecker

        eval_text = "eval text " * 50
        candidate_text = "eval text " * 40 + "unique " * 10

        checker = ContaminationChecker(
            eval_examples=[
                {
                    "messages": [
                        {"role": "user", "content": eval_text},
                        {"role": "assistant", "content": "resp"},
                    ]
                }
            ],
            threshold=0.3,
        )
        candidate = {
            "messages": [
                {"role": "user", "content": candidate_text},
                {"role": "assistant", "content": "resp"},
            ]
        }
        result = checker.check(candidate)
        assert result.has_contamination


class TestPipelineValidator:
    def test_clean_candidate_passes(self):
        from backend.pipeline.validator import PipelineValidator

        validator = PipelineValidator()
        candidate = {
            "category": "plan",
            "messages": [
                {"role": "user", "content": "unique task xyz 123"},
                {"role": "assistant", "content": '{"status": "APPROVED"}'},
            ],
        }
        ok, reasons = validator.validate(candidate)
        assert ok
        assert len(reasons) == 0

    def test_duplicate_detected(self):
        from backend.pipeline.validator import PipelineValidator

        validator = PipelineValidator()
        messages = [
            {"role": "user", "content": "task 1"},
            {"role": "assistant", "content": "response 1"},
        ]

        ok1, _ = validator.validate({"messages": messages, "category": "plan"})
        assert ok1

        ok2, reasons2 = validator.validate({"messages": messages, "category": "plan"})
        assert not ok2
        assert "exact_duplicate" in reasons2


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


class TestToolGenerator:
    def test_generates_template_trajectory(self, tmp_workbench):
        from backend.pipeline.tool_generator import ToolGenerator

        gen = ToolGenerator()
        seeds = [
            {
                "task": "read app.py and create output",
                "repo": "test-repo",
                "branch": "main",
                "expected_files": ["app.py"],
                "group_id": "gen/test-repo",
            }
        ]
        result = gen.generate(seeds)
        assert len(result.candidates) > 0

    def test_tool_example_has_three_calls(self, tmp_workbench):
        from backend.pipeline.tool_generator import ToolGenerator

        gen = ToolGenerator()
        seeds = [
            {
                "task": "test",
                "repo": "r",
                "branch": "main",
                "expected_files": ["a.py"],
                "group_id": "gen/r",
            }
        ]
        result = gen.generate(seeds)
        for cand in result.candidates:
            meta = cand.get("meta", {})
            tool_calls = meta.get("tool_calls", [])
            assert len(tool_calls) >= 3


class TestLoopGenerator:
    def test_generates_loop_example(self, tmp_workbench):
        from backend.pipeline.loop_generator import LoopGenerator

        gen = LoopGenerator(error_types=["file_not_found"])
        seeds = [
            {
                "task": "read missing file",
                "repo": "r",
                "branch": "main",
                "target_file": "src/app.py",
                "group_id": "gen/loop",
            }
        ]
        result = gen.generate(seeds)
        assert len(result.candidates) > 0

    def test_loop_has_failure(self, tmp_workbench):
        from backend.pipeline.loop_generator import LoopGenerator

        gen = LoopGenerator(error_types=["file_not_found"])
        seeds = [
            {
                "task": "read missing file",
                "repo": "r",
                "branch": "main",
                "target_file": "app.py",
                "group_id": "gen/loop",
            }
        ]
        result = gen.generate(seeds)
        for cand in result.candidates:
            messages = cand["messages"]
            # Must have a failure somewhere
            assert any("Error" in m.get("content", "") for m in messages)


class TestPlanGenerator:
    def test_generates_approved_plan(self, tmp_workbench):
        from backend.pipeline.plan_generator import PlanGenerator

        gen = PlanGenerator(clean_ratio=1.0)
        seeds = [{"plan": "test plan", "version": "v1.0.0"}]
        result = gen.generate(seeds)
        assert len(result.candidates) > 0
        assert result.candidates[0]["messages"][-1]["content"]  # has assistant response

    def test_generates_flawed_plan(self, tmp_workbench):
        from backend.pipeline.plan_generator import PlanGenerator

        gen = PlanGenerator(clean_ratio=0.0, flaw_types=["missing_rollback"])
        seeds = [{"plan": "test plan", "version": "v1.0.0"}]
        result = gen.generate(seeds)
        assert len(result.candidates) > 0
        response = json.loads(result.candidates[0]["messages"][-1]["content"])
        assert response["status"] == "NEEDS_REVISION"


class TestSecGenerator:
    def test_generates_vuln_review(self, tmp_workbench):
        from backend.pipeline.sec_generator import SecGenerator

        gen = SecGenerator(pass_ratio=0.0)
        seeds = [{"group_id": "gen/sec"}]
        result = gen.generate(seeds)
        assert len(result.candidates) > 0

    def test_generates_clean_review(self, tmp_workbench):
        from backend.pipeline.sec_generator import SecGenerator

        gen = SecGenerator(pass_ratio=1.0)
        seeds = [{"group_id": "gen/sec"}]
        result = gen.generate(seeds)
        assert len(result.candidates) > 0
        response = json.loads(result.candidates[0]["messages"][-1]["content"])
        assert response["status"] == "PASS"

    def test_vuln_review_has_correct_line(self, tmp_workbench):
        from backend.pipeline.sec_generator import SecGenerator

        gen = SecGenerator(pass_ratio=0.0, vuln_types=["sql_injection"])
        seeds = [{"group_id": "gen/sec"}]
        result = gen.generate(seeds)
        for cand in result.candidates:
            response = json.loads(cand["messages"][-1]["content"])
            for issue in response.get("issues", []):
                assert "line" in issue
                assert "type" in issue


# ---------------------------------------------------------------------------
# Review queue tests
# ---------------------------------------------------------------------------


class TestReviewQueue:
    def _conn(self, tmp_path):
        from backend.app.db import connect, init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        return connect(db_path)

    def test_list_pending_empty(self, tmp_path):
        from backend.app.services import review as svc

        conn = self._conn(tmp_path)
        try:
            total, items = svc.list_pending(conn)
            assert total == 0
            assert items == []
        finally:
            conn.close()

    def test_approve_moves_to_examples(self, tmp_path):
        from backend.app.services import review as svc
        from backend.app import ids
        from datetime import datetime, timezone

        conn = self._conn(tmp_path)
        try:
            # Insert a test item
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            item_id = ids.example_id()
            conn.execute(
                """INSERT INTO review_queue
                   (id, category, example_data_json, status, group_id, created_at, updated_at)
                   VALUES (?, 'plan', ?, 'pending', 'test', ?, ?)""",
                (item_id, json.dumps({
                    "messages": [
                        {"role": "user", "content": "test"},
                        {"role": "assistant", "content": '{"status": "APPROVED"}'},
                    ]
                }), ts, ts),
            )
            conn.commit()

            # Approve
            result = svc.approve(conn, item_id, reviewer="tester")
            assert result is not None
            assert result["status"] == "approved"
            assert result["reviewer"] == "tester"

            # Verify in examples
            row = conn.execute(
                "SELECT * FROM examples WHERE id != ?", (item_id,)
            ).fetchall()
            assert len(row) == 1
            assert row[0]["category"] == "plan"
        finally:
            conn.close()

    def test_reject_with_reason(self, tmp_path):
        from backend.app.services import review as svc
        from backend.app import ids
        from datetime import datetime, timezone

        conn = self._conn(tmp_path)
        try:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            item_id = ids.example_id()
            conn.execute(
                """INSERT INTO review_queue
                   (id, category, example_data_json, status, group_id, created_at, updated_at)
                   VALUES (?, 'tool', ?, 'pending', 'test', ?, ?)""",
                (item_id, json.dumps({"messages": []}), ts, ts),
            )
            conn.commit()

            result = svc.reject(conn, item_id, "bad quality", reviewer="tester")
            assert result is not None
            assert result["status"] == "rejected"
            assert result["reason"] == "bad quality"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Cost control tests
# ---------------------------------------------------------------------------


class TestBudget:
    def test_unlimited_budget(self):
        from backend.pipeline.cost import Budget

        b = Budget()
        assert b.can_spend(99999)
        b.record(1000, 500, 1.5)
        assert b.spent_usd == 1.5

    def test_budget_enforced(self):
        from backend.pipeline.cost import Budget

        b = Budget(max_usd=1.0)
        assert b.can_spend(0.5)
        b.record(1000, 500, 0.5)
        assert not b.can_spend(0.6)
        assert b.can_spend(0.5)

    def test_cache_tracking(self):
        from backend.pipeline.cost import Budget

        b = Budget()
        b.record_cache_hit()
        b.record_cache_hit()
        b.record_cache_miss()
        d = b.to_dict()
        assert d["cache_hits"] == 2
        assert d["cache_misses"] == 1


class TestResultCache:
    def test_put_and_get(self, tmp_path):
        from backend.pipeline.cost import ResultCache

        cache = ResultCache(cache_dir=str(tmp_path / "cache"))
        prompt = "test prompt"
        result = {"key": "value", "nested": {"a": 1}}

        cache.put(prompt, result)
        loaded = cache.get(prompt)
        assert loaded == result

    def test_miss(self, tmp_path):
        from backend.pipeline.cost import ResultCache

        cache = ResultCache(cache_dir=str(tmp_path / "cache"))
        assert cache.get("unknown prompt") is None

    def test_clear(self, tmp_path):
        from backend.pipeline.cost import ResultCache

        cache = ResultCache(cache_dir=str(tmp_path / "cache"))
        cache.put("a", {"x": 1})
        cache.put("b", {"y": 2})
        assert cache.clear() == 2
        assert cache.get("a") is None
