"""Tests for Phase 6: Active Learning."""
from __future__ import annotations

import json
import pytest
from backend.app import config
from backend.app.services import redaction, activity_log, case_detector, cases, metrics, active_learning
from backend.app.db import init_db, connect


def _tool_call(name, arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def _valid_loop_example(path="missing.py", recovered_path="src/app.py"):
    from backend.tools.registry import tool_schemas
    return {
        "category": "loop",
        "tools": tool_schemas(),
        "messages": [
            {"role": "system", "content": "You are a coding agent. Use tools and recover after errors."},
            {"role": "user", "content": "Read the requested Python file."},
            {"role": "assistant", "content": _tool_call("read_file", {"path": path})},
            {"role": "tool", "content": f"Error: no such file or directory: {path}"},
            {"role": "assistant", "content": _tool_call("read_file", {"path": recovered_path})},
            {"role": "tool", "content": "print('ready')\n"},
            {"role": "assistant", "content": "I found the file and read it successfully."},
        ],
        "source": "active_learning",
    }


def _valid_sec_example():
    return {
        "category": "sec",
        "messages": [
            {"role": "system", "content": "Review the patch and return JSON only."},
            {"role": "user", "content": "diff --git a/safe.py b/safe.py\n--- a/safe.py\n+++ b/safe.py\n@@ -0,0 +1,1 @@\n+def safe(): return True\n"},
            {"role": "assistant", "content": json.dumps({"status": "PASS", "issues": []})},
        ],
        "source": "active_learning",
    }


def _valid_tool_example(i):
    from backend.tools.registry import tool_schemas
    return {
        "category": "tool",
        "tools": tool_schemas(),
        "group_id": f"manual/group-{i}",
        "messages": [
            {"role": "system", "content": "You are a coding agent. Use tools to edit the repository."},
            {"role": "user", "content": f"Checkout repo {i}, read source{i}.py, and write output{i}.py."},
            {"role": "assistant", "content": _tool_call("git_checkout", {"repo": f"repo-{i}", "branch": "main"})},
            {"role": "tool", "content": f"checked out repo-{i} at main"},
            {"role": "assistant", "content": _tool_call("read_file", {"path": f"source{i}.py"})},
            {"role": "tool", "content": f"print({i})\n"},
            {"role": "assistant", "content": _tool_call("write_file", {"path": f"output{i}.py", "content": f"print({i})\n"})},
            {"role": "tool", "content": f"wrote output{i}.py"},
            {"role": "assistant", "content": f"Created output{i}.py with the source content."},
        ],
    }


@pytest.fixture()
def tmp_workbench(tmp_path, monkeypatch):
    """Isolated DB for tests."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "examples.db")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "datasets")
    init_db(tmp_path / "examples.db")
    return tmp_path


@pytest.fixture()
def conn(tmp_workbench):
    c = connect(tmp_workbench / "examples.db")
    yield c
    c.close()


# ── Redaction Tests ──────────────────────────────────────────────────────


class TestRedaction:
    def test_redacts_api_key(self):
        text = 'API key is "sk-abc123def456ghi789jkl012mno345pqr678stu901"'
        result = redaction.redact(text)
        assert result.had_secrets
        assert "[REDACTED_SK_KEY]" in result.text
        assert "sk-abc123" not in result.text

    def test_redacts_aws_key(self):
        text = "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE"
        result = redaction.redact(text)
        assert "[REDACTED_AWS_ACCESS_KEY]" in result.text

    def test_redacts_email(self):
        text = "Contact user@example.com for details"
        result = redaction.redact(text)
        assert "[REDACTED_EMAIL]" in result.text
        assert "user@example.com" not in result.text

    def test_redacts_password(self):
        text = "password=supersecret123"
        result = redaction.redact(text)
        assert "[REDACTED_PASSWORD]" in result.text

    def test_redacts_private_key(self):
        key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        text = f"Here is the key:\n{key}\nend"
        result = redaction.redact(text)
        assert "[REDACTED_PRIVATE_KEY]" in result.text

    def test_no_secrets_in_clean_text(self):
        text = "Hello, this is a normal message with no secrets."
        result = redaction.redact(text)
        assert not result.had_secrets
        assert result.redaction_count == 0
        assert result.text == text

    def test_redacts_multiple_secrets(self):
        text = "Key: sk-abc123def456ghi789jkl012mno345pqr678stu901 and email: test@x.com"
        result = redaction.redact(text)
        assert result.redaction_count >= 2

    def test_redact_dict(self):
        data = {"message": "key is sk-abc123def456ghi789jkl012mno345pqr678stu901"}
        cleaned, info = redaction.redact_dict(data)
        assert info is not None
        assert info.had_secrets
        assert "sk-abc123" not in cleaned["message"]

    def test_audit_map_has_placeholders(self):
        text = "email: user@test.com and key: sk-abc123def456ghi789jkl012mno345pqr678stu901"
        result = redaction.redact(text)
        assert len(result.redaction_map) > 0
        for placeholder in result.redaction_map.keys():
            assert placeholder.startswith("[REDACTED_")


# ── Activity Log Tests ──────────────────────────────────────────────────


class TestActivityLog:
    def test_insert_event(self, conn):
        activity_log.insert_event(conn, "s_test_1", "v0.1.0", "code", "tool_call",
                                  {"name": "read_file", "arguments": {"path": "a.py"}}, project="test-project")
        events = activity_log.get_events_for_session(conn, "s_test_1")
        assert len(events) == 1
        assert events[0]["model_version"] == "v0.1.0"
        assert events[0]["event"] == "tool_call"

    def test_insert_with_redaction(self, conn):
        activity_log.insert_event(conn, "s_test_2", "v0.1.0", "code", "message",
                                  {"content": "The API key is sk-abc123def456ghi789jkl012mno345pqr678stu901"})
        events = activity_log.get_events_for_session(conn, "s_test_2")
        assert "sk-abc123" not in json.dumps(events[0]["data"])
        assert events[0]["redaction_count"] > 0

    def test_insert_batch(self, conn):
        events = [{"session_id": "s_batch", "model_version": "v0.1.0", "state": "code",
                   "event": "tool_call", "data": {"name": f"call_{i}"}} for i in range(5)]
        assert activity_log.insert_batch(conn, events) == 5
        assert len(activity_log.get_events_for_session(conn, "s_batch")) == 5

    def test_invalid_state_rejected(self, conn):
        with pytest.raises(ValueError, match="invalid state"):
            activity_log.insert_event(conn, "s_x", "v0.1.0", "invalid_state", "message", {})

    def test_invalid_event_rejected(self, conn):
        with pytest.raises(ValueError, match="invalid event"):
            activity_log.insert_event(conn, "s_x", "v0.1.0", "code", "invalid_event", {})


# ── Case Detector Tests ─────────────────────────────────────────────────


class TestCaseDetector:
    def _make_events(self, session_id, events_data):
        return [{"session_id": session_id, "model_version": "v0.1.0",
                 "state": e.get("state", "code"), "event": e["event"], "data": e["data"],
                 "project": "test", "redaction_count": 0,
                 "ts": f"2026-09-20T10:00:{i:02d}+00:00", "id": i,
                 "created_at": "2026-09-20T10:00:00+00:00"} for i, e in enumerate(events_data)]

    def _insert_events(self, conn, events):
        for e in events:
            conn.execute(
                "INSERT INTO activity_log (session_id, ts, model_version, state, event, data, project, redaction_count, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (e["session_id"], e["ts"], e["model_version"], e["state"], e["event"],
                 json.dumps(e["data"]), e["project"], 0, e["created_at"]),
            )
        conn.commit()

    def test_detect_circuit_breaker(self, conn):
        events = self._make_events("s_det_cb", [{"event": "circuit_breaker", "data": {"reason": "max_retries"}}])
        self._insert_events(conn, events)
        cases_found = case_detector.detect_cases_for_session(conn, "s_det_cb")
        assert any(c.case_type == "loop" for c in cases_found)

    def test_detect_repeated_tool_call(self, conn):
        events = self._make_events("s_det_rep", [
            {"event": "tool_call", "data": {"name": "read_file", "arguments": {"path": "a.py"}}},
            {"event": "tool_result", "data": {"content": "file content"}},
            {"event": "tool_call", "data": {"name": "read_file", "arguments": {"path": "a.py"}}},
        ])
        self._insert_events(conn, events)
        detected = case_detector.detect_cases_for_session(conn, "s_det_rep")
        assert any(c.case_type == "loop" and "Repeated" in c.signal for c in detected)

    def test_detect_malformed_json(self, conn):
        events = self._make_events("s_det_json", [{"event": "tool_call", "data": {"name": "", "arguments": "not_json"}}])
        self._insert_events(conn, events)
        assert any(c.case_type == "syntax" for c in case_detector.detect_cases_for_session(conn, "s_det_json"))

    def test_detect_same_call_after_tool_error(self, conn):
        repeated = _tool_call("read_file", {"path": "a.py"})
        events = self._make_events("s_det_no_change", [
            {"event": "tool_call", "data": {"name": "read_file", "arguments": {"path": "a.py"}}},
            {"event": "tool_result", "data": {"content": "Error: not found"}},
            {"event": "message", "data": {"role": "assistant", "content": repeated}},
        ])
        self._insert_events(conn, events)
        detected = case_detector.detect_cases_for_session(conn, "s_det_no_change")
        assert any(c.case_type == "loop" and "No strategy change" in c.signal for c in detected)

    def test_detect_cancel(self, conn):
        events = self._make_events("s_det_cancel", [{"event": "cancel", "data": {"reason": "user stopped"}}])
        self._insert_events(conn, events)
        assert any(c.case_type == "cancel" for c in case_detector.detect_cases_for_session(conn, "s_det_cancel"))

    def test_detect_success(self, conn):
        events = self._make_events("s_det_ok", [
            {"event": "message", "data": {"role": "assistant", "content": "Done"}},
            {"event": "outcome", "data": {"success": True}},
        ])
        self._insert_events(conn, events)
        assert any(c.case_type == "success" for c in case_detector.detect_cases_for_session(conn, "s_det_ok"))

    def test_scan_all_sessions(self, conn):
        for sid in ["s_scan_1", "s_scan_2"]:
            self._insert_events(conn, self._make_events(sid, [{"event": "cancel", "data": {"reason": "user stopped"}}]))
        assert len(case_detector.scan_all_sessions(conn, sample_success_rate=0.0)) >= 2


# ── Case Queue Tests ────────────────────────────────────────────────────


class TestCaseQueue:
    def _make_case(self, session_id="s_q", case_type="loop"):
        from backend.app.services.case_detector import DetectedCase
        return DetectedCase(session_id=session_id, model_version="v0.1.0", case_type=case_type,
                            priority=4, signal="Test", events=[{"event": "circuit_breaker"}])

    def test_enqueue_and_list(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case())
        total, items = cases.list_cases(conn)
        assert case_id is not None and total == 1 and items[0]["case_type"] == "loop"

    def test_get_case_with_timeline(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case("s_timeline"))
        stored = cases.get_case(conn, case_id)
        assert stored is not None and len(stored["timeline"]) == 1

    def test_start_editing(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case("s_edit"))
        updated = cases.start_editing(conn, case_id, "reviewer_alice")
        assert updated is not None and updated["status"] == "editing" and updated["reviewer"] == "reviewer_alice"

    def test_approve_case(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case("s_app"))
        approved = cases.approve_case(conn, case_id, final_example=_valid_loop_example())
        assert approved is not None and approved["status"] == "approved"
        assert json.loads(approved["edited_example_json"])["source"] == "active_learning"

    def test_approve_sec_requires_second_reviewer(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case("s_sec", "sec"))
        with pytest.raises(ValueError, match="second_reviewer"):
            cases.approve_case(conn, case_id, final_example=_valid_sec_example())
        approved = cases.approve_case(conn, case_id, final_example=_valid_sec_example(), second_reviewer="reviewer_bob")
        assert approved is not None and approved["status"] == "approved"

    def test_approve_rejects_invalid_example(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case("s_invalid"))
        with pytest.raises(ValueError, match="schema|shared validators"):
            cases.approve_case(conn, case_id, final_example={"category": "loop", "messages": []})

    def test_approval_redacts_secret_before_persisting(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case("s_secret"))
        example = _valid_loop_example()
        example["messages"][1]["content"] += " sk-123456789012345678901234567890123456"
        approved = cases.approve_case(conn, case_id, final_example=example)
        assert approved is not None
        assert "sk-123456" not in approved["edited_example_json"]
        assert "[REDACTED_SK_KEY]" in approved["edited_example_json"]

    def test_reject_case(self, conn):
        case_id = cases.enqueue_case(conn, self._make_case("s_rej"))
        rejected = cases.reject_case(conn, case_id, "Not useful")
        assert rejected is not None and rejected["status"] == "rejected"

    def test_filter_cases(self, conn):
        for i, ct in enumerate(["loop", "syntax", "loop"]):
            cases.enqueue_case(conn, self._make_case(f"s_f_{i}", ct))
        total, items = cases.list_cases(conn, case_type="loop")
        assert total == 2 and all(item["case_type"] == "loop" for item in items)


# ── Metrics Tests ───────────────────────────────────────────────────────


class TestMetrics:
    def test_empty_metrics(self, conn):
        m = metrics.compute_metrics(conn)
        assert "circuit_breaker_rate" in m
        assert "tool_error_metrics" in m
        assert "queue_metrics" in m
        assert m["queue_metrics"]["total_cases"] == 0

    def test_circuit_breaker_rate_metric(self, conn):
        activity_log.insert_event(conn, "s_m_1", "v0.1.0", "code", "tool_call", {})
        activity_log.insert_event(conn, "s_m_1", "v0.1.0", "code", "circuit_breaker", {})
        activity_log.insert_event(conn, "s_m_2", "v0.1.0", "code", "tool_call", {})
        m = metrics.compute_metrics(conn)
        v_metrics = m["circuit_breaker_rate"].get("v0.1.0", {})
        assert v_metrics.get("total_sessions") == 2
        assert v_metrics.get("circuit_breaker_events") == 1
        assert v_metrics.get("rate_per_100_sessions") == 50.0

    def test_queue_metrics(self, conn):
        for i in range(5):
            cases.enqueue_case(conn, self._make_case(f"s_qm_{i}"))
        q = metrics.compute_metrics(conn)["queue_metrics"]
        assert q["total_cases"] == 5 and q["pending"] == 5

    def _make_case(self, session_id="s_qm", case_type="loop"):
        from backend.app.services.case_detector import DetectedCase
        return DetectedCase(session_id=session_id, model_version="v0.1.0", case_type=case_type,
                            priority=3, signal="Test", events=[])

    def test_case_type_distribution(self, conn):
        for i, ct in enumerate(["loop", "loop", "syntax", "cancel"]):
            cases.enqueue_case(conn, self._make_case(f"s_ct_{i}", ct))
        dist = metrics.compute_metrics(conn)["case_type_distribution"]
        assert dist.get("loop") == 2 and dist.get("syntax") == 1 and dist.get("cancel") == 1


# ── Active Learning Dataset Tests ────────────────────────────────────────


class TestActiveLearningDataset:
    def test_threshold_not_met(self, conn):
        count, met = active_learning.check_threshold(conn)
        assert count == 0 and not met

    def test_build_dataset_requires_approved(self, conn):
        with pytest.raises(ValueError, match="no approved"):
            active_learning.build_al_dataset(conn)

    def test_regression_split_holds_out_whole_groups(self):
        examples = [
            {"id": "orig-a", "source": "manual", "group_id": "repo/a"},
            {"id": "orig-b", "source": "manual", "group_id": "repo/b"},
            {"id": "al-1", "source": "active_learning", "group_id": "session/1"},
            {"id": "al-2", "source": "active_learning", "group_id": "session/1"},
            {"id": "al-3", "source": "active_learning", "group_id": "session/2"},
        ]
        train, val, regression = active_learning._split_with_regression(examples, 0.25, 7)
        train_groups = {item.get("group_id") for item in train}
        regression_groups = {item.get("group_id") for item in regression}
        assert regression
        assert not (train_groups & regression_groups)
        assert len(val) >= 1

    def test_build_dataset_with_approved_cases(self, conn):
        from backend.app.services.examples import create_example
        from backend.app import schemas
        from backend.app.services.case_detector import DetectedCase

        for i in range(4):
            example = _valid_tool_example(i)
            create_example(conn, schemas.ExampleIn(**example))
        rows = conn.execute("SELECT id FROM examples").fetchall()
        for row in rows:
            conn.execute("UPDATE examples SET status='approved' WHERE id=?", (row["id"],))
        conn.commit()

        case = DetectedCase(session_id="s_al_test", model_version="v0.1.0", case_type="loop",
                            priority=4, signal="Test", events=[])
        case_id = cases.enqueue_case(conn, case)
        cases.approve_case(conn, case_id, final_example=_valid_loop_example("missing.py", "src/missing.py"))

        record = active_learning.build_al_dataset(conn, name="test_al")
        assert record["manifest"]["active_learning"]["al_case_count"] == 1
        assert record["manifest"]["active_learning"]["al_example_count"] == 1
        assert record["manifest"]["files"]["train"]["sha256"]
        assert record["manifest"]["files"]["val"]["sha256"]
        assert record["manifest"]["counts"]["total"] == 5


# ── Integration Test ─────────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline(self, conn):
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "tool_call",
                                  {"name": "read_file", "arguments": {"path": "a.py"}})
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "tool_result", {"content": "Error: not found"})
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "tool_call",
                                  {"name": "read_file", "arguments": {"path": "a.py"}})
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "circuit_breaker", {"reason": "max retries"})
        detected = case_detector.scan_all_sessions(conn, sample_success_rate=0.0)
        assert len(detected) >= 1
        case_ids = cases.enqueue_batch(conn, detected)
        assert len(case_ids) >= 1

        cases.approve_case(conn, case_ids[0], final_example=_valid_loop_example("a.py", "src/a.py"))
        m = metrics.compute_metrics(conn)
        assert m["queue_metrics"]["total_cases"] >= 1
        assert m["queue_metrics"]["approved"] >= 1
''