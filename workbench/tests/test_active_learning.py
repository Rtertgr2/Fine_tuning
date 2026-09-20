"""Tests for Phase 6: Active Learning."""
from __future__ import annotations

import json
import pytest
from backend.app import config
from backend.app.services import redaction, activity_log, case_detector, cases, metrics, active_learning
from backend.app.db import init_db, connect


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
        # redaction_map maps placeholder -> hash
        assert len(result.redaction_map) > 0
        for placeholder in result.redaction_map.keys():
            assert placeholder.startswith("[REDACTED_")


# ── Activity Log Tests ──────────────────────────────────────────────────


class TestActivityLog:
    def test_insert_event(self, conn):
        activity_log.insert_event(
            conn,
            session_id="s_test_1",
            model_version="v0.1.0",
            state="code",
            event="tool_call",
            data={"name": "read_file", "arguments": {"path": "a.py"}},
            project="test-project",
        )
        events = activity_log.get_events_for_session(conn, "s_test_1")
        assert len(events) == 1
        assert events[0]["model_version"] == "v0.1.0"
        assert events[0]["event"] == "tool_call"

    def test_insert_with_redaction(self, conn):
        activity_log.insert_event(
            conn,
            session_id="s_test_2",
            model_version="v0.1.0",
            state="code",
            event="message",
            data={"content": "The API key is sk-abc123def456ghi789jkl012mno345pqr678stu901"},
        )
        events = activity_log.get_events_for_session(conn, "s_test_2")
        data = events[0]["data"]
        assert "sk-abc123" not in json.dumps(data)
        assert events[0]["redaction_count"] > 0

    def test_insert_batch(self, conn):
        events = [
            {"session_id": "s_batch", "model_version": "v0.1.0", "state": "code",
             "event": "tool_call", "data": {"name": f"call_{i}"}}
            for i in range(5)
        ]
        count = activity_log.insert_batch(conn, events)
        assert count == 5
        stored = activity_log.get_events_for_session(conn, "s_batch")
        assert len(stored) == 5

    def test_invalid_state_rejected(self, conn):
        with pytest.raises(ValueError, match="invalid state"):
            activity_log.insert_event(
                conn, "s_x", "v0.1.0", "invalid_state", "message", {}
            )

    def test_invalid_event_rejected(self, conn):
        with pytest.raises(ValueError, match="invalid event"):
            activity_log.insert_event(
                conn, "s_x", "v0.1.0", "code", "invalid_event", {}
            )


# ── Case Detector Tests ─────────────────────────────────────────────────


class TestCaseDetector:
    def _make_events(self, session_id, events_data):
        """Helper to build events for a session."""
        return [
            {
                "session_id": session_id,
                "model_version": "v0.1.0",
                "state": e.get("state", "code"),
                "event": e["event"],
                "data": e["data"],
                "project": "test",
                "redaction_count": 0,
                "ts": f"2026-09-20T10:00:{i:02d}+00:00",
                "id": i,
                "created_at": "2026-09-20T10:00:00+00:00",
            }
            for i, e in enumerate(events_data)
        ]

    def _insert_events(self, conn, events):
        for e in events:
            conn.execute(
                "INSERT INTO activity_log (session_id, ts, model_version, state, event, data, project, redaction_count, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (e["session_id"], e["ts"], e["model_version"], e["state"], e["event"],
                 json.dumps(e["data"]), e["project"], 0, e["created_at"]),
            )
        conn.commit()

    def test_detect_circuit_breaker(self, conn):
        events = self._make_events("s_det_cb", [
            {"event": "circuit_breaker", "data": {"reason": "max_retries"}},
        ])
        self._insert_events(conn, events)

        cases = case_detector.detect_cases_for_session(conn, "s_det_cb")
        assert len(cases) >= 1
        assert any(c.case_type == "loop" for c in cases)

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
        events = self._make_events("s_det_json", [
            {"event": "tool_call", "data": {"name": "", "arguments": "not_json"}},
        ])
        self._insert_events(conn, events)

        detected = case_detector.detect_cases_for_session(conn, "s_det_json")
        # Should at minimum not crash
        assert isinstance(detected, list)

    def test_detect_cancel(self, conn):
        events = self._make_events("s_det_cancel", [
            {"event": "cancel", "data": {"reason": "user stopped"}},
        ])
        self._insert_events(conn, events)

        detected = case_detector.detect_cases_for_session(conn, "s_det_cancel")
        assert any(c.case_type == "cancel" for c in detected)

    def test_detect_success(self, conn):
        events = self._make_events("s_det_ok", [
            {"event": "message", "data": {"role": "assistant", "content": "Done"}},
            {"event": "outcome", "data": {"success": True}},
        ])
        self._insert_events(conn, events)

        detected = case_detector.detect_cases_for_session(conn, "s_det_ok")
        assert any(c.case_type == "success" for c in detected)

    def test_scan_all_sessions(self, conn):
        for sid in ["s_scan_1", "s_scan_2"]:
            events = self._make_events(sid, [
                {"event": "cancel", "data": {"reason": "user stopped"}},
            ])
            self._insert_events(conn, events)

        found = case_detector.scan_all_sessions(conn, sample_success_rate=0.0)
        assert len(found) >= 2


# ── Case Queue Tests ────────────────────────────────────────────────────


class TestCaseQueue:
    def _make_case(self, session_id="s_q", case_type="loop"):
        from backend.app.services.case_detector import DetectedCase
        return DetectedCase(
            session_id=session_id, model_version="v0.1.0", case_type=case_type,
            priority=4, signal="Test", events=[{"event": "circuit_breaker"}],
        )

    def test_enqueue_and_list(self, conn):
        case = self._make_case()
        case_id = cases.enqueue_case(conn, case)
        assert case_id is not None

        total, items = cases.list_cases(conn)
        assert total == 1
        assert items[0]["case_type"] == "loop"

    def test_get_case_with_timeline(self, conn):
        case = self._make_case(session_id="s_timeline")
        case_id = cases.enqueue_case(conn, case)
        stored = cases.get_case(conn, case_id)
        assert stored is not None
        assert "timeline" in stored
        assert len(stored["timeline"]) == 1

    def test_start_editing(self, conn):
        case = self._make_case(session_id="s_edit")
        case_id = cases.enqueue_case(conn, case)
        updated = cases.start_editing(conn, case_id, "reviewer_alice")
        assert updated is not None
        assert updated["status"] == "editing"
        assert updated["reviewer"] == "reviewer_alice"

    def test_approve_case(self, conn):
        case = self._make_case(session_id="s_app")
        case_id = cases.enqueue_case(conn, case)
        approved = cases.approve_case(conn, case_id)
        assert approved is not None
        assert approved["status"] == "approved"

    def test_approve_sec_requires_second_reviewer(self, conn):
        case = self._make_case(session_id="s_sec", case_type="sec")
        case_id = cases.enqueue_case(conn, case)
        with pytest.raises(ValueError, match="second_reviewer"):
            cases.approve_case(conn, case_id)
        approved = cases.approve_case(conn, case_id, second_reviewer="reviewer_bob")
        assert approved is not None
        assert approved["status"] == "approved"

    def test_reject_case(self, conn):
        case = self._make_case(session_id="s_rej")
        case_id = cases.enqueue_case(conn, case)
        rejected = cases.reject_case(conn, case_id, "Not useful")
        assert rejected is not None
        assert rejected["status"] == "rejected"
        assert rejected["rejection_reason"] == "Not useful"

    def test_filter_cases(self, conn):
        for i, ct in enumerate(["loop", "syntax", "loop"]):
            case = self._make_case(session_id=f"s_f_{i}", case_type=ct)
            cases.enqueue_case(conn, case)

        total, items = cases.list_cases(conn, case_type="loop")
        assert total == 2
        assert all(i["case_type"] == "loop" for i in items)


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
            case = self._make_case(session_id=f"s_qm_{i}")
            cases.enqueue_case(conn, case)

        m = metrics.compute_metrics(conn)
        q = m["queue_metrics"]
        assert q["total_cases"] == 5
        assert q["pending"] == 5

    def _make_case(self, session_id="s_qm", case_type="loop"):
        from backend.app.services.case_detector import DetectedCase
        return DetectedCase(
            session_id=session_id, model_version="v0.1.0", case_type=case_type,
            priority=3, signal="Test", events=[],
        )

    def test_case_type_distribution(self, conn):
        for i, ct in enumerate(["loop", "loop", "syntax", "cancel"]):
            case = self._make_case(session_id=f"s_ct_{i}", case_type=ct)
            cases.enqueue_case(conn, case)

        m = metrics.compute_metrics(conn)
        dist = m["case_type_distribution"]
        assert dist.get("loop") == 2
        assert dist.get("syntax") == 1
        assert dist.get("cancel") == 1


# ── Active Learning Dataset Tests ────────────────────────────────────────


class TestActiveLearningDataset:
    def test_threshold_not_met(self, conn):
        count, met = active_learning.check_threshold(conn)
        assert count == 0
        assert not met

    def test_build_dataset_requires_approved(self, conn):
        with pytest.raises(ValueError, match="no approved"):
            active_learning.build_al_dataset(conn)

    def test_build_dataset_with_approved_cases(self, conn):
        # Add an approved original example (will auto-get tools from registry)
        from backend.app.services.examples import create_example
        from backend.app import schemas
        from backend.app.services.case_detector import DetectedCase

        original_msg = [
            schemas.Message(role="system", content="You are a coding agent."),
            schemas.Message(role="user", content="read a.py"),
            schemas.Message(role="assistant", content="<tool_call>read_file a.py</tool_call>"),
            schemas.Message(role="tool", content="print('hi')"),
            schemas.Message(role="assistant", content="Done"),
        ]
        create_example(conn, schemas.ExampleIn(
            category="tool",
            messages=original_msg,
            source="manual",
        ))

        # Approve it
        rows = conn.execute("SELECT id, content_hash FROM examples").fetchall()
        for r in rows:
            conn.execute("UPDATE examples SET status='approved' WHERE id=?", (r["id"],))
        conn.commit()

        # Create a loop example (different from the tool example above)
        al_example = {
            "category": "loop",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "read missing.py"},
                {"role": "assistant", "content": "<tool_call>read_file missing.py</tool_call>"},
                {"role": "tool", "content": "Error: not found"},
                {"role": "assistant", "content": "<tool_call>read_file other.py</tool_call>"},
                {"role": "tool", "content": "Error: not found"},
                {"role": "assistant", "content": "I cannot find the file. Let me check what exists."},
            ],
            "source": "active_learning",
        }

        c = DetectedCase(
            session_id="s_al_test", model_version="v0.1.0", case_type="loop",
            priority=4, signal="Test", events=[],
        )
        case_id = cases.enqueue_case(conn, c)
        cases.approve_case(conn, case_id, final_example=al_example)

        # Build dataset
        record = active_learning.build_al_dataset(conn, name="test_al")
        assert "id" in record
        assert "manifest" in record
        assert record["manifest"]["active_learning"]["al_case_count"] == 1


# ── Integration Test ─────────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline(self, conn):
        """End-to-end: log events → detect cases → approve → build dataset."""
        # Step 1: Log a failing session
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "tool_call",
                                  {"name": "read_file", "arguments": {"path": "a.py"}})
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "tool_result",
                                  {"content": "Error: not found"})
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "tool_call",
                                  {"name": "read_file", "arguments": {"path": "a.py"}})
        activity_log.insert_event(conn, "s_int", "v0.1.0", "code", "circuit_breaker",
                                  {"reason": "max retries"})

        # Step 2: Scan and detect cases
        detected = case_detector.scan_all_sessions(conn, sample_success_rate=0.0)
        assert len(detected) >= 1

        # Step 3: Enqueue
        case_ids = cases.enqueue_batch(conn, detected)
        assert len(case_ids) >= 1

        # Step 4: Approve one
        al_example = {
            "id": "ex_int",
            "category": "loop",
            "messages": [
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": "read a.py"},
                {"role": "assistant", "content": "<tool_call>read_file a.py</tool_call>"},
                {"role": "tool", "content": "Error: not found"},
                {"role": "assistant", "content": "<tool_call>read_file src/a.py</tool_call>"},
                {"role": "tool", "content": "content here"},
                {"role": "assistant", "content": "Found it at src/a.py"},
            ],
            "source": "active_learning",
        }
        cases.approve_case(conn, case_ids[0], final_example=al_example)

        # Step 5: Check metrics
        m = metrics.compute_metrics(conn)
        assert m["queue_metrics"]["total_cases"] >= 1
        assert m["queue_metrics"]["approved"] >= 1
