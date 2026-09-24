"""Tests for the eval harness itself (T4.10 self-test).

Uses fake model (mock responses) to verify gate logic works correctly.
"""
import json, sys
from pathlib import Path
from unittest.mock import patch, MagicMock

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from eval.client import LlamaClient
from eval.config import EC


def test_json_parser():
    """Test JSON extraction from various formats."""
    from eval.runners.run_all import parse_json_from_text
    
    # Plain JSON
    assert parse_json_from_text('{"status": "PASS"}') == {"status": "PASS"}
    
    # With code fences
    result = parse_json_from_text('```json\n{"status": "REJECT"}\n```')
    assert result == {"status": "REJECT"}
    
    # Invalid JSON
    result = parse_json_from_text("not json")
    assert "_error" in result
    
    # Empty
    result = parse_json_from_text("")
    assert "_error" in result
    
    print("JSON parser: OK")


def test_overlap_check():
    """Verify suites don't overlap with training groups.
    
    All gold set examples use "gold/" prefix, eval uses "eval/" prefix.
    This ensures no group leakage between train and eval.
    """
    suite_path = BASE / "eval/suites/tool_syntax/v001/cases.jsonl"
    cases = [json.loads(l) for l in suite_path.read_text().splitlines() if l.strip()]
    
    for case in cases:
        g = case.get("group", "")
        # Eval groups must start with "eval/", not "gold/"
        assert g.startswith("eval/"), f"Eval group should start with 'eval/': {g}"
        assert not g.startswith("gold/"), f"Eval group should not start with 'gold/': {g}"
    
    print(f"Overlap check: OK ({len(cases)} cases, all use 'eval/' prefix)")


def test_gates():
    """Test gate threshold comparison."""
    # Pass case
    pct = 95.0
    assert pct >= EC.gates[1].min_pct  # anti_loop >= 90
    
    # Fail case
    pct = 85.0
    assert not (pct >= EC.gates[0].min_pct)  # tool_syntax >= 100
    
    # Range case (FP)
    fp = 5.0
    assert fp <= EC.gates[3].max_pct  # security_fp <= 10
    
    print("Gate logic: OK")


def test_wilson_interval():
    """Test Wilson confidence interval calculation."""
    from eval.report import wilson_interval

    lo, hi = wilson_interval(100, 100)
    assert lo > 90 and hi == 100
    lo, hi = wilson_interval(50, 100)
    assert lo < 50 and hi > 50
    lo, hi = wilson_interval(3, 10)
    assert hi - lo > 30


def test_strict_judge_rejects_fences_and_empty_reject_issues():
    from eval.judge import strict_json_object, security_review_pass

    assert strict_json_object('{"status":"PASS"}')[0] == {"status": "PASS"}
    assert strict_json_object('```json\\n{"status":"PASS"}\\n```')[1] is not None
    case = {"expected_status": "REJECT", "expected_type": "sql_injection", "expected_line": 4}
    passed, details = security_review_pass('{"status":"REJECT","issues":[]}', case)
    assert not passed
    assert "non-empty" in details["reason"]


def test_tool_judge_checks_nested_json_and_exact_arguments():
    from eval.judge import tool_syntax_pass

    content = "print({'nested': 1})\n"
    response = {"content": "", "tool_calls": [{"function": {
        "name": "write_file",
        "arguments": json.dumps({"path": "src/a.py", "content": content}),
    }}]}
    case = {"expected_tool": "write_file", "expected_args": {
        "path": "src/a.py", "content": content
    }}
    passed, details = tool_syntax_pass(response, case)
    assert passed, details
    case["expected_args"]["path"] = "wrong.py"
    assert not tool_syntax_pass(response, case)[0]


def test_report_and_registry_share_gate_thresholds():
    from backend.app.services.models import GATE_THRESHOLDS
    from eval.report import GATE_LIMITS

    assert GATE_THRESHOLDS == GATE_LIMITS


def test_report_fails_closed_when_production_suites_are_missing(tmp_path):
    from eval.report import generate

    report = generate(
        [{"suite": "tool_syntax", "passed": 100, "total": 100}],
        "partial", "test-model", output_dir=tmp_path,
    )
    assert not report["overall_pass"]
    missing = {gate["name"] for gate in report["gates"] if gate["actual"] is None}
    assert {"regression", "end_to_end", "speed_8k", "json_validity"} <= missing


def test_report_computes_false_positive_rate_and_relative_regression(tmp_path):
    from eval.report import generate

    results = [
        {"suite": "tool_syntax", "passed": 200, "total": 200},
        {"suite": "anti_loop", "passed": 80, "total": 80},
        {"suite": "security_catch", "passed": 90, "total": 100,
         "cases": [{"valid_json": True} for _ in range(100)]},
        {"suite": "security_fp", "passed": 95, "total": 100, "false_positives": 5,
         "cases": [{"valid_json": True} for _ in range(100)]},
        {"suite": "plan_json", "passed": 50, "total": 50,
         "cases": [{"valid_json": True} for _ in range(50)]},
        {"suite": "speed", "speed_8k": 35},
        {"suite": "regression", "candidate_passed": 59, "candidate_total": 100,
         "base_passed": 60, "base_total": 100},
        {"suite": "end_to_end", "candidate_success_rate": 0.9,
         "base_success_rate": 0.85, "previous_success_rate": 0.9},
    ]
    report = generate(results, "complete", "test-model", output_dir=tmp_path)
    assert report["overall_pass"]
    assert report["metrics"]["security_fp"] == 5.0
    assert report["metrics"]["regression"] >= 97



if __name__ == "__main__":
    test_json_parser()
    test_overlap_check()
    test_gates()
    test_wilson_interval()
    print("\nAll eval harness self-tests passed.")
