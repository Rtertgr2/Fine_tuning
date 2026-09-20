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
    
    # Perfect score
    lo, hi = wilson_interval(100, 100)
    assert lo > 90 and hi == 100
    
    # 50/50
    lo, hi = wilson_interval(50, 100)
    assert lo < 50 and hi > 50
    
    # Small sample
    lo, hi = wilson_interval(3, 10)
    assert hi - lo > 30  # Wide interval for small sample
    
    print("Wilson interval: OK")


if __name__ == "__main__":
    test_json_parser()
    test_overlap_check()
    test_gates()
    test_wilson_interval()
    print("\nAll eval harness self-tests passed.")
