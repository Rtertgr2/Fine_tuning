"""Rule-by-rule tests: every rule gets at least one passing and one failing case."""

from __future__ import annotations

from backend.adapters.registry import get_adapter
from backend.validators.base import Example, ValidationContext
from backend.validators.registry import validate_dataset, validate_example

from tests.conftest import msg, read_call, write_call


def make_ctx(peers=(), tokenizer_available=False, adapter=None, **kw):
    adapter = adapter or get_adapter()
    defaults = dict(
        adapter=adapter,
        max_seq_len=8192,
        peers=list(peers),
        tokenizer_available=tokenizer_available,
        near_dup_threshold=0.85,
        l4_giveup_ratio=0.15,
        s6_pass_window=(0.35, 0.65),
        s7_min_per_type=1,
    )
    defaults.update(kw)
    return ValidationContext(**defaults)


def codes(report, level=None):
    return {x.code for x in report.violations if level is None or x.level == level}


class StubAdapter:
    """For C4: no tokenizer, fixed count."""

    def __init__(self, count):
        self._count = count

    def render_conversation(self, *a, **k):
        return "rendered"

    def count_tokens(self, text):
        return self._count


# ---------------------------------------------------------------------------
# C1–C6
# ---------------------------------------------------------------------------


def test_c1_pass(tool_example):
    assert "C1" not in codes(validate_example(tool_example, make_ctx()))


def test_c1_fail_two_users(tool_example):
    bad = Example(
        id="x",
        category="tool",
        messages=[msg("user", "a"), msg("user", "b"), msg("assistant", "c")],
    )
    assert "C1" in codes(validate_example(bad, make_ctx()))


def test_c1_fail_ends_with_tool(tool_example):
    bad = Example(
        id="x",
        category="tool",
        messages=[
            msg("user", "a"),
            msg("assistant", read_call()),
            msg("tool", "result"),
        ],
    )
    assert "C1" in codes(validate_example(bad, make_ctx()))


def test_c2_fail_empty_content(tool_example):
    bad = Example(
        id="x",
        category="plan",
        messages=[msg("user", "a"), msg("assistant", "   ")],
    )
    assert "C2" in codes(validate_example(bad, make_ctx()))


def test_c3_fail_no_assistant():
    bad = Example(id="x", category="plan", messages=[msg("user", "a")])
    assert "C3" in codes(validate_example(bad, make_ctx()))


def test_c4_pass_under_limit(plan_example):
    ctx = make_ctx(tokenizer_available=True, adapter=StubAdapter(100))
    assert "C4" not in codes(validate_example(plan_example, ctx))


def test_c4_fail_over_limit(plan_example):
    ctx = make_ctx(tokenizer_available=True, adapter=StubAdapter(999999))
    assert "C4" in codes(validate_example(plan_example, ctx))


def test_c5_fail_aws_key():
    bad = Example(
        id="x",
        category="plan",
        messages=[
            msg("user", "a"),
            msg("assistant", '{"status":"APPROVED","aws":"AKIAIOSFODNN7EXAMPLE"}'),
        ],
    )
    assert "C5" in codes(validate_example(bad, make_ctx()))


def test_c6_exact_duplicate(plan_example):
    peer = Example(
        id="peer1",
        category="plan",
        messages=list(plan_example.messages),
    )
    report = validate_example(plan_example, make_ctx(peers=[peer]))
    assert "C6" in codes(report, "err")


def test_c6_near_duplicate_warns():
    text = "ช่วยอ่านไฟล์ a.py แล้วเขียนฟังก์ชัน hello world ให้หน่อยครับ " * 3
    a = Example(id="a", category="plan", messages=[msg("user", text), msg("assistant", '{"status":"APPROVED","target_version":"v1.0.0","critique":"x","final_plan":"y"}')])
    b = Example(id="b", category="plan", messages=[msg("user", text), msg("assistant", '{"status":"APPROVED","target_version":"v1.0.0","critique":"z","final_plan":"y"}')])
    report = validate_example(a, make_ctx(peers=[b]))
    assert "C6" in codes(report, "warn")


def test_c6_clean_pass(plan_example):
    report = validate_example(plan_example, make_ctx(peers=[]))
    assert "C6" not in codes(report)


# ---------------------------------------------------------------------------
# T1–T6
# ---------------------------------------------------------------------------


def test_t1_pass(tool_example):
    assert "T1" not in codes(validate_example(tool_example, make_ctx()))


def test_t1_fail_unclosed(tool_example):
    bad = Example(
        id="x",
        category="tool",
        messages=[
            msg("user", "a"),
            msg("assistant", '<tool_call>\n{"name": "read_file", "arguments": {}}'),
        ],
    )
    assert "T1" in codes(validate_example(bad, make_ctx()))


def test_t2_fail_unknown_tool(tool_example):
    bad = Example(
        id="x",
        category="tool",
        messages=[msg("user", "a"), msg("assistant", read_call("x.py").replace("read_file", "delete_repo"))],
    )
    assert "T2" in codes(validate_example(bad, make_ctx()))


def test_t3_fail_missing_argument(tool_example):
    bad = Example(
        id="x",
        category="tool",
        messages=[
            msg("user", "a"),
            msg("assistant", '<tool_call>\n{"name": "read_file", "arguments": {}}\n</tool_call>'),
        ],
    )
    assert "T3" in codes(validate_example(bad, make_ctx()))


def test_t4_fail_missing_result(tool_example):
    bad = Example(
        id="x",
        category="tool",
        messages=[msg("user", "a"), msg("assistant", read_call()), msg("assistant", "เสร็จแล้ว")],
    )
    report = validate_example(bad, make_ctx())
    assert "T4" in codes(report)


def test_t5_pass_three_calls():
    calls = "\n".join([read_call("a.py"), read_call("b.py"), read_call("c.py")])
    ex = Example(
        id="x",
        category="tool",
        messages=[
            msg("user", "a"),
            msg("assistant", calls),
            msg("tool", "r1"),
            msg("tool", "r2"),
            msg("tool", "r3"),
            msg("assistant", "done"),
        ],
    )
    assert "T5" not in codes(validate_example(ex, make_ctx()))


def test_t5_warn_two_calls():
    ex = Example(
        id="x",
        category="tool",
        messages=[
            msg("user", "a"),
            msg("assistant", read_call("a.py")),
            msg("tool", "r"),
            msg("assistant", write_call("b.py", "print(1)\n")),
            msg("tool", "wrote b.py"),
            msg("assistant", "done"),
        ],
    )
    report = validate_example(ex, make_ctx())
    assert "T5" in codes(report, "warn")


def test_t5_pass_three_calls_fixture(tool_example):
    assert "T5" not in codes(validate_example(tool_example, make_ctx()))


def test_t6_fail_diff_content():
    bad = Example(
        id="x",
        category="tool",
        messages=[
            msg("user", "a"),
            msg("assistant", read_call("a.py")),
            msg("tool", "r"),
            msg("assistant", write_call("a.py", "@@ -1,3 +1,4 @@\n+import os\n rest unchanged")),
        ],
    )
    assert "T6" in codes(validate_example(bad, make_ctx()))


def test_t6_pass_full_file():
    ok = Example(
        id="x",
        category="tool",
        messages=[
            msg("user", "a"),
            msg("assistant", read_call("a.py")),
            msg("tool", "r"),
            msg("assistant", write_call("a.py", "import os\n\nprint('ok')\n")),
        ],
    )
    assert "T6" not in codes(validate_example(ok, make_ctx()))


# ---------------------------------------------------------------------------
# L1–L4
# ---------------------------------------------------------------------------


def test_l1_pass(loop_example):
    assert "L1" not in codes(validate_example(loop_example, make_ctx()))


def test_l1_warn_no_failure():
    ex = Example(
        id="x",
        category="loop",
        messages=[
            msg("user", "a"),
            msg("assistant", read_call()),
            msg("tool", "file contents"),
            msg("assistant", "อ่านเสร็จแล้วครับ"),
        ],
    )
    report = validate_example(ex, make_ctx())
    assert "L1" in codes(report, "warn")


def test_l2_pass_recovery(loop_example):
    assert "L2" not in codes(validate_example(loop_example, make_ctx()))


def test_l2_fail_identical_retry():
    ex = Example(
        id="x",
        category="loop",
        messages=[
            msg("user", "a"),
            msg("assistant", read_call("b.py")),
            msg("tool", "Error: no such file"),
            msg("assistant", read_call("b.py")),
            msg("tool", "Error: no such file"),
            msg("assistant", "หยุดและรายงานว่าทำไม่ได้ครับ"),
        ],
    )
    report = validate_example(ex, make_ctx())
    assert "L2" in codes(report, "err")
    assert "L3" in codes(report, "err")


def test_l4_pass_dataset():
    loops = []
    for i in range(4):
        loops.append(
            Example(
                id=f"l{i}",
                category="loop",
                messages=[
                    msg("user", "a"),
                    msg("assistant", read_call(f"{i}.py")),
                    msg("tool", "Error: not found"),
                    msg("assistant", "ไม่สามารถดำเนินการต่อได้ครับ ขอรายงานว่าล้มเหลว"),
                ],
            )
        )
    report = validate_dataset(loops, make_ctx())
    assert "L4" not in codes(report)


def test_l4_warn_dataset():
    loops = []
    for i in range(4):
        loops.append(
            Example(
                id=f"l{i}",
                category="loop",
                messages=[
                    msg("user", "a"),
                    msg("assistant", read_call(f"{i}.py")),
                    msg("tool", "Error: not found"),
                    msg("assistant", "อ่านไฟล์ใหม่แล้วเจอครับ"),
                ],
            )
        )
    report = validate_dataset(loops, make_ctx())
    assert "L4" in codes(report, "warn")


# ---------------------------------------------------------------------------
# P1–P4
# ---------------------------------------------------------------------------


def test_p1_pass(plan_example):
    assert "P1" not in codes(validate_example(plan_example, make_ctx()))


def test_p1_fail_code_fence(plan_example):
    bad = Example(
        id="x",
        category="plan",
        messages=[
            msg("user", "a"),
            msg("assistant", '```json\n{"status":"APPROVED"}\n```'),
        ],
    )
    assert "P1" in codes(validate_example(bad, make_ctx()))


def test_p2_fail_missing_fields(plan_example):
    bad = Example(
        id="x",
        category="plan",
        messages=[msg("user", "a"), msg("assistant", '{"status": "APPROVED"}')],
    )
    assert "P2" in codes(validate_example(bad, make_ctx()))


def test_p3_fail_bad_status(plan_example):
    bad = Example(
        id="x",
        category="plan",
        messages=[
            msg("user", "a"),
            msg(
                "assistant",
                '{"status": "MAYBE", "target_version": "v1.0.0", "critique": "c", "final_plan": "f"}',
            ),
        ],
    )
    assert "P3" in codes(validate_example(bad, make_ctx()))


def test_p4_warn_bad_version(plan_example):
    bad = Example(
        id="x",
        category="plan",
        messages=[
            msg("user", "a"),
            msg(
                "assistant",
                '{"status": "APPROVED", "target_version": "1.0", "critique": "c", "final_plan": "f"}',
            ),
        ],
    )
    assert "P4" in codes(validate_example(bad, make_ctx()), "warn")


# ---------------------------------------------------------------------------
# S1–S7
# ---------------------------------------------------------------------------


def test_s1_pass(sec_example):
    assert "S1" not in codes(validate_example(sec_example, make_ctx()))


def test_s1_fail_not_json(sec_example):
    bad = Example(
        id="x",
        category="sec",
        messages=[msg("user", "a"), msg("assistant", "looks fine to me!")],
    )
    assert "S1" in codes(validate_example(bad, make_ctx()))


def test_s2_fail_bad_status(sec_example):
    bad = Example(
        id="x",
        category="sec",
        messages=[msg("user", "a"), msg("assistant", '{"status": "OK"}')],
    )
    assert "S2" in codes(validate_example(bad, make_ctx()))


def test_s3_pass(sec_example):
    assert "S3" not in codes(validate_example(sec_example, make_ctx()))


def test_s3_fail_missing_fix():
    bad = Example(
        id="x",
        category="sec",
        messages=[
            msg("user", "a"),
            msg(
                "assistant",
                '{"status": "REJECT", "issues": [{"file": "a.py", "line": 1, "type": "x", "severity": "high"}]}',
            ),
        ],
    )
    assert "S3" in codes(validate_example(bad, make_ctx()))


def test_s4_pass(sec_example):
    assert "S4" not in codes(validate_example(sec_example, make_ctx()))


def test_s4_fail_line_not_in_diff(sec_example):
    bad = Example(
        id="x",
        category="sec",
        messages=[
            sec_example.messages[0],
            sec_example.messages[1],
            msg(
                "assistant",
                '{"status": "REJECT", "issues": [{"file": "app.py", "line": 999, "type": "sql_injection", "severity": "high", "fix": "f"}]}',
            ),
        ],
    )
    assert "S4" in codes(validate_example(bad, make_ctx()))


def test_s5_warn_pass_with_issues(sec_example):
    bad = Example(
        id="x",
        category="sec",
        messages=[
            sec_example.messages[0],
            sec_example.messages[1],
            msg(
                "assistant",
                '{"status": "PASS", "issues": [{"file": "app.py", "line": 1, "type": "x", "severity": "low", "fix": "f"}]}',
            ),
        ],
    )
    assert "S5" in codes(validate_example(bad, make_ctx()), "warn")


def _sec(status, issues=None, i=0):
    obj = {"status": status}
    if issues:
        obj["issues"] = issues
    import json as _json

    return Example(
        id=f"s{i}",
        category="sec",
        messages=[
            msg("system", "s"),
            msg("user", "diff"),
            msg("assistant", _json.dumps(obj)),
        ],
    )


def test_s6_pass_balanced():
    examples = [
        _sec("PASS", i=1),
        _sec("PASS", i=2),
        _sec("REJECT", [{"file": "a", "line": 1, "type": "x", "severity": "low", "fix": "f"}], i=3),
        _sec("REJECT", [{"file": "a", "line": 1, "type": "y", "severity": "low", "fix": "f"}], i=4),
    ]
    report = validate_dataset(examples, make_ctx())
    assert "S6" not in codes(report)


def test_s6_warn_all_pass():
    examples = [_sec("PASS", i=i) for i in range(4)]
    report = validate_dataset(examples, make_ctx())
    assert "S6" in codes(report, "warn")


def test_s7_warn_missing_type():
    examples = [
        _sec("REJECT", [{"file": "a", "line": 1, "type": "sql_injection", "severity": "low", "fix": "f"}], i=1),
        _sec("PASS", i=2),
    ]
    ctx = make_ctx(s7_min_per_type=1)
    report = validate_dataset(examples, ctx)
    # resource_leak / insecure_deserialization / owasp have zero examples
    assert "S7" in codes(report, "warn")


def test_s7_pass_all_types():
    examples = [
        _sec("REJECT", [{"file": "a", "line": 1, "type": "sql_injection", "severity": "low", "fix": "f"}], i=1),
        _sec("REJECT", [{"file": "a", "line": 1, "type": "resource_leak", "severity": "low", "fix": "f"}], i=2),
        _sec("REJECT", [{"file": "a", "line": 1, "type": "insecure_deserialization", "severity": "low", "fix": "f"}], i=3),
        _sec("REJECT", [{"file": "a", "line": 1, "type": "owasp_a03", "severity": "low", "fix": "f"}], i=4),
    ]
    ctx = make_ctx(s7_min_per_type=1)
    report = validate_dataset(examples, ctx)
    assert "S7" not in codes(report)
