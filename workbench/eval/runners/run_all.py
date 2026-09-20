#!/usr/bin/env python3
"""Runners for anti_loop and security suites (T4.6)."""
import json, re
from pathlib import Path
from eval.client import LlamaClient
from eval.config import EC

BASE = Path(__file__).resolve().parent.parent
SUITES = BASE / "eval/suites"

def parse_json_from_text(content: str) -> dict:
    """Try to parse JSON from text (handles code fences)."""
    if not content:
        return {"_error": "empty_content"}
    # Strip code fences
    cleaned = re.sub(r'^```json\s*', '', content.strip())
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"_error": f"invalid_json: {str(e)[:100]}", "raw": cleaned[:300]}

def anti_loop():
    """Test that agent changes behavior after errors (L2 rule)."""
    client = LlamaClient()
    suite_path = SUITES / "anti_loop/v001/cases.jsonl"
    cases = [json.loads(l) for l in suite_path.read_text().splitlines() if l.strip()]
    
    results = []
    passed = 0
    
    for case in cases:
        prompt = case["prompt"]
        inject_error = case.get("inject_error", "")
        
        messages = [
            {"role": "system", "content": "You are a coding agent with tools. If a tool call fails, try a different approach or report the failure clearly."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"some_file.py\"}}\n</tool_call>"},
            {"role": "tool", "content": inject_error},
        ]
        
        result = client.chat(messages, temperature=EC.temperature)
        content = result.get("content", "")
        
        # Check: does response differ from original tool call?
        # Must either: call a different tool, or report failure
        repeated_same_call = "read_file" in content and "some_file.py" in content
        reports_failure = any(w in content.lower() for w in ["error", "fail", "cannot", "unable", "report", "not found", "permission"])
        
        ok = reports_failure or not repeated_same_call
        if ok:
            passed += 1
        
        results.append({
            "id": case["id"],
            "pass": ok,
            "repeated": repeated_same_call,
            "reports_failure": reports_failure,
            "snippet": content[:150],
        })
    
    pct = passed / len(cases) * 100
    gate = EC.gates[1]  # anti_loop
    gate_pass = pct >= gate.min_pct
    print(f"Anti-loop: {passed}/{len(cases)} = {pct:.1f}% (gate: {gate.min_pct}%)")
    print(f"Gate: {'PASS' if gate_pass else 'FAIL'}")
    return passed

def security_vuln():
    """Test vulnerability detection (3a catch rate)."""
    client = LlamaClient()
    suite_path = SUITES / "security_vuln/v001/cases.jsonl"
    cases = [json.loads(l) for l in suite_path.read_text().splitlines() if l.strip()]
    
    results = []
    passed = 0
    
    for case in cases:
        prompt = case["prompt"]
        expected_status = case.get("expected_status", "REJECT")
        expected_type = case.get("expected_type", "")
        expected_line = case.get("expected_line", 0)
        
        messages = [
            {"role": "system", "content": "You are a security review agent. Analyze code diffs for security issues. Return JSON with: status (PASS or REJECT), and if REJECT, issues[] with file, line, type, severity, fix."},
            {"role": "user", "content": prompt},
        ]
        
        result = client.chat(messages, temperature=EC.temperature)
        content = result.get("content", "")
        parsed = parse_json_from_text(content)
        
        if "_error" in parsed:
            results.append({"id": case["id"], "pass": False, "reason": parsed["_error"]})
            continue
        
        actual_status = parsed.get("status", "").upper()
        status_match = actual_status == expected_status
        
        # For REJECT, check type and line
        type_match = True
        line_match = True
        if expected_status == "REJECT" and actual_status == "REJECT":
            issues = parsed.get("issues", [])
            if issues:
                first_issue = issues[0]
                if expected_type:
                    type_match = expected_type in str(first_issue.get("type", ""))
                if expected_line:
                    line_match = first_issue.get("line", 0) == expected_line
        
        ok = status_match and type_match and line_match
        if ok:
            passed += 1
        
        results.append({
            "id": case["id"],
            "pass": ok,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "status_match": status_match,
        })
    
    pct = passed / len(cases) * 100
    gate = EC.gates[2]  # security_catch
    gate_pass = pct >= gate.min_pct
    print(f"Security vuln: {passed}/{len(cases)} = {pct:.1f}% (gate: {gate.min_pct}%)")
    print(f"Gate: {'PASS' if gate_pass else 'FAIL'}")
    return passed

def security_clean():
    """Test false positive rate (3b)."""
    client = LlamaClient()
    suite_path = SUITES / "security_clean/v001/cases.jsonl"
    cases = [json.loads(l) for l in suite_path.read_text().splitlines() if l.strip()]
    
    results = []
    passed = 0  # passed = PASS (correct)
    
    for case in cases:
        prompt = case["prompt"]
        expected_status = case.get("expected_status", "PASS")
        
        messages = [
            {"role": "system", "content": "You are a security review agent. Analyze code diffs for security issues. Return JSON with: status (PASS or REJECT), and if REJECT, issues[] with file, line, type, severity, fix."},
            {"role": "user", "content": prompt},
        ]
        
        result = client.chat(messages, temperature=EC.temperature)
        content = result.get("content", "")
        parsed = parse_json_from_text(content)
        
        if "_error" in parsed:
            results.append({"id": case["id"], "pass": False, "reason": parsed["_error"]})
            continue
        
        actual_status = parsed.get("status", "").upper()
        ok = actual_status == expected_status
        if ok:
            passed += 1
        
        results.append({
            "id": case["id"],
            "pass": ok,
            "expected_status": expected_status,
            "actual_status": actual_status,
        })
    
    pct = passed / len(cases) * 100
    # FP rate = 100 - pct (how many wrongly rejected)
    fp_pct = 100 - pct
    gate = EC.gates[3]  # security_fp
    gate_pass = fp_pct <= gate.max_pct
    print(f"Security clean: {passed}/{len(cases)} correct = {pct:.1f}% (FP: {fp_pct:.1f}%, max: {gate.max_pct}%)")
    print(f"Gate: {'PASS' if gate_pass else 'FAIL'}")
    return passed

def plan_json():
    """Test plan JSON validity (P1-P3 rules)."""
    client = LlamaClient()
    suite_path = SUITES / "plan_json/v001/cases.jsonl"
    cases = [json.loads(l) for l in suite_path.read_text().splitlines() if l.strip()]
    
    results = []
    passed = 0
    
    for case in cases:
        prompt = case["prompt"]
        required_fields = case.get("required_fields", [])
        expected_statuses = case.get("expected_status_values", [])
        
        messages = [
            {"role": "system", "content": "You are a planning agent. Given a task, return a JSON plan with: status (APPROVED or NEEDS_REVISION), target_version, critique, final_plan."},
            {"role": "user", "content": prompt},
        ]
        
        result = client.chat(messages, temperature=EC.temperature)
        content = result.get("content", "")
        parsed = parse_json_from_text(content)
        
        if "_error" in parsed:
            results.append({"id": case["id"], "pass": False, "reason": parsed["_error"]})
            continue
        
        # Check required fields
        has_fields = all(f in parsed for f in required_fields)
        status_valid = parsed.get("status", "").upper() in [s.upper() for s in expected_statuses]
        
        ok = has_fields and status_valid
        if ok:
            passed += 1
        
        results.append({
            "id": case["id"],
            "pass": ok,
            "has_fields": has_fields,
            "status_valid": status_valid,
        })
    
    pct = passed / len(cases) * 100
    gate = EC.gates[4]  # json_validity (note: this is for security, but applies here too)
    # Actually plan_json gate is 100% like json_validity
    gate_pass = pct >= 100.0
    print(f"Plan JSON: {passed}/{len(cases)} = {pct:.1f}% (gate: 100%)")
    print(f"Gate: {'PASS' if gate_pass else 'FAIL'}")
    return passed

if __name__ == "__main__":
    print("=== ANTI_LOOP ===")
    anti_loop()
    print("\n=== SECURITY VULN ===")
    security_vuln()
    print("\n=== SECURITY CLEAN ===")
    security_clean()
    print("\n=== PLAN JSON ===")
    plan_json()
