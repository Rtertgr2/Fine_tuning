#!/usr/bin/env python3
"""Runner for tool_syntax suite (T4.5).

Calls llama-server with each test case, parses response for tool calls.
Checks: T1 valid JSON, T2 tool name, T3 args present.
"""
import json, re, sys
from pathlib import Path
from eval.client import LlamaClient
from eval.config import EC

BASE = Path(__file__).resolve().parent.parent
SUITES = BASE / "eval/suites"

def parse_tool_calls(content: str) -> list:
    """Extract tool_call tags from assistant content."""
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    calls = []
    for m in re.finditer(pattern, content, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            calls.append(obj)
        except json.JSONDecodeError:
            calls.append({"_error": "invalid_json", "raw": m.group(1)[:100]})
    return calls

def has_tools_in_response(result: dict) -> bool:
    """Check both content tool_calls tags and API tool_calls."""
    content = result.get("content", "")
    api_calls = result.get("tool_calls", [])
    content_calls = parse_tool_calls(content)
    return len(api_calls) > 0 or len(content_calls) > 0

def evaluate():
    client = LlamaClient()
    if not client.health():
        print("FAIL: llama-server not reachable at http://127.0.0.1:8080")
        return
    
    suite_path = SUITES / "tool_syntax/v001/cases.jsonl"
    cases = [json.loads(l) for l in suite_path.read_text().splitlines() if l.strip()]
    
    passed = 0
    results = []
    
    for i, case in enumerate(cases):
        prompt = case["prompt"]
        expected_tool = case["expected_tool"]
        expected_args = case["expected_args"]
        
        messages = [
            {"role": "system", "content": "You are a coding agent with tools: git_checkout, read_file, write_file."},
            {"role": "user", "content": prompt},
        ]
        
        result = client.chat(messages, temperature=EC.temperature)
        
        if "error" in result:
            results.append({"id": case["id"], "pass": False, "reason": result["error"]})
            continue
        
        content = result.get("content", "")
        api_calls = result.get("tool_calls", [])
        content_calls = parse_tool_calls(content)
        
        # Check tool call presence
        all_calls = content_calls if content_calls else []
        if not all_calls and api_calls:
            # Use API tool_calls
            for c in api_calls:
                fn = c.get("function", {})
                all_calls.append({"name": fn.get("name"), "arguments": fn.get("arguments", {})})
        
        if not all_calls:
            results.append({"id": case["id"], "pass": False, "reason": "no_tool_call", "content": content[:200]})
            continue
        
        # Check first call matches expected
        first = all_calls[0]
        name_match = first.get("name") == expected_tool
        
        args = first.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        
        args_match = all(k in args for k in expected_args.keys())
        
        ok = name_match and args_match
        if ok:
            passed += 1
        results.append({
            "id": case["id"],
            "pass": ok,
            "expected_tool": expected_tool,
            "got_tool": first.get("name"),
            "args_match": args_match,
        })
    
    pct = passed / len(cases) * 100
    gate_pass = pct >= EC.gates[0].min_pct  # tool_syntax gate = 100%
    
    print(f"Tool syntax: {passed}/{len(cases)} = {pct:.1f}% (gate: {EC.gates[0].min_pct}%)")
    print(f"Gate: {'PASS' if gate_pass else 'FAIL'}")
    return passed

if __name__ == "__main__":
    evaluate()
