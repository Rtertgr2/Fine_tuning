#!/usr/bin/env python3
"""Run all eval suites and generate report (T4.10).

Usage:
    python -m eval.runners.run_all_baseline --model "Hermes-2-Pro-Llama-3-8B" --run-id baseline_hermes2pro
"""
import json, sys, argparse
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from eval.config import EC
from eval.client import LlamaClient
from eval import runners
from eval.report import generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=EC.base_model)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    
    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    client = LlamaClient(args.server)
    if not client.health():
        print(f"ERROR: llama-server not reachable at {args.server}")
        sys.exit(1)
    
    print(f"=== Eval Harness ===")
    print(f"Model: {args.model}")
    print(f"Run ID: {run_id}")
    print(f"Temperature: {EC.temperature}")
    print()
    
    all_results = []
    
    print("--- Tool Syntax ---")
    r = runners.run_tool_syntax()
    all_results.append({"suite": "tool_syntax", "passed": r, "total": 100})
    
    print("\n--- Anti-Loop ---")
    r = runners.anti_loop()
    all_results.append({"suite": "anti_loop", "passed": r, "total": 40})
    
    print("\n--- Security Vuln ---")
    r = runners.security_vuln()
    all_results.append({"suite": "security_catch", "passed": r, "total": 100})
    
    print("\n--- Security Clean ---")
    r = runners.security_clean()
    all_results.append({"suite": "security_fp", "passed": r, "total": 100})
    
    print("\n--- Plan JSON ---")
    r = runners.plan_json()
    all_results.append({"suite": "plan_json", "passed": r, "total": 50})
    
    print("\n--- Speed ---")
    runners.run_speed()
    
    # Generate report
    report = generate(all_results, run_id, args.model)
    print(f"\n=== Report: eval/reports/{run_id}/ ===")
    print(f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    
    return 0 if report['overall_pass'] else 1

if __name__ == "__main__":
    sys.exit(main())
