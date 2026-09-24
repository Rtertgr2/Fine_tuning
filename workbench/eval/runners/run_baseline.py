#!/usr/bin/env python3
"""Run frozen evaluation suites and write a fail-closed report (T4.10)."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from eval.client import LlamaClient
from eval.config import EC
from eval.report import generate
from eval.runners import anti_loop, plan_json, run_speed, run_tool_syntax, security_clean, security_vuln


def _combine_modes(raw: dict, api: dict, suite: str) -> dict:
    return {
        "suite": suite,
        "passed": int(raw.get("passed", 0)) + int(api.get("passed", 0)),
        "total": int(raw.get("total", 0)) + int(api.get("total", 0)),
        "valid_json": int(raw.get("valid_json", 0)) + int(api.get("valid_json", 0)),
        "mode": "raw+api",
        "cases": list(raw.get("cases", [])) + list(api.get("cases", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all available frozen eval suites")
    parser.add_argument("--model", default=EC.base_model)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--artifact-hash", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    client = LlamaClient(args.server, model=args.model, revision=args.revision)
    if not client.health():
        print(f"ERROR: llama-server not reachable at {args.server}")
        return 2

    print(f"=== Eval Harness ===\nModel: {args.model}\nRevision: {args.revision}\nRun ID: {run_id}\nServer: {args.server}\nTemperature: {EC.temperature}\n")
    results = []
    for title, fn, name in (
        ("Tool syntax (raw tags)", run_tool_syntax, "tool_syntax"),
        ("Anti-loop (raw tags)", anti_loop, "anti_loop"),
    ):
        print(f"--- {title} ---")
        raw = fn(client=client, mode="raw")
        print(f"  {raw['passed']}/{raw['total']}")
        if name == "tool_syntax":
            api = run_tool_syntax(client=client, mode="api")
            print(f"--- Tool syntax (API tools) ---\n  {api['passed']}/{api['total']}")
            results.append(_combine_modes(raw, api, name))
        else:
            api = anti_loop(client=client, mode="api")
            print(f"--- Anti-loop (API tools) ---\n  {api['passed']}/{api['total']}")
            results.append(_combine_modes(raw, api, name))

    for title, fn in (
        ("Security vulnerability catch", security_vuln),
        ("Security clean / false positives", security_clean),
        ("Plan JSON", plan_json),
    ):
        print(f"--- {title} ---")
        item = fn(client=client)
        results.append(item)
        print(f"  {item['passed']}/{item['total']}")
        if item["suite"] == "security_fp":
            print(f"  false positives: {item['false_positives']}/{item['total']}")

    print("--- Speed ---")
    speed = run_speed(client=client)
    results.append(speed)
    for context in speed.get("contexts", []):
        print(f"  {context['label']}: {context['tok_s']:.2f} tok/s; prompt={context.get('prompt_tokens', 'n/a')}")

    # These suites are mandatory for a production report but are not implemented
    # yet; report.generate intentionally fails closed rather than silently
    # marking a partially measured model as promotable.
    report = generate(results, run_id, args.model,
                    model_id=args.model, revision=args.revision,
                    artifact_hash=args.artifact_hash)
    print(f"\nReport: eval/reports/{run_id}/report.html")
    print(f"Overall: {'PASS' if report['overall_pass'] else 'FAIL / INCOMPLETE'}")
    missing = [gate["name"] for gate in report["gates"] if gate["actual"] is None]
    if missing:
        print("Missing required measurements (promotion remains blocked): " + ", ".join(missing))
    else:
        failed = [gate["name"] for gate in report["gates"] if not gate["pass"]]
        if failed:
            print("Failed gates: " + ", ".join(failed))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
