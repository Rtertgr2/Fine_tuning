#!/usr/bin/env python3
"""Deterministic JSON/HTML evaluation report and fail-closed gates (T4.9)."""
from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.config import EC

BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / "eval/reports"

GATE_LIMITS: dict[str, tuple[float, float | None]] = {
    "tool_syntax": (100.0, None),
    "anti_loop": (90.0, None),
    "security_catch": (85.0, None),
    "security_fp": (0.0, 10.0),
    "json_validity": (100.0, None),
    "plan_json": (100.0, None),
    "speed_8k": (30.0, None),
    "regression": (97.0, None),  # ratio = 100*candidate/base; 97% ≈ 3 pts regression when base acc is high (~95%+). Align with Plan/00-overview acceptance criterion.
    "end_to_end": (100.0, None),
}


def wilson_interval(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, expressed as percent."""
    if total <= 0:
        return (0.0, 0.0)
    p = min(max(passes / total, 0.0), 1.0)
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (round(max(0.0, center - spread) * 100, 1), round(min(1.0, center + spread) * 100, 1))


def _rate(result: dict[str, Any]) -> tuple[int, int, float]:
    try:
        passed = int(result.get("passed", 0))
        total = int(result.get("total", 0))
    except (TypeError, ValueError):
        return 0, 0, 0.0
    return passed, total, (100.0 * passed / total if total > 0 else 0.0)


def _case_json_counts(suites: list[dict[str, Any]]) -> tuple[int, int]:
    valid = total = 0
    for suite in suites:
        if suite.get("suite") not in {"security_catch", "security_fp", "plan_json"}:
            continue
        cases = suite.get("cases")
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict) and "valid_json" in case:
                    total += 1
                    valid += int(bool(case["valid_json"]))
        elif "valid_json" in suite:
            total += int(suite.get("total", 0))
            valid += int(suite.get("valid_json", 0))
    return valid, total


def _suite_summary(result: dict[str, Any]) -> dict[str, Any]:
    name = str(result.get("suite", "unknown"))
    passed, total, pct = _rate(result)
    lo, hi = wilson_interval(passed, total)
    summary = {
        "name": name,
        "passed": passed,
        "total": total,
        "pct": round(pct, 1),
        "ci95": [lo, hi],
        "warning": "sample size < 30; interpret cautiously" if total < 30 else None,
    }
    if name == "security_fp":
        fp = int(result.get("false_positives", max(0, total - passed)))
        fp_lo, fp_hi = wilson_interval(fp, total)
        summary["false_positives"] = fp
        summary["false_positive_rate"] = round(100 * fp / total, 1) if total else None
        summary["false_positive_ci95"] = [fp_lo, fp_hi]
        summary["pct"] = summary["false_positive_rate"] if total else 0.0
        summary["ci95"] = [fp_lo, fp_hi]
    if result.get("mode"):
        summary["mode"] = result["mode"]
    return summary


def _gate(name: str, actual: float | None, note: str | None = None) -> dict[str, Any]:
    minimum, maximum = GATE_LIMITS[name]
    if actual is None or not math.isfinite(actual):
        passed = False
        actual_value: float | None = None
    else:
        passed = actual >= minimum if maximum is None else minimum <= actual <= maximum
        actual_value = round(actual, 2)
    threshold = f">={minimum:g}" if maximum is None else f"{minimum:g}-{maximum:g}"
    units = " tok/s" if name == "speed_8k" else "%"
    item = {
        "name": name,
        "threshold": threshold + units,
        "actual": actual_value,
        "pass": passed,
    }
    if note:
        item["note"] = note
    if actual is None:
        item["note"] = note or "required measurement missing"
    return item


def _suite_hashes() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    suite_root = BASE / "eval/suites"
    if not suite_root.exists():
        return result
    for manifest_path in sorted(suite_root.glob("*/v*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        suite = str(manifest.get("suite", manifest_path.parent.parent.name))
        result[suite] = {
            "version": manifest.get("version"),
            "cases_sha256": manifest.get("cases_sha256"),
            "cases_count": manifest.get("cases_count"),
        }
    return result


def generate(
    results: list[dict[str, Any]],
    run_id: str,
    model_name: str,
    output_dir: Path | None = None,
    *,
    baseline: dict[str, Any] | None = None,
    previous: dict[str, Any] | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    artifact_hash: str | None = None,
) -> dict[str, Any]:
    """Generate a report. Missing mandatory measurements fail the overall gate.

    ``regression`` input, when supplied, must contain candidate/base pass@1
    counts. ``end_to_end`` input must contain candidate, base, and previous
    success rates; the candidate must be no worse than either reference.
    """
    output_dir = Path(output_dir) if output_dir is not None else REPORTS / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = [_suite_summary(result) for result in results]
    result_by_name = {str(result.get("suite")): result for result in results}

    metrics: dict[str, float | None] = {}
    for name in ("tool_syntax", "anti_loop", "security_catch", "plan_json"):
        source = result_by_name.get(name)
        metrics[name] = _rate(source)[2] if source else None

    fp_result = result_by_name.get("security_fp")
    if fp_result:
        fp_count = int(fp_result.get("false_positives", max(0, int(fp_result.get("total", 0)) - int(fp_result.get("passed", 0)))))
        fp_total = int(fp_result.get("total", 0))
        metrics["security_fp"] = 100 * fp_count / fp_total if fp_total else None
    else:
        metrics["security_fp"] = None

    valid_json, json_total = _case_json_counts(results)
    metrics["json_validity"] = 100 * valid_json / json_total if json_total else None

    speed = result_by_name.get("speed")
    if speed:
        try:
            metrics["speed_8k"] = float(speed.get("speed_8k"))
        except (TypeError, ValueError):
            metrics["speed_8k"] = None
    else:
        metrics["speed_8k"] = None

    regression = result_by_name.get("regression")
    if regression:
        candidate = regression.get("candidate_passed", regression.get("passed"))
        candidate_total = regression.get("candidate_total", regression.get("total"))
        base_passed = regression.get("base_passed")
        base_total = regression.get("base_total", candidate_total)
        try:
            candidate_rate = float(candidate) / float(candidate_total)
            base_rate = float(base_passed) / float(base_total)
            metrics["regression"] = 100 * candidate_rate / base_rate if base_rate > 0 else None
        except (TypeError, ValueError, ZeroDivisionError):
            metrics["regression"] = None
    else:
        metrics["regression"] = None

    e2e = result_by_name.get("end_to_end")
    if e2e:
        try:
            candidate_rate = float(e2e["candidate_success_rate"])
            base_rate = float(e2e["base_success_rate"])
            previous_rate = float(e2e["previous_success_rate"])
            metrics["end_to_end"] = 100.0 if candidate_rate >= base_rate and candidate_rate >= previous_rate else 0.0
        except (KeyError, TypeError, ValueError):
            metrics["end_to_end"] = None
    else:
        metrics["end_to_end"] = None

    gates = [_gate(name, metrics.get(name)) for name in GATE_LIMITS]
    report = {
        "run_id": run_id,
        "model": model_name,
        "model_id": model_id or model_name,
        "revision": revision,
        "artifact_hash": artifact_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": EC.temperature,
            "seed": EC.seed,
            "base_model": EC.base_model,
            "adapter_id": EC.adapter_id,
        },
        "suite_manifests": _suite_hashes(),
        "suites": suites,
        "metrics": metrics,
        "gates": gates,
        "overall_pass": all(gate["pass"] for gate in gates),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.html").write_text(_render_html(report), encoding="utf-8")
    return report


def _render_html(report: dict[str, Any]) -> str:
    status = "PASS" if report["overall_pass"] else "FAIL / INCOMPLETE"
    color = "#15803d" if report["overall_pass"] else "#b91c1c"
    gates_html = "".join(
        "<li><b>{}</b>: {} (threshold {}) — <span style='color:{}'>{}</span>{}</li>".format(
            html.escape(gate["name"]),
            html.escape("missing" if gate["actual"] is None else str(gate["actual"])),
            html.escape(gate["threshold"]),
            "#15803d" if gate["pass"] else "#b91c1c",
            "PASS" if gate["pass"] else "FAIL",
            " — " + html.escape(gate.get("note", "")) if gate.get("note") else "",
        )
        for gate in report["gates"]
    )
    rows = "".join(
        "<tr><td>{}</td><td>{}/{}</td><td>{}%</td><td>[{}%, {}%]</td><td>{}</td></tr>".format(
            html.escape(suite["name"]), suite["passed"], suite["total"], suite["pct"],
            suite["ci95"][0], suite["ci95"][1],
            "⚠ " + html.escape(suite["warning"]) if suite.get("warning") else "",
        )
        for suite in report["suites"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Eval report — {html.escape(report['run_id'])}</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.55rem;text-align:left}}th{{background:#f5f5f5}}.meta{{color:#555}}</style></head>
<body><h1>Eval report — {html.escape(report['run_id'])}</h1>
<div class="meta"><b>Model:</b> {html.escape(report['model'])}<br><b>Temperature:</b> {EC.temperature} | <b>Seed:</b> {EC.seed}<br><b>Created:</b> {html.escape(report['created_at'])}</div>
<h2>Overall: <span style="color:{color}">{status}</span></h2>
<h2>Required gates (missing measurements fail closed)</h2><ul>{gates_html}</ul>
<h2>Suite results</h2><table><tr><th>Suite</th><th>Correct</th><th>Rate</th><th>95% CI</th><th>Note</th></tr>{rows}</table>
</body></html>"""


if __name__ == "__main__":
    print("Report module; run the full evaluator with: python -m eval.runners.run_baseline")
