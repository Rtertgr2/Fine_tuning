#!/usr/bin/env python3
"""Generate eval report (T4.9).

Produces JSON + HTML report with:
- Per-suite pass rates with Wilson confidence intervals
- Gate pass/fail summary
- Version comparison (if previous reports exist)
"""
import json, math
from pathlib import Path
from datetime import datetime, timezone
from eval.config import EC

BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / "eval/reports"

def wilson_interval(passes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval for binomial proportion."""
    if total == 0:
        return (0, 0)
    p = passes / total
    denom = 1 + z*z / total
    center = (p + z*z / (2*total)) / denom
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total) / denom
    lo = max(0, center - spread) * 100
    hi = min(1, center + spread) * 100
    return (round(lo, 1), round(hi, 1))

def generate(results: list, run_id: str, model_name: str, output_dir: Path | None = None) -> dict:
    """Generate report from runner results."""
    if output_dir is None:
        output_dir = REPORTS / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "run_id": run_id,
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "temperature": EC.temperature,
            "seed": EC.seed,
            "base_model": EC.base_model,
        },
        "suites": [],
        "gates": [],
        "overall_pass": True,
    }
    
    for suite_result in results:
        suite_name = suite_result["suite"]
        passed = suite_result["passed"]
        total = suite_result["total"]
        pct = passed / total * 100 if total > 0 else 0
        lo, hi = wilson_interval(passed, total)
        
        # Find matching gate
        gate = None
        for g in EC.gates:
            if g.name == suite_name.replace("_vuln", "_catch").replace("_clean", "_fp") or \
               g.name in suite_name:
                gate = g
                break
        
        gate_pass = True
        if gate:
            if gate.max_pct is not None:
                gate_pass = pct >= gate.min_pct and pct <= gate.max_pct
            else:
                gate_pass = pct >= gate.min_pct
        
        suite_entry = {
            "name": suite_name,
            "passed": passed,
            "total": total,
            "pct": round(pct, 1),
            "ci95": [lo, hi],
            "gate": gate.name if gate else None,
            "gate_pass": gate_pass,
        }
        report["suites"].append(suite_entry)
        
        if not gate_pass:
            report["overall_pass"] = False
        
        if gate:
            report["gates"].append({
                "name": gate.name,
                "threshold": f">={gate.min_pct}%" if gate.max_pct is None else f"{gate.min_pct}-{gate.max_pct}%",
                "actual": round(pct, 1),
                "pass": gate_pass,
            })
    
    # Write JSON
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Write HTML
    html = _render_html(report)
    with open(output_dir / "report.html", "w") as f:
        f.write(html)
    
    return report

def _render_html(report: dict) -> str:
    """Render HTML report."""
    status = "PASS" if report["overall_pass"] else "FAIL"
    color = "#15803d" if report["overall_pass"] else "#b91c1c"
    
    rows = ""
    for s in report["suites"]:
        s_status = "PASS" if s["gate_pass"] else "FAIL"
        s_color = "#15803d" if s["gate_pass"] else "#b91c1c"
        rows += f"""
        <tr>
            <td>{s['name']}</td>
            <td>{s['passed']}/{s['total']}</td>
            <td>{s['pct']}%</td>
            <td>[{s['ci95'][0]}%, {s['ci95'][1]}%]</td>
            <td>{s['gate'] or '-'}</td>
            <td style="color:{s_color};font-weight:bold">{s_status}</td>
        </tr>"""
    
    gates_html = ""
    for g in report["gates"]:
        g_color = "#15803d" if g["pass"] else "#b91c1c"
        gates_html += f"<li><b>{g['name']}</b>: {g['actual']}% (threshold: {g['threshold']}) — <span style='color:{g_color}'>{'PASS' if g['pass'] else 'FAIL'}</span></li>"
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>Eval Report — {report['run_id']}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;}}
th,td{{border:1px solid #ddd;padding:0.5rem;text-align:left;}}
th{{background:#f5f5f5;}}
h1{{margin-bottom:0;}}
.meta{{color:#666;margin-bottom:2rem;}}
</style>
</head>
<body>
<h1>Eval Report — {report['run_id']}</h1>
<div class="meta">
    <b>Model:</b> {report['model']}<br>
    <b>Temperature:</b> {report['config']['temperature']} | <b>Seed:</b> {report['config']['seed']}<br>
    <b>Created:</b> {report['created_at']}
</div>

<h2>Overall: <span style="color:{color}">{status}</span></h2>

<h3>Gate Summary</h3>
<ul>{gates_html}</ul>

<h3>Suite Results</h3>
<table>
<tr><th>Suite</th><th>Passed</th><th>Rate</th><th>CI 95%</th><th>Gate</th><th>Status</th></tr>
{rows}
</table>
</body>
</html>"""

if __name__ == "__main__":
    # Demo with dummy data
    demo_results = [
        {"suite": "tool_syntax", "passed": 95, "total": 100},
        {"suite": "anti_loop", "passed": 37, "total": 40},
        {"suite": "security_catch", "passed": 88, "total": 100},
        {"suite": "security_fp", "passed": 95, "total": 100},
        {"suite": "plan_json", "passed": 49, "total": 50},
    ]
    report = generate(demo_results, "demo_run", EC.base_model)
    print(f"Report generated: eval/reports/demo_run/")
    print(f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
