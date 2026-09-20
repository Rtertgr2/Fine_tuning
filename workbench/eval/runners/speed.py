#!/usr/bin/env python3
"""Speed benchmark runner (T4.7).

Measures tokens/s at various context levels.
"""
import json, time
from pathlib import Path
from eval.client import LlamaClient
from eval.config import EC

def make_prompt(context_lines: int) -> str:
    """Generate a prompt with roughly `context_lines` lines."""
    lines = []
    for i in range(context_lines):
        lines.append(f"// Line {i}: sample code for context padding.")
    lines.append("Read the file test.py")
    return "\n".join(lines)

def measure_speed(client: LlamaClient, label: str, n_lines: int) -> dict:
    """Measure tokens/s for a given prompt length."""
    prompt = make_prompt(n_lines)
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": prompt},
    ]
    result = client.chat(messages, temperature=EC.temperature, max_tokens=50)
    
    if "error" in result:
        print(f"  {label}: ERROR - {result['error']}")
        return {"label": label, "tok_s": 0, "error": result["error"]}
    
    completion_tok = result.get("completion_tokens", 0)
    elapsed = result.get("elapsed", 0.001)
    tok_s = completion_tok / elapsed
    
    print(f"  {label}: {tok_s:.1f} tok/s ({completion_tok} tok / {elapsed:.2f}s)")
    return {
        "label": label,
        "prompt_lines": n_lines,
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": completion_tok,
        "elapsed": round(elapsed, 3),
        "tok_s": round(tok_s, 1),
    }

def evaluate():
    client = LlamaClient()
    if not client.health():
        print("FAIL: llama-server not reachable")
        return []
    
    results = []
    # Approximate: ~8 tokens/line
    for n_lines, label in [(50, "4k"), (100, "8k"), (200, "16k"), (400, "32k")]:
        r = measure_speed(client, label, n_lines)
        results.append(r)
    
    # Gate check: speed_8k >= 30 tok/s
    gate = EC.gates[5]
    eightk = next((r for r in results if r.get("label") == "8k"), None)
    if eightk and eightk.get("tok_s", 0) >= gate.min_pct:
        print(f"\nGate ({gate.name}): PASS")
    elif eightk:
        print(f"\nGate ({gate.name}): FAIL ({eightk.get('tok_s', 0):.1f} < {gate.min_pct})")
    else:
        print(f"\nGate ({gate.name}: could not measure 8k)")
    
    return results

if __name__ == "__main__":
    evaluate()
