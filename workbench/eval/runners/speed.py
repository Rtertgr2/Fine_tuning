#!/usr/bin/env python3
"""Speed benchmark at measured context sizes (plan 04 §6)."""
from __future__ import annotations

from eval.client import LlamaClient
from eval.config import EC

TARGETS = [(4096, "4k"), (8192, "8k"), (16384, "16k"), (32768, "32k")]


def make_prompt(target_tokens: int, client: LlamaClient | None = None) -> tuple[str, int | None]:
    """Build a prompt near a target token count using llama-server /tokenize.

    Falls back to a clearly marked character-based approximation only when the
    server's tokenize endpoint is unavailable; the response's actual prompt
    token count remains authoritative in the report.
    """
    client = client or LlamaClient()
    line = "# context padding for a coding-agent speed benchmark.\n"
    line_tokens = client.tokenize(line)
    if line_tokens:
        estimated_per_line = max(1, len(line_tokens))
        count = max(1, target_tokens // estimated_per_line)
    else:
        # about four bytes per token for ordinary ASCII text
        count = max(1, target_tokens * 4 // max(1, len(line)))
    chunks = [line] * count
    prompt = "".join(chunks)
    measured_tokens = client.tokenize(prompt)
    measured: int | None = len(measured_tokens) if measured_tokens else None
    # Refine the approximate context count using server tokenizer feedback.
    for _ in range(3):
        if measured is None or measured == 0 or abs(measured - target_tokens) <= max(64, target_tokens * 0.02):
            break
        ratio = target_tokens / measured
        count = max(1, int(count * ratio))
        prompt = line * count
        measured_tokens = client.tokenize(prompt)
        measured = len(measured_tokens) if measured_tokens else None
    return prompt, measured


def _decode_rate(result: dict) -> float:
    timings = result.get("timings") or {}
    for key in ("predicted_per_second", "tokens_per_second", "generation_tokens_per_second"):
        try:
            value = float(timings.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    try:
        tokens = float(result.get("completion_tokens", 0))
        elapsed = max(float(result.get("elapsed", 0)), 0.001)
    except (TypeError, ValueError):
        return 0.0
    return tokens / elapsed


def measure_speed(client: LlamaClient, label: str, target_tokens: int) -> dict:
    prompt, measured = make_prompt(target_tokens, client)
    response = client.chat(
        [
            {"role": "system", "content": "You are a coding agent. Reply briefly."},
            {"role": "user", "content": prompt},
        ],
        temperature=EC.temperature,
        max_tokens=50,
    )
    if response.get("error"):
        return {
            "label": label,
            "target_context_tokens": target_tokens,
            "tok_s": 0.0,
            "error": response["error"],
            "tokenize_estimate": measured,
        }
    return {
        "label": label,
        "target_context_tokens": target_tokens,
        "prompt_tokens": response.get("prompt_tokens", 0),
        "completion_tokens": response.get("completion_tokens", 0),
        "elapsed": round(float(response.get("elapsed", 0)), 3),
        "tok_s": round(_decode_rate(response), 2),
        "tokenize_estimate": measured,
        "timings": response.get("timings", {}),
    }


def evaluate(client: LlamaClient | None = None) -> dict:
    client = client or LlamaClient()
    results = [measure_speed(client, label, target) for target, label in TARGETS]
    eightk = next((result for result in results if result["label"] == "8k"), None)
    return {
        "suite": "speed",
        "speed_8k": eightk.get("tok_s", 0.0) if eightk else 0.0,
        "contexts": results,
    }


if __name__ == "__main__":
    result = evaluate()
    for context in result["contexts"]:
        print(f"{context['label']}: {context['tok_s']:.1f} tok/s, prompt={context.get('prompt_tokens', 'n/a')} tokens")
    threshold = EC.gates[5].min_pct
    speed = result["speed_8k"]
    print(f"Gate: {'PASS' if speed >= threshold else 'FAIL'} ({speed:.1f} / {threshold:.1f} tok/s @8k)")
