"""Eval config and suite definitions (T4.1, T4.2, T4.4).

Frozen test suites with locked pass criteria.
"""
from dataclasses import dataclass, field
from typing import Optional

BASE_MODEL = "NousResearch/Hermes-2-Pro-Llama-3-8B"
ALTERNATIVE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
]

ADAPTER_ID = "hermes2pro-llama3-8b"

TEMP = 0.0
SEED = 42
SANDBOX_MAX_ROUNDS = 8

@dataclass
class Gate:
    name: str
    min_pct: float
    max_pct: Optional[float] = None
    unit: str = "percent"
    note: str = ""

@dataclass
class EvalConfig:
    base_model: str = BASE_MODEL
    adapter_id: str = ADAPTER_ID
    temperature: float = TEMP
    seed: int = SEED
    max_rounds: int = SANDBOX_MAX_ROUNDS
    
    gates: list = field(default_factory=lambda: [
        # Section 5 gates
        Gate("tool_syntax", 100.0, note="T1 valid JSON, T2 tool name, T3 args"),
        Gate("anti_loop", 90.0, note="L2: different tool call or stop+report"),
        Gate("security_catch", 85.0, note="3a: REJECT + correct type + line"),
        Gate("security_fp", 0.0, 10.0, note="3b: clean diff wrongly REJECTed"),
        Gate("json_validity", 100.0, note="3c: valid JSON, no grammar"),
        Gate("speed_8k", 30.0, note="tokens/s at 8k context"),
        Gate("regression", 97.0, note="HumanEval+ pass@1 within 3 points"),
    ])
    
    suites: dict = field(default_factory=lambda: {
        "tool_syntax": {"min_size": 100, "path": "eval/suites/tool_syntax/v001"},
        "anti_loop": {"min_size": 40, "path": "eval/suites/anti_loop/v001"},
        "security_vuln": {"min_size": 100, "path": "eval/suites/security_vuln/v001"},
        "security_clean": {"min_size": 100, "path": "eval/suites/security_clean/v001"},
        "plan_json": {"min_size": 50, "path": "eval/suites/plan_json/v001"},
    })
    
    # For overlap check
    train_groups: set = field(default_factory=set)
    
EC = EvalConfig()
