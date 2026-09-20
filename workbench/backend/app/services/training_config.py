"""Training run configuration schema and validation (T3.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# Defaults per Plan/03 §5
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_SEED = 3407
DEFAULT_MAX_SEQ_LENGTH = 8192


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    epochs: int = 2
    learning_rate: float = 2.0e-4
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = 0.05
    per_device_batch_size: int = 2
    grad_accum: int = 4
    max_seq_length: int = 8192
    eval_steps: int = 50
    save_steps: int = 50
    early_stopping_patience: int = 3


@dataclass
class TrainingConfig:
    run_name: str = "r001"
    base_model: str = DEFAULT_BASE_MODEL
    dataset_version: str = "v0001"
    seed: int = DEFAULT_SEED
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss_masking: Literal["assistant_only", "all"] = "assistant_only"

    def validate(self) -> list[str]:
        """Validate config, return list of errors (empty if valid)."""
        errors: list[str] = []
        if not self.run_name or not self.run_name.strip():
            errors.append("run_name must be non-empty")
        if not _is_power_of_2(self.lora.r):
            errors.append(f"LoRA r must be a power of 2, got {self.lora.r}")
        if self.lora.alpha <= 0:
            errors.append(f"LoRA alpha must be positive, got {self.lora.alpha}")
        if not 0.0 <= self.lora.dropout <= 1.0:
            errors.append(f"LoRA dropout must be in [0, 1], got {self.lora.dropout}")
        if self.seed <= 0:
            errors.append(f"seed must be a positive integer, got {self.seed}")
        if self.train.epochs < 1:
            errors.append(f"epochs must be >= 1, got {self.train.epochs}")
        if self.train.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.train.learning_rate}")
        if self.train.per_device_batch_size < 1:
            errors.append(f"per_device_batch_size must be >= 1, got {self.train.per_device_batch_size}")
        if self.train.grad_accum < 1:
            errors.append(f"grad_accum must be >= 1, got {self.train.grad_accum}")
        if self.train.max_seq_length < 256:
            errors.append(f"max_seq_length must be >= 256, got {self.train.max_seq_length}")
        if self.train.eval_steps < 1:
            errors.append(f"eval_steps must be >= 1, got {self.train.eval_steps}")
        if self.train.save_steps < 1:
            errors.append(f"save_steps must be >= 1, got {self.train.save_steps}")
        if self.train.early_stopping_patience < 1:
            errors.append(f"early_stopping_patience must be >= 1, got {self.train.early_stopping_patience}")
        if self.loss_masking not in ("assistant_only", "all"):
            errors.append(f"loss_masking must be 'assistant_only' or 'all', got {self.loss_masking!r}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "base_model": self.base_model,
            "dataset_version": self.dataset_version,
            "seed": self.seed,
            "quantization": {
                "load_in_4bit": self.quantization.load_in_4bit,
                "quant_type": self.quantization.quant_type,
                "double_quant": self.quantization.double_quant,
            },
            "lora": {
                "r": self.lora.r,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": list(self.lora.target_modules),
            },
            "train": {
                "epochs": self.train.epochs,
                "learning_rate": self.train.learning_rate,
                "lr_scheduler": self.train.lr_scheduler,
                "warmup_ratio": self.train.warmup_ratio,
                "per_device_batch_size": self.train.per_device_batch_size,
                "grad_accum": self.train.grad_accum,
                "max_seq_length": self.train.max_seq_length,
                "eval_steps": self.train.eval_steps,
                "save_steps": self.train.save_steps,
                "early_stopping_patience": self.train.early_stopping_patience,
            },
            "loss_masking": self.loss_masking,
        }

    def save_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        q = data.get("quantization", {})
        l = data.get("lora", {})
        t = data.get("train", {})
        return cls(
            run_name=data.get("run_name", "r001"),
            base_model=data.get("base_model", DEFAULT_BASE_MODEL),
            dataset_version=data.get("dataset_version", "v0001"),
            seed=data.get("seed", DEFAULT_SEED),
            quantization=QuantizationConfig(
                load_in_4bit=q.get("load_in_4bit", True),
                quant_type=q.get("quant_type", "nf4"),
                double_quant=q.get("double_quant", True),
            ),
            lora=LoraConfig(
                r=l.get("r", 16),
                alpha=l.get("alpha", 32),
                dropout=l.get("dropout", 0.0),
                target_modules=l.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]),
            ),
            train=TrainConfig(
                epochs=t.get("epochs", 2),
                learning_rate=t.get("learning_rate", 2.0e-4),
                lr_scheduler=t.get("lr_scheduler", "cosine"),
                warmup_ratio=t.get("warmup_ratio", 0.05),
                per_device_batch_size=t.get("per_device_batch_size", 2),
                grad_accum=t.get("grad_accum", 4),
                max_seq_length=t.get("max_seq_length", 8192),
                eval_steps=t.get("eval_steps", 50),
                save_steps=t.get("save_steps", 50),
                early_stopping_patience=t.get("early_stopping_patience", 3),
            ),
            loss_masking=data.get("loss_masking", "assistant_only"),
        )
