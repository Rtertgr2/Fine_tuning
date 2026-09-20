"""API schemas (pydantic)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.services.training_config import DEFAULT_BASE_MODEL

Category = Literal["tool", "loop", "plan", "sec"]
Status = Literal["draft", "approved", "rejected"]
Source = Literal["manual", "generated", "active_learning"]

CATEGORIES: tuple[str, ...] = ("tool", "loop", "plan", "sec")

ModelStatus = Literal["candidate", "staging", "production", "retired"]
QuantLevel = Literal["Q4_K_M", "Q5_K_M", "Q8_0", "Q2_K", "Q3_K_M", "Q6_K"]

MODEL_STATUSES: tuple[str, ...] = ("candidate", "staging", "production", "retired")
QUANT_LEVELS: tuple[str, ...] = ("Q4_K_M", "Q5_K_M", "Q8_0", "Q2_K", "Q3_K_M", "Q6_K")

TrainingStatus = Literal["pending", "running", "completed", "stopped", "failed"]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ExampleIn(BaseModel):
    category: Category
    messages: list[Message] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    source: Source = "manual"
    group_id: str | None = None


class ExamplePatch(BaseModel):
    category: Category | None = None
    messages: list[Message] | None = None
    tools: list[dict[str, Any]] | None = None
    source: Source | None = None
    group_id: str | None = None


class ExampleOut(BaseModel):
    id: str
    category: Category
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    source: Source
    group_id: str | None = None
    status: Status
    content_hash: str
    token_count: int | None = None
    created_at: str
    updated_at: str


class ExampleListOut(BaseModel):
    total: int
    items: list[ExampleOut]


class ViolationOut(BaseModel):
    code: str
    level: Literal["err", "warn"]
    message: str
    location: str | None = None


class ValidationReportOut(BaseModel):
    ok: bool
    violations: list[ViolationOut] = []
    skipped: list[str] = []


class ExampleWithReport(BaseModel):
    example: ExampleOut
    report: ValidationReportOut


class StatusChange(BaseModel):
    status: Status


class DatasetCreateIn(BaseModel):
    name: str | None = None
    seed: int | None = None
    val_ratio: float | None = Field(default=None, gt=0, lt=0.5)
    categories: list[Category] | None = None


class DatasetOut(BaseModel):
    id: str
    name: str | None = None
    seed: int
    created_at: str
    manifest: dict[str, Any]


class ImportResult(BaseModel):
    imported: int
    failed: int
    errors: list[dict[str, Any]] = []
    example_ids: list[str] = []


# ── Model Registry Schemas (Phase 4) ──────────────────────────────────────


class ModelRegisterIn(BaseModel):
    version: str = Field(..., pattern=r"^v\d+\.\d+\.\d+$")
    dataset_version: str | None = None
    train_run: str | None = None
    llama_cpp_commit: str | None = None
    quant: QuantLevel = "Q5_K_M"
    gguf_sha256: str | None = None
    eval_report: str | None = None
    status: ModelStatus = "candidate"


class ModelPromoteIn(BaseModel):
    target_status: ModelStatus = "production"


class ModelOut(BaseModel):
    version: str
    dataset_version: str | None = None
    train_run: str | None = None
    llama_cpp_commit: str | None = None
    quant: QuantLevel
    gguf_sha256: str | None = None
    eval_report: str | None = None
    status: ModelStatus
    manifest: dict[str, Any]
    created_at: str
    promoted_at: str | None = None


class ModelListOut(BaseModel):
    total: int
    items: list[ModelOut]


class PromoteResult(BaseModel):
    version: str
    previous_status: ModelStatus
    new_status: ModelStatus
    previous_production: str | None = None
    gate_passed: bool
    message: str


class RollbackResult(BaseModel):
    version: str
    previous_production: str | None = None
    message: str


# ── Review Queue Schemas (Phase 2) ─────────────────────────────────────


class ReviewQueueItem(BaseModel):
    id: str
    category: str
    example_data: dict[str, Any]
    status: str
    reason: str | None = None
    reviewer: str | None = None
    group_id: str | None = None
    created_at: str
    updated_at: str


class ReviewQueueList(BaseModel):
    total: int
    items: list[ReviewQueueItem]


# ── Training Schemas (Phase 3) ───────────────────────────────────────────


class TrainingRunIn(BaseModel):
    config: dict[str, Any] | None = None


class TrainingRunOut(BaseModel):
    run_id: str
    config: dict[str, Any]
    status: TrainingStatus
    dataset_version: str | None = None
    adapter_path: str | None = None
    metrics_path: str | None = None
    report_path: str | None = None
    created_at: str
    completed_at: str | None = None


class TrainingRunListOut(BaseModel):
    total: int
    items: list[TrainingRunOut]


class PreflightResultOut(BaseModel):
    code: str
    passed: bool
    message: str
    details: dict[str, Any] = {}
    critical: bool = False


class PreflightReportOut(BaseModel):
    can_proceed: bool
    all_passed: bool
    results: list[PreflightResultOut]


class MetricEntryOut(BaseModel):
    step: int
    train_loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    epoch: float | None = None
    elapsed_seconds: float | None = None
    timestamp: str = ""
