"""Tests for Phase 3: Training Launcher."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app import config
from backend.app.services.training_config import (
    DEFAULT_BASE_MODEL,
    LoraConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)
from backend.app.services.preflight import (
    pf1_dataset_hash,
    pf2_render_samples,
    pf3_loss_mask,
    pf4_token_length,
    run_preflight,
)
from backend.app.services import training as training_svc
from runner.metrics import MetricsTracker, load_metrics


# ---------------------------------------------------------------------------
# Config tests (T3.1)
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.run_name == "r001"
        assert cfg.base_model == DEFAULT_BASE_MODEL
        assert cfg.dataset_version == "v0001"
        assert cfg.seed == 3407
        assert cfg.loss_masking == "assistant_only"

    def test_validate_valid_config(self):
        cfg = TrainingConfig()
        errors = cfg.validate()
        assert errors == []

    def test_validate_invalid_lora_r(self):
        cfg = TrainingConfig(lora=LoraConfig(r=15))
        errors = cfg.validate()
        assert any("power of 2" in e for e in errors)

    def test_validate_invalid_seed(self):
        cfg = TrainingConfig(seed=-1)
        errors = cfg.validate()
        assert any("seed" in e for e in errors)

    def test_validate_invalid_loss_masking(self):
        cfg = TrainingConfig(loss_masking="invalid")  # type: ignore
        errors = cfg.validate()
        assert any("loss_masking" in e for e in errors)

    def test_validate_invalid_epochs(self):
        cfg = TrainingConfig(train=TrainConfig(epochs=0))
        errors = cfg.validate()
        assert any("epochs" in e for e in errors)

    def test_to_dict_roundtrip(self):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg2.run_name == cfg.run_name
        assert cfg2.base_model == cfg.base_model
        assert cfg2.lora.r == cfg.lora.r

    def test_save_and_load_yaml(self, tmp_path):
        cfg = TrainingConfig(run_name="r042")
        yaml_path = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_path)
        assert yaml_path.exists()

        cfg2 = TrainingConfig.load_yaml(yaml_path)
        assert cfg2.run_name == "r042"
        assert cfg2.base_model == cfg.base_model


# ---------------------------------------------------------------------------
# Metrics tests (T3.4)
# ---------------------------------------------------------------------------

class TestMetricsTracker:
    def test_log_and_retrieve(self, tmp_path):
        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        tracker.log(step=1, train_loss=2.5, learning_rate=2e-4)
        tracker.log(step=2, train_loss=2.0, eval_loss=2.1)

        assert len(tracker.entries) == 2
        assert tracker.get_latest().step == 2
        assert tracker.best_eval_loss == 2.1

    def test_load_metrics(self, tmp_path):
        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        tracker.log(step=1, train_loss=3.0)
        tracker.log(step=2, train_loss=2.5)

        loaded = load_metrics(tmp_path / "metrics.jsonl")
        assert len(loaded) == 2
        assert loaded[0]["step"] == 1
        assert loaded[1]["train_loss"] == 2.5

    def test_load_missing_file(self, tmp_path):
        loaded = load_metrics(tmp_path / "nonexistent.jsonl")
        assert loaded == []

    def test_best_step(self, tmp_path):
        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        tracker.log(step=10, eval_loss=2.0)
        tracker.log(step=20, eval_loss=1.5)
        tracker.log(step=30, eval_loss=1.8)

        assert tracker.best_step == 20
        assert tracker.best_eval_loss == 1.5


# ---------------------------------------------------------------------------
# Training service tests (T3.4)
# ---------------------------------------------------------------------------

class TestTrainingService:
    def test_create_run(self, conn, tmp_workbench):
        record = training_svc.create_run(
            conn,
            config_json={"run_name": "test", "base_model": "test/model"},
            dataset_version="v0001",
        )
        assert record["run_id"] == "r001"
        assert record["status"] == "pending"
        assert record["dataset_version"] == "v0001"

    def test_get_run(self, conn, tmp_workbench):
        training_svc.create_run(conn, config_json={"run_name": "test"})
        record = training_svc.get_run(conn, "r001")
        assert record is not None
        assert record["run_id"] == "r001"

    def test_get_run_not_found(self, conn):
        assert training_svc.get_run(conn, "r999") is None

    def test_list_runs(self, conn, tmp_workbench):
        training_svc.create_run(conn, config_json={"run_name": "a"})
        training_svc.create_run(conn, config_json={"run_name": "b"})
        runs = training_svc.list_runs(conn)
        assert len(runs) == 2

    def test_list_runs_with_filter(self, conn, tmp_workbench):
        training_svc.create_run(conn, config_json={"run_name": "a"})
        training_svc.create_run(conn, config_json={"run_name": "b"})
        training_svc.update_run(conn, "r001", status="running")

        pending = training_svc.list_runs(conn, status_filter="pending")
        running = training_svc.list_runs(conn, status_filter="running")
        assert len(pending) == 1
        assert len(running) == 1
        assert running[0]["run_id"] == "r001"

    def test_stop_run(self, conn, tmp_workbench):
        training_svc.create_run(conn, config_json={"run_name": "test"})
        record = training_svc.stop_run(conn, "r001")
        assert record is not None
        assert record["status"] == "stopped"

    def test_resume_run(self, conn, tmp_workbench, monkeypatch):
        training_svc.create_run(conn, config_json={"run_name": "test"})
        training_svc.stop_run(conn, "r001")
        monkeypatch.setattr(training_svc, "spawn_training_process", lambda *args: None)
        record = training_svc.resume_run(conn, "r001")
        assert record is not None
        assert record["status"] == "pending"

    def test_resume_running_run_fails(self, conn, tmp_workbench):
        training_svc.create_run(conn, config_json={"run_name": "test"})
        training_svc.update_run(conn, "r001", status="running")
        with pytest.raises(ValueError, match="cannot resume"):
            training_svc.resume_run(conn, "r001")

    def test_delete_run(self, conn, tmp_workbench):
        training_svc.create_run(conn, config_json={"run_name": "test"})
        assert training_svc.delete_run(conn, "r001") is True
        assert training_svc.get_run(conn, "r001") is None

    def test_sequential_run_ids(self, conn, tmp_workbench):
        r1 = training_svc.create_run(conn, config_json={"run_name": "a"})
        r2 = training_svc.create_run(conn, config_json={"run_name": "b"})
        r3 = training_svc.create_run(conn, config_json={"run_name": "c"})
        assert r1["run_id"] == "r001"
        assert r2["run_id"] == "r002"
        assert r3["run_id"] == "r003"


# ---------------------------------------------------------------------------
# Pre-flight tests (T3.2)
# ---------------------------------------------------------------------------

class TestPreflight:
    def test_pf1_missing_dataset(self, conn):
        cfg = TrainingConfig(dataset_version="v9999")
        report = pf1_dataset_hash(cfg)
        assert not report.results[0].passed
        assert report.results[0].code == "PF1"
        assert not report.can_proceed

    def test_pf2_missing_dataset(self, conn):
        cfg = TrainingConfig(dataset_version="v9999")
        report = pf2_render_samples(cfg)
        assert not report.results[0].passed

    def test_pf3_loss_mask_all_fails(self, conn):
        cfg = TrainingConfig(loss_masking="all")  # type: ignore
        report = pf3_loss_mask(cfg)
        assert not report.results[0].passed
        assert report.results[0].critical
        assert not report.can_proceed

    def test_pf3_missing_dataset(self, conn):
        cfg = TrainingConfig(dataset_version="v9999")
        report = pf3_loss_mask(cfg)
        assert not report.results[0].passed

    def test_pf4_missing_dataset(self, conn):
        cfg = TrainingConfig(dataset_version="v9999")
        report = pf4_token_length(cfg)
        assert not report.results[0].passed

    def test_run_preflight_all_missing(self, conn):
        cfg = TrainingConfig(dataset_version="v9999")
        report = run_preflight(cfg)
        assert not report.can_proceed
        assert any(r.code == "PF1" and not r.passed for r in report.results)
        assert any(r.code == "PF3" and not r.passed for r in report.results)


# ---------------------------------------------------------------------------
# Script generation tests (T3.3)
# ---------------------------------------------------------------------------

class TestScriptGeneration:
    def test_generate_script(self, tmp_path):
        from scripts.generate_training_script import generate_script

        cfg = TrainingConfig(run_name="test_run")
        output = tmp_path / "train.py"
        generate_script(cfg, output)

        assert output.exists()
        content = output.read_text()
        assert "test_run" in content
        assert "Trainer(" in content
        assert "LoraConfig" in content
        assert "assistant_only" in content
        assert "response_template =" not in content
        assert "prepare_model_for_kbit_training" in content
        assert "EarlyStoppingCallback" in content
        assert "metrics.jsonl" in content
        assert "run_manifest.json" in content
        import py_compile
        py_compile.compile(str(output), doraise=True)

    def test_generate_colab(self, tmp_path):
        from scripts.generate_training_script import generate_script

        cfg = TrainingConfig(run_name="test_run")
        output = tmp_path / "train_colab.py"
        generate_script(cfg, output, colab=True)

        assert output.exists()
        content = output.read_text()
        assert "#@title" in content
        assert "'pip', 'install'" in content
        assert "subprocess.check_call" in content
        assert "build_example" in content
        import py_compile
        py_compile.compile(str(output), doraise=True)


# ---------------------------------------------------------------------------
# Overfit detection tests (T3.5)
# ---------------------------------------------------------------------------

class TestOverfitDetection:
    def test_detect_overfit_increasing_eval(self, tmp_path):
        from runner.trainer import _detect_overfit

        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        # Simulate overfitting: train loss decreases, eval loss increases
        tracker.log(step=10, train_loss=2.0, eval_loss=1.8)
        tracker.log(step=20, train_loss=1.5, eval_loss=1.9)
        tracker.log(step=30, train_loss=1.0, eval_loss=2.0)
        tracker.log(step=40, train_loss=0.5, eval_loss=2.1)

        is_overfit, msg = _detect_overfit(tracker, patience=3)
        assert is_overfit
        assert "Overfit detected" in msg

    def test_no_overfit_decreasing_both(self, tmp_path):
        from runner.trainer import _detect_overfit

        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        tracker.log(step=10, train_loss=2.0, eval_loss=2.0)
        tracker.log(step=20, train_loss=1.5, eval_loss=1.5)
        tracker.log(step=30, train_loss=1.0, eval_loss=1.0)
        tracker.log(step=40, train_loss=0.5, eval_loss=0.5)

        is_overfit, msg = _detect_overfit(tracker, patience=3)
        assert not is_overfit

    def test_not_enough_eval_points(self, tmp_path):
        from runner.trainer import _detect_overfit

        tracker = MetricsTracker(tmp_path / "metrics.jsonl")
        tracker.log(step=10, train_loss=2.0, eval_loss=1.8)

        is_overfit, msg = _detect_overfit(tracker, patience=3)
        assert not is_overfit
