"""Dataset build (T1.8) at the service level."""

from __future__ import annotations

import json

import pytest

from backend.app import schemas
from backend.app.services import datasets as ds_service
from backend.app.services import examples as svc
from tests.conftest import read_call, tool_call_block, write_call


def _tool_payload(i: int) -> schemas.ExampleIn:
    return schemas.ExampleIn(
        category="tool",
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {
                "role": "user",
                "content": f"งานที่ {i}: checkout repo/g{i} แล้วอ่าน f{i}.py เขียน g{i}.py ใหม่",
            },
            {
                "role": "assistant",
                "content": tool_call_block(
                    "git_checkout", {"repo": f"repo/g{i}", "branch": "main"}
                ),
            },
            {"role": "tool", "content": f"checked out repo/g{i} at main\n"},
            {"role": "assistant", "content": read_call(f"f{i}.py")},
            {"role": "tool", "content": f"# content of f{i}\n"},
            {
                "role": "assistant",
                "content": write_call(f"g{i}.py", f"print({i})\nprint('done')\n"),
            },
            {"role": "tool", "content": f"wrote g{i}.py\n"},
            {"role": "assistant", "content": f"เสร็จแล้วครับ เขียน g{i}.py เรียบร้อย"},
        ],
        group_id=f"repo/g{i}",
    )


def _make_approved(conn, n: int = 6):
    ids = []
    for i in range(n):
        record, report = svc.create_example(conn, _tool_payload(i))
        assert report.ok, [x.message for x in report.errors]
        svc.set_status(conn, record["id"], "approved")
        ids.append(record["id"])
    return ids


def test_build_writes_files_and_manifest(conn, tmp_workbench):
    _make_approved(conn, 6)
    record = ds_service.build_dataset(conn, seed=42)

    assert record["id"] == "v0001"
    manifest = record["manifest"]
    assert manifest["seed"] == 42
    assert manifest["counts"]["total"] == 6
    assert manifest["files"]["train"]["sha256"]
    assert manifest["template_hashes"]["tool_use"]

    base = tmp_workbench / "datasets" / "v0001"
    assert (base / "train.jsonl").exists()
    assert (base / "val.jsonl").exists()
    assert (base / "manifest.json").exists()
    manifest_on_disk = json.loads((base / "manifest.json").read_text())
    assert manifest_on_disk["dataset_id"] == "v0001"

    lines = (base / "train.jsonl").read_text().splitlines() + (
        base / "val.jsonl"
    ).read_text().splitlines()
    assert len(lines) == 6


def test_no_group_leakage_in_files(conn, tmp_workbench):
    _make_approved(conn, 8)
    ds_service.build_dataset(conn, seed=7, val_ratio=0.25)
    base = tmp_workbench / "datasets" / "v0001"

    def groups(path):
        out = set()
        for line in path.read_text().splitlines():
            out.add(json.loads(line)["meta"]["group"])
        return out

    train_groups = groups(base / "train.jsonl")
    val_groups = groups(base / "val.jsonl")
    assert not (train_groups & val_groups)


def test_deterministic_with_same_seed(conn, tmp_workbench):
    _make_approved(conn, 8)
    a = ds_service.build_dataset(conn, seed=99)
    b = ds_service.build_dataset(conn, seed=99)
    assert a["manifest"]["counts"] == b["manifest"]["counts"]
    assert a["manifest"]["files"]["train"]["sha256"] == b["manifest"]["files"]["train"]["sha256"]
    assert a["manifest"]["files"]["val"]["sha256"] == b["manifest"]["files"]["val"]["sha256"]


def test_different_seed_changes_split(conn, tmp_workbench):
    _make_approved(conn, 10)
    ds_service.build_dataset(conn, seed=1, val_ratio=0.3)
    ds_service.build_dataset(conn, seed=2, val_ratio=0.3)
    split1 = json.loads(
        conn.execute("SELECT split_json FROM dataset_versions WHERE id='v0001'").fetchone()[0]
    )
    split2 = json.loads(
        conn.execute("SELECT split_json FROM dataset_versions WHERE id='v0002'").fetchone()[0]
    )
    assert split1 != split2


def test_invalid_examples_excluded(conn, tmp_workbench):
    _make_approved(conn, 3)
    bad, _report = svc.create_example(
        conn,
        schemas.ExampleIn(
            category="plan",
            messages=[
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "not json at all"},
            ],
            group_id="plan/bad",
        ),
    )
    svc.set_status(conn, bad["id"], "approved")

    record = ds_service.build_dataset(conn, seed=5)
    manifest = record["manifest"]
    assert bad["id"] in manifest["selection"]["excluded_invalid"]
    assert manifest["counts"]["total"] == 3


def test_build_requires_approved(conn):
    with pytest.raises(ValueError):
        ds_service.build_dataset(conn, seed=1)
