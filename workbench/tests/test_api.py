"""API smoke tests via TestClient."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import read_call, tool_call_block, write_call


@pytest.fixture()
def client(tmp_workbench):
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def _tool_payload(i: int, group: str | None = None) -> dict:
    return {
        "category": "tool",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {
                "role": "user",
                "content": f"งานที่ {i}: checkout repo/g{i} แล้วอ่าน f{i}.py เขียน g{i}.py ใหม่",
            },
            {
                "role": "assistant",
                "content": tool_call_block("git_checkout", {"repo": f"repo/g{i}", "branch": "main"}),
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
        "group_id": group or f"repo/g{i}",
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_adapters(client):
    r = client.get("/adapters")
    body = r.json()
    assert body["active"] == "hermes2pro-llama3-8b"
    assert any(a["name"] == "qwen2.5-7b-instruct" for a in body["available"])


def test_create_valid_example(client):
    r = client.post("/examples", json=_tool_payload(1))
    assert r.status_code == 201
    body = r.json()
    assert body["report"]["ok"] is True
    assert body["example"]["status"] == "draft"
    assert body["example"]["id"].startswith("ex_")


def test_create_invalid_example_reports(client):
    payload = {
        "category": "plan",
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "```json\n{}\n```"},
        ],
    }
    r = client.post("/examples", json=payload)
    assert r.status_code == 201
    body = r.json()
    assert body["report"]["ok"] is False
    codes = {x["code"] for x in body["report"]["violations"]}
    assert "P1" in codes


def test_validate_endpoint_does_not_store(client):
    r = client.post("/validate", json=_tool_payload(2))
    assert r.status_code == 200
    assert r.json()["ok"] is True
    r2 = client.get("/examples")
    assert r2.json()["total"] == 0


def test_validators_inventory(client):
    r = client.get("/validators")
    body = r.json()
    assert [x["code"] for x in body["tool"]][:3] == ["C1", "C2", "C3"]
    assert any(x["code"] == "S7" for x in body["sec"])


def test_list_filter_and_update(client):
    a = client.post("/examples", json=_tool_payload(3)).json()["example"]
    client.post("/examples", json=_tool_payload(4)).json()["example"]

    r = client.get("/examples", params={"category": "tool"})
    assert r.json()["total"] == 2

    r = client.put(f"/examples/{a['id']}", json={"group_id": "repo/renamed"})
    assert r.status_code == 200
    assert r.json()["example"]["group_id"] == "repo/renamed"

    r = client.post(f"/examples/{a['id']}/status", json={"status": "approved"})
    assert r.json()["status"] == "approved"

    r = client.get("/examples", params={"status": "approved"})
    assert r.json()["total"] == 1


def test_import_export_roundtrip_api(client):
    client.post("/examples", json=_tool_payload(5))
    r = client.get("/export")
    text = r.text
    assert text.count("\n") == 1

    r2 = client.post("/import", content=text.encode())
    body = r2.json()
    assert body["imported"] == 0 and body["failed"] == 1  # duplicate


def test_dataset_build_and_export(client, tmp_workbench):
    ids = []
    for i in range(6):
        ex = client.post("/examples", json=_tool_payload(10 + i)).json()["example"]
        client.post(f"/examples/{ex['id']}/status", json={"status": "approved"})
        ids.append(ex["id"])

    r = client.post("/datasets", json={"seed": 42})
    assert r.status_code == 201
    ds = r.json()
    assert ds["id"] == "v0001"
    manifest = ds["manifest"]
    assert manifest["counts"]["total"] == 6

    r = client.get("/datasets/v0001/export", params={"which": "train"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")

    r = client.get("/datasets/v0001/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"

    r = client.get("/datasets")
    assert len(r.json()) == 1


def test_stats(client):
    client.post("/examples", json=_tool_payload(21))
    client.post("/examples", json=_tool_payload(22))
    r = client.get("/stats")
    body = r.json()
    assert body["total"] == 2
    assert body["by_category"]["tool"] == 2
    assert "token_histogram" in body