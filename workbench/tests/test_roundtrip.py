"""JSONL round-trip (T1.6): export -> import into a fresh DB -> same content."""

from __future__ import annotations

import json

from backend.app import schemas
from backend.app.db import connect, init_db
from backend.app.services import examples as svc
from backend.app.services import io_jsonl


def _seed_examples(conn):
    svc.create_example(
        conn,
        schemas.ExampleIn(
            category="plan",
            messages=[
                {"role": "system", "content": "S"},
                {"role": "user", "content": "ตรวจแผน"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "status": "APPROVED",
                            "target_version": "v1.0.0",
                            "critique": "ดี",
                            "final_plan": "ทำต่อ",
                        }
                    ),
                },
            ],
            source="manual",
            group_id="plan/rt-1",
        ),
    )
    svc.create_example(
        conn,
        schemas.ExampleIn(
            category="sec",
            messages=[
                {"role": "system", "content": "S"},
                {"role": "user", "content": "diff"},
                {"role": "assistant", "content": '{"status": "PASS"}'},
            ],
            source="generated",
            group_id="repo/rt-2",
        ),
    )


def test_roundtrip_lossless(tmp_workbench):
    db1 = tmp_workbench / "a.db"
    init_db(db1)
    conn = connect(db1)
    _seed_examples(conn)
    rows = conn.execute("SELECT * FROM examples ORDER BY id").fetchall()
    text = io_jsonl.export_rows(rows)
    conn.close()

    db2 = tmp_workbench / "b.db"
    init_db(db2)
    conn2 = connect(db2)
    result = io_jsonl.import_lines(conn2, text)
    assert result["imported"] == 2
    assert result["failed"] == 0

    out1 = {r["id"]: json.loads(r["messages_json"]) for r in rows}
    out2 = {
        r["id"]: json.loads(r["messages_json"])
        for r in conn2.execute("SELECT * FROM examples").fetchall()
    }
    assert len(out1) == len(out2)
    for (id1, msgs1), (id2, msgs2) in zip(sorted(out1.items()), sorted(out2.items())):
        assert msgs1 == msgs2

    meta1 = {r["id"]: (r["category"], r["source"], r["group_id"], r["status"]) for r in rows}
    meta2 = {
        r["id"]: (r["category"], r["source"], r["group_id"], r["status"])
        for r in conn2.execute("SELECT * FROM examples").fetchall()
    }
    assert sorted(meta1.values()) == sorted(meta2.values())
    conn2.close()


def test_import_skips_duplicates(tmp_workbench):
    db = tmp_workbench / "c.db"
    init_db(db)
    conn = connect(db)
    _seed_examples(conn)
    rows = conn.execute("SELECT * FROM examples ORDER BY id").fetchall()
    text = io_jsonl.export_rows(rows)

    # re-import into the same db -> everything is a duplicate
    result = io_jsonl.import_lines(conn, text)
    assert result["imported"] == 0
    assert result["failed"] == 2
    conn.close()


def test_import_reports_bad_lines(tmp_workbench):
    db = tmp_workbench / "d.db"
    init_db(db)
    conn = connect(db)
    text = "not json\n" + json.dumps(
        {
            "category": "plan",
            "messages": [{"role": "user", "content": "x"}],
        }
    )
    result = io_jsonl.import_lines(conn, text)
    # second line has no assistant message but is structurally valid -> imported
    assert result["imported"] == 1
    assert result["failed"] == 1
    assert result["errors"][0]["line"] == 1
    conn.close()
