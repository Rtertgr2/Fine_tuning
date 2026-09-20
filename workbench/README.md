# Fine-tuning Workbench

วงจรเตรียมข้อมูล → เทรน → ประเมิน → deploy → เก็บเคสผิดพลาด สำหรับ fine-tune โมเดล agent
(Plan → Code → Review) บนเครื่อง VRAM 12 GB

**สถานะ: ระยะ 1 (Dataset Studio) กำลังดำเนินการ** — ดู `docs/decisions.md` สำหรับบันทึกการตัดสินใจ

## หลักการออกแบบ

- **Model-agnostic**: ทุกอย่างที่ขึ้นกับโมเดลอยู่ใน `backend/adapters/` — validator, API และ
  pipeline ไม่รู้จักโมเดลเลย เปลี่ยนโมเดล = เปลี่ยน adapter name
- **Chat template ล็อกตัวเดียว**: template เก็บเป็นไฟล์ jinja ที่คัดมาจาก HF repo ตรงๆ
  (`backend/adapters/templates/`) มีเทสต์พิสูจน์ว่า render ตรงกับ `apply_chat_template`
  ของโมเดลจริงทุก byte
- **ทุกอย่างมี hash**: ตัวอย่าง (content_hash), template, ไฟล์ dataset (sha256 ใน manifest)

## โครงสร้าง

```
backend/
  adapters/        # ModelAdapter: chat template, tool-call format, tokenizer
    templates/     # ไฟล์ jinja คัดจาก HF repo (default + tool_use)
  app/             # FastAPI: routers, services, SQLite
  validators/      # กฎตรวจ C1-C6, T1-T6, L1-L4, P1-P4, S1-S7
  tools/           # registry ของ 3 tools (git_checkout, read_file, write_file)
runner/            # (ระยะ 3) ตัวรันเทรนบน GPU
eval/              # (ระยะ 4) ชุดทดสอบ + รายงาน
data/              # examples.db, datasets/v0001/, gold/
models/            # tokenizers/, (ระยะ 3-5) adapters + gguf
docs/              # decisions.md, คู่มือ
tests/             # pytest
```

## เริ่มใช้งาน

```bash
cd workbench
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" pydantic jinja2 \
    pytest transformers huggingface_hub httpx

# รัน server
.venv/bin/python -m uvicorn backend.app.main:app --reload --port 8300
# เปิด http://127.0.0.1:8300/docs

# รันเทสต์
.venv/bin/python -m pytest tests/
```

Tokenizer จะถูกดาวน์โหลดอัตโนมัติเมื่อใช้ครั้งแรก (เก็บที่ `models/tokenizers/`)
ถ้าไม่มีการเชื่อมต่อ จะทำงานแบบ offline ได้ แต่ C4 (จำกัด token) จะถูกข้ามและรายงานว่า skip

## API หลัก (ระยะ 1)

| Method | Path | หน้าที่ |
|---|---|---|
| GET | /examples | รายการ + กรองตามหมวด/สถานะ/คำค้น |
| POST | /examples | เพิ่มตัวอย่าง (ตรวจให้ในตัว คืน report) |
| PUT | /examples/{id} | แก้ไข |
| POST | /examples/{id}/status | เปลี่ยนสถานะ (draft/approved/rejected) |
| POST | /validate | ตรวจร่างโดยไม่บันทึก |
| POST | /import | นำเข้า .jsonl |
| GET | /export | ส่งออก .jsonl |
| GET | /stats | สรุปสัดส่วนและคุณภาพ |
| POST | /datasets | สร้างเวอร์ชันชุดข้อมูล (แบ่ง train/val) |
| GET | /datasets/{id}/export | ดาวน์โหลด train/val/zip |

## ตั้งค่าผ่าน environment

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `WORKBENCH_ADAPTER` | `hermes2pro-llama3-8b` | adapter ที่ใช้ (`/adapters` แสดงรายการ) |
| `WORKBENCH_MAX_SEQ_LEN` | `8192` | เพดาน token ของ C4 |
| `WORKBENCH_DB` | `data/examples.db` | ไฟล์ SQLite |
| `WORKBENCH_VAL_RATIO` | `0.1` | สัดส่วน val ตอนแบ่งข้อมูล |
| `WORKBENCH_NEAR_DUP` | `0.85` | เกณฑ์ near-duplicate (C6) |
| `WORKBENCH_S7_MIN` | `3` | จำนวนตัวอย่างขั้นต่ำต่อประเภทช่องโหว่ (S7) |

## ตัวอย่างการเรียก API

```bash
# เพิ่มตัวอย่างหมวด plan
curl -X POST localhost:8300/examples -H 'Content-Type: application/json' -d '{
  "category": "plan",
  "messages": [
    {"role": "system", "content": "You are a plan reviewer. Answer with JSON only."},
    {"role": "user", "content": "ตรวจแผน release v0.2.0"},
    {"role": "assistant", "content": "{\"status\": \"APPROVED\", \"target_version\": \"v0.2.0\", \"critique\": \"...\", \"final_plan\": \"...\"}"}
  ],
  "group_id": "plan/release-1"
}'

# สร้าง dataset จากตัวอย่างที่ approved ทั้งหมด
curl -X POST localhost:8300/datasets -H 'Content-Type: application/json' -d '{"seed": 42}'
# -> data/datasets/v0001/{train.jsonl,val.jsonl,manifest.json}
```
