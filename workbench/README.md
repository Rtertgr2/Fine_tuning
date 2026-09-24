# Fine-tuning Workbench

วงจรเตรียมข้อมูล → เทรน → ประเมิน → deploy → เก็บเคสผิดพลาด สำหรับ fine-tune โมเดล agent (Plan → Code → Review) บนเครื่องเป้าหมาย VRAM 12 GB

> **สถานะ ณ 2026-09-23: prototype ที่มีโครง backend หลาย phase แต่ยังไม่พร้อม production** — ต้องมี GPU/llama-server เพื่อวัดผลจริง และ eval gate จะปิดการ promote ไว้จนกว่าจะมีผล regression + end-to-end ครบ ดู `docs/decisions.md` และ `docs/execution-plan.md` ประกอบ

## สถานะตามแผน

| Phase | สถานะโค้ด | ข้อจำกัดสำคัญ |
|---|---|---|
| 1 Dataset Studio | ✅ | Validator, CRUD, group-aware split, manifest และ UI editor |
| 2 Data Pipeline | ⚠️ | Generators/review queue มีแล้ว; sandbox ปัจจุบันเป็น temp-directory isolation ใน process ไม่ใช่ Docker container ที่ตัด network จึงห้ามใช้กับโค้ดที่ไม่น่าเชื่อถือ |
| 3 Training Launcher | ⚠️ | สร้างสคริปต์, metrics, checkpoint/resume และ PF1–PF6; PF5 ต้องใช้ CUDA จริงและรัน 10 steps |
| 4 Eval Harness | ⚠️ | 5 frozen suites และ runner บางด่าน; HumanEval+/MBPP+ regression กับ E2E runner ยังไม่ครบ รายงานจะ fail-closed และ block production |
| 5 Deploy | ✅ | ตรวจ GGUF magic/hash, registry, candidate-gated promote, `models/current` symlink และ rollback; ต้องติดตั้ง llama.cpp เอง |
| 6 Active Learning | ✅ | Redact log, validate ก่อน approve, คุมสัดส่วน AL และแยก regression ตามกลุ่ม session/project |
| 7 UI v1 | ⚠️ | หน้า HTML มีภาพรวม, ตัวอย่าง และเทรน; หน้า eval/deploy/case queue ยังไม่ครบตาม UI plan |
| 8 Execution Plan | ✅ | ตารางงานและ Go/No-go อยู่ใน `docs/execution-plan.md` |

**ผลทดสอบล่าสุด: 238 passed** (`.venv/bin/python -m pytest tests/`). มี warning จาก `starlette.testclient`/`httpx` ที่ยังไม่กระทบผลเทสต์

## หลักการออกแบบ

- **Model/template mapping ชัดเจน**: adapter ที่รองรับอยู่ใน `backend/adapters/`; config จะไม่ fallback ไปใช้ template ผิดตระกูล
- **Template parity**: pre-flight เทียบผลจาก adapter กับ `tokenizer.apply_chat_template()` ของ tokenizer ที่เลือก
- **ทุกอย่างมี hash**: content, tokenizer template, dataset file, GGUF และรายงาน eval
- **Validator จุดเดียว**: UI, Dataset Studio, pipeline และ active-learning approval ใช้ backend validator ชุดเดียว
- **Gate fail-closed**: รายงานไม่มี measurement หรือไม่ผ่านข้อใดข้อหนึ่งจะ promote เป็น production ไม่ได้
- **ไม่ commit artifacts**: DB, dataset ที่สร้าง, checkpoints และ model outputs ถูก ignore; เก็บนอก Git

## โครงสร้าง

```
backend/
  adapters/        # model templates, tokenizers, adapter mapping
  app/             # FastAPI: routers, services, SQLite
  validators/      # C1-C6, T1-T6, L1-L4, P1-P4, S1-S7
  tools/           # tool registry + workspace sandbox
  pipeline/        # generators, review validation, contamination check
runner/            # training metrics and utilities
eval/              # frozen suites, judges, runners, reports
frontend/          # UI v1 (HTML/CSS/JS)
data/              # local SQLite, generated datasets, gold set
models/            # tokenizer files are tracked; model outputs are local-only
scripts/           # training-script generation, merge/quantize/deploy
```

## เริ่มใช้งาน

```bash
cd workbench
.venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8300

# UI:  http://127.0.0.1:8300/ui/
# API: http://127.0.0.1:8300/docs
# tests
.venv/bin/python -m pytest tests/
```

เมื่อต้องเปิด UI ผ่าน reverse proxy ให้ใช้ same-origin และอย่าเปิด service สู่ public network โดยไม่มี authentication

## Gold set และชุดข้อมูล

Gold set ที่ commit ไว้: 108 ตัวอย่าง 4 หมวด (tool:30, loop:28, plan:30, sec:20)

Dataset v0001 เดิม: seed=42, 97 train / 11 val, group leakage = 0

## Training

สร้าง run จะทำ PF1–PF6 ก่อนเริ่ม โดย PF5 ต้องใช้ GPU/CUDA บนเครื่องที่รัน workbench หากไม่มี GPU จะรายงานว่าไม่พร้อมและไม่เริ่มเทรน

สคริปต์ที่สร้างจะเขียน `config.yaml`, `dataset_manifest.json`, `run_manifest.json`, `env.txt`, `metrics.jsonl`, checkpoint, adapter และ `report.md`; loss labels สร้างเฉพาะช่วง assistant ทุก turn (ไม่ใช้ marker ที่ hardcode เฉพาะ Hermes)

## Eval Harness

Frozen suites: tool_syntax 100, anti_loop 40, security_vuln 100, security_clean 100, plan_json 50

ใช้ `--server` ระบุ llama-server ได้; runner ทดสอบทั้ง tool-call tags และ API tools:

```bash
PYTHONPATH=. .venv/bin/python -m eval.runners.run_baseline \
  --model "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --server "http://127.0.0.1:8080"
```

**สำคัญ:** ตอนนี้ runner สำหรับ regression และ E2E ยังไม่มี รายงานจึงระบุ gate ที่ขาดและถือว่าไม่ผ่านโดยตั้งใจ ห้ามถือผล suite ที่มีอยู่เพียงบางส่วนเป็น baseline สำหรับ deploy

## Deploy และ registry

Deploy output อยู่ใน `models/<version>/`; quantization หลายระดับสร้างเป็นไฟล์ `model-Q*.gguf` แต่ลงทะเบียนเลือกเพียงหนึ่ง quant ต่อ semantic version:

```bash
.venv/bin/python scripts/deploy.py r001 \
  --version v0.1.0 --dataset v0001 --register-quant Q5_K_M
```

การลงทะเบียนต้องระบุ GGUF จริงและ SHA256; ไฟล์ต้องอยู่ใน `models/<version>/` และเริ่มด้วย `GGUF` magic การแนบ eval report ทำได้ด้วย `POST /models/{version}/eval-report`; production promotion จะตรวจ artifact, hash และ gate ทุกข้อก่อนสลับ `models/current`

ไฟล์ GGUF, checkpoints, datasets และ `models/current` **ไม่ถูก commit เข้า Git**

## แผนโครงการ

ดู `Plan/00-overview.md` → `Plan/08-execution-plan.md` สำหรับ scope, ลำดับงาน และ Go/No-go
