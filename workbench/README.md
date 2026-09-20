# Fine-tuning Workbench

วงจรเตรียมข้อมูล → เทรน → ประเมิน → deploy → เก็บเคสผิดพลาด สำหรับ fine-tune โมเดล agent
(Plan → Code → Review) บนเครื่อง VRAM 12 GB

**สถานatus: ครบทุก 8 Phase** — ดู `docs/decisions.md` สำหรับบันทึกการตัดสินใจ

## สถานะโปรเจกต์

| Phase | สถานะ | รายละเอียด |
|---|---|---|
| 1 Dataset Studio | ✅ | Gold set 108, dataset v0001, UI editor |
| 2 Data Pipeline | ✅ | Sandbox, generators, review queue |
| 3 Training Launcher | ✅ | Config, pre-flight, Mode A/B |
| 4 Eval Harness | ✅ | 5 suites, runners, gates |
| 5 Deploy | ✅ | Merge/quantize, model registry |
| 6 Active Learning | ✅ | Activity log, redaction, case queue |
| 7 UI v1 | ✅ | HTML + CSS + JS (frontend/) |
| 8 Execution Plan | ✅ | docs/execution-plan.md |

**223 tests ผ่านทั้งหมด**

## หลักการออกแบบ

- **Model-agnostic**: ทุกอย่างที่ขึ้นกับโมเดลอยู่ใน `backend/adapters/` — เปลี่ยนโมเดล = เปลี่ยน adapter name
- **Chat template ล็อกตัวเดียว**: template เก็บเป็นไฟล์ jinja คัดมาจาก HF repo + เทสต์พิสูจน์ render ตรงกับ `apply_chat_template`
- **ทุกอย่างมี hash**: content_hash, template, dataset (sha256), model versions
- **ตัวตรวจชุดเดียว**: validator ฝั่ง backend เดียว UI แค่แสดงผล (แผน 07 ข้อ 6)
- **ไม่มี credential ใน repo**: สแกนด้วย detect-secrets patterns ทุกครั้งก่อน commit

## โครงสร้าง

```
backend/
  adapters/        # ModelAdapter: chat template, tool-call format, tokenizer
  app/             # FastAPI: routers, services, SQLite
  validators/      # กฎตรวจ C1-C6, T1-T6, L1-L4, P1-P4, S1-S7
  tools/           # registry ของ 3 tools + sandbox implementations
  pipeline/        # generators 4 หมวด + validator + cost control
  services/        # examples, datasets, models, review, training, cases, etc.
runner/            # Training runner (Mode B)
eval/              # Test suites + runners + report
frontend/          # UI v1 (HTML + CSS + JS)
data/              # examples.db, datasets/, gold/
scripts/           # gen_suites, merge_adapter, deploy, etc.
tests/             # pytest (223 tests)
docs/              # decisions.md, execution-plan.md
```

## เริ่มใช้งาน

```bash
cd workbench
.venv/bin/python -m uvicorn backend.app.main:app --port 8300

# UI: http://127.0.0.1:8300/ui/
# API docs: http://127.0.0.1:8300/docs
# รันเทสต์: .venv/bin/python -m pytest tests/
```

## Gold Set

108 ตัวอย่าง 4 หมวด — tool:30, loop:28, plan:30, sec:20 (all err=0)

Dataset v0001 (seed=42): 97 train / 11 val, no group leakage

## Eval Harness

5 frozen suites: tool 100, loop 40, sec_vuln 100, sec_clean 100, plan 50

Gates: tool_syntax≥100%, anti_loop≥90%, sec_catch≥85%, fp≤10%, speed_8k≥30 tok/s

```bash
PYTHONPATH=. .venv/bin/python eval/runners/run_baseline.py --model "Hermes-2-Pro-Llama-3-8B"
```

## Model Registry

```bash
# Register after training
curl -X POST localhost:8300/models/register -H 'Content-Type: application/json' -d '{"version": "v0.1.0", ...}'

# Promote (gate-checked)
curl -X POST localhost:8300/models/v0.1.0/promote -H 'Content-Type: application/json' -d '{"target_status": "production"}'
```

## แผนโครงการ

ดู `Plan/00-overview.md` → `Plan/08-execution-plan.md` (สรุปแผน + ตารางสัปดาห์ + go/no-go)
