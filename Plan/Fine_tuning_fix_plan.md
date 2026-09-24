# Fine-tuning Workbench — Fix & Improvement Plan

## 1. เป้าหมายของการแก้ไข

เป้าหมายหลักคือทำให้ Training Run Lifecycle ของระบบทำงานสอดคล้องกันระหว่าง

- Frontend
- Backend API
- Database
- Training Process
- GPU/Resource
- Metrics
- Checkpoints
- Artifacts

Flow ที่ต้องการหลังแก้:

```text
Create Config
    ↓
Preflight
    ↓
Pass
    ↓
Create Training Run
    ↓
Queued
    ↓
Running
    ↓
Monitor Metrics
    ↓
Completed
```

กรณีหยุดหรือเกิดปัญหา:

```text
Running
 ├── Stop → Stopped → Resume → Running
 └── Error → Failed
```

---

# 2. Priority

| Priority | ความหมาย |
|---|---|
| P0 | Critical — ต้องแก้ก่อน เพราะกระทบการทำงานหลักของระบบ |
| P1 | High — มีผลต่อความถูกต้อง ความปลอดภัย และความน่าเชื่อถือ |
| P2 | Medium — ปรับปรุงคุณภาพและความพร้อมใช้งาน |

---

# 3. Phase 1 — แยก Preflight ออกจาก Training

**Priority: P0**

## ปัญหา

Flow ปัจจุบันมีลักษณะ:

```text
กด Preflight
    ↓
POST /training/runs
    ↓
เริ่ม Training
    ↓
จึงค่อยเรียก Preflight
```

ทำให้ Preflight ไม่ได้ทำหน้าที่เป็น validation gate จริง

## เป้าหมายใหม่

```text
POST /training/preflight
        ↓
PF1 - PF6
        ↓
ผ่านหรือไม่
   ├── ไม่ผ่าน → Reject
   └── ผ่าน → Submit
                    ↓
             POST /training/runs
                    ↓
                 Training
```

## งานที่ต้องแก้

### Backend

- แยก endpoint สำหรับ preflight
- Preflight ต้องรับ config โดยไม่สร้าง Training Run
- `POST /training/runs` ต้องตรวจว่ามีผล Preflight ที่ผ่านแล้ว
- ห้ามเริ่ม training โดย bypass preflight

### Frontend

- ปุ่ม Preflight ต้องไม่สร้าง Run
- ปุ่ม Submit ต้อง disabled จนกว่า Preflight จะผ่าน
- แสดงผล PF1-PF6 จาก response ของ backend

## Definition of Done

- กด Preflight แล้วไม่มี GPU training
- Preflight FAIL → Submit ไม่ได้
- Preflight PASS → Submit ได้
- เรียก API ตรงโดยไม่มี Preflight → backend reject

---

# 4. Phase 2 — สร้าง Training State Machine

**Priority: P0**

## State ที่แนะนำ

```text
CREATED
   ↓
PREFLIGHTING
   ↓
READY
   ↓
QUEUED
   ↓
RUNNING
   ↓
COMPLETED
```

กรณีหยุด:

```text
RUNNING
   ↓
STOPPING
   ↓
STOPPED
   ↓
QUEUED
   ↓
RUNNING
```

กรณี error:

```text
RUNNING
   ↓
FAILED
```

## ตัวอย่าง Transition

```python
ALLOWED_TRANSITIONS = {
    "created": ["preflighting"],
    "preflighting": ["ready", "failed"],
    "ready": ["queued"],
    "queued": ["running", "failed"],
    "running": ["stopping", "completed", "failed"],
    "stopping": ["stopped", "failed"],
    "stopped": ["queued"],
}
```

## งานที่ต้องแก้

- สร้าง transition function กลาง
- ห้าม router เปลี่ยน status แบบอิสระ
- ทุก status transition ต้อง validate
- UI และ DB ต้องใช้ status ชุดเดียวกัน

## Definition of Done

สถานะต่อไปนี้ต้องตรงกัน:

```text
Database
Frontend
Training Process
Log
```

---

# 5. Phase 3 — แก้ Stop Training

**Priority: P0**

## ปัญหา

Backend ใช้ `asyncio.subprocess.Process` แต่มีการเรียก API ในรูปแบบ `subprocess.Popen` เช่น `poll()` และ `wait(timeout=...)`

## เป้าหมาย

ใช้ async process model ให้สอดคล้องกัน:

```python
if proc.returncode is None:
    proc.terminate()

await asyncio.wait_for(
    proc.wait(),
    timeout=30
)
```

ถ้ายังไม่หยุด:

```text
SIGTERM
   ↓
รอ
   ↓
ยังไม่หยุด
   ↓
SIGKILL
```

## ต้องรองรับ

- Stop Training
- Cancel queued run
- Process crash
- Server restart recovery

## Definition of Done

เมื่อผู้ใช้กด Stop:

```text
GPU Process หยุดจริง
       ↓
DB = stopped
       ↓
Frontend = stopped
```

ไม่มี zombie process

---

# 6. Phase 4 — ทำ Resume ให้ Resume จริง

**Priority: P0**

## ปัญหา

Resume ปัจจุบันเปลี่ยน status อย่างเดียว เช่น

```text
stopped → pending
```

แต่ไม่ได้สร้าง Training Process ใหม่

## Flow ใหม่

```text
STOPPED
   ↓
ค้นหา checkpoint ล่าสุด
   ↓
สร้าง Training Process
   ↓
resume_from_checkpoint
   ↓
QUEUED
   ↓
RUNNING
```

## ต้องบันทึก

- `checkpoint_path`
- `global_step`
- `epoch`
- metrics ล่าสุด
- training config
- model/tokenizer information

## Definition of Done

ทดสอบ:

```text
Train
→ Stop ที่ step 500
→ Resume
→ เริ่มจาก checkpoint
→ ไม่กลับไป step 0
```

---

# 7. Phase 5 — จัด Artifact Directory ใหม่

**Priority: P0**

## ปัญหา

Output path ของ Backend และ Training Script มีโอกาสไม่ตรงกัน และเสี่ยงเกิด directory ซ้อนกัน เช่น:

```text
runs/r001/
    runs/r001/
```

## โครงสร้างใหม่

```text
runs/
└── r001/
    ├── config.json
    ├── environment.txt
    ├── model_info.json
    ├── dataset_manifest.json
    ├── train.py
    ├── training.log
    ├── metrics.jsonl
    ├── checkpoints/
    ├── adapter/
    └── report.json
```

## หลักการ

Run หนึ่งต้องมี root directory เดียว

ทุก artifact ของ Run ต้องอยู่ภายใน directory เดียวกัน

## DB ต้องเก็บ

```text
adapter_path
checkpoint_path
metrics_path
report_path
```

และทุก path ต้องตรวจว่ามีไฟล์จริง

## Definition of Done

หลัง Training สำเร็จ:

- artifact ครบ
- path ตรง
- ไม่มี nested run directory
- DB สามารถเปิด artifact ได้จริง

---

# 8. Phase 6 — แก้ Metrics Contract

**Priority: P0**

## ปัญหา

Backend และ Frontend ใช้ชื่อ field ไม่ตรงกัน เช่น:

```text
Backend → results
Frontend → checks
```

## แนะนำ API Contract

```json
{
  "run_id": "r001",
  "status": "running",
  "metrics": {
    "step": 100,
    "loss": 1.23,
    "learning_rate": 0.0002,
    "epoch": 0.5
  }
}
```

## Endpoint ที่แนะนำ

```text
GET /training/runs/{id}
GET /training/runs/{id}/metrics
GET /training/runs/{id}/logs
GET /training/runs/{id}/artifacts
```

## Frontend ควรแสดง

- Step
- Loss
- Learning Rate
- Epoch
- GPU Memory
- Status

## Definition of Done

UI แสดง metrics จาก API ที่กำหนดชัดเจน โดยไม่เดา path ของไฟล์เอง

---

# 9. Phase 7 — ทำ Preflight ให้ตรวจจริง

**Priority: P1**

## PF1 — Configuration

ตรวจ:

- Base Model
- Dataset
- Batch Size
- Sequence Length
- Learning Rate
- LoRA Config
- Output Directory

---

## PF2 — Dataset

ตรวจ:

- Dataset มีจริง
- JSON/JSONL format ถูก
- `messages` ถูกต้อง
- role ถูกต้อง
- content ไม่ว่าง
- train/validation split ถูกต้อง

---

## PF3 — Loss Masking

### ปัญหา

การตรวจเดิมตรวจว่ามี assistant message แต่ไม่ได้ตรวจ token labels จริง

### ต้องเปลี่ยนเป็น

สร้าง actual training batch แล้วตรวจ:

```text
system/user tokens → labels = -100
assistant tokens    → labels != -100
```

ต้องตรวจอย่างน้อย:

- input_ids
- labels
- attention_mask

---

## PF4 — GPU / Memory

ตรวจ:

- CUDA availability
- GPU name
- VRAM
- dtype
- device
- bitsandbytes
- available memory

---

## PF5 — Real Dry Run

Dry run ต้องใกล้เคียง training จริง:

```text
Load Model
    ↓
Prepare LoRA
    ↓
Tokenize
    ↓
Forward
    ↓
Backward
    ↓
Optimizer Step
    ↓
3-10 Steps
```

ใช้ config จริงของผู้ใช้

ไม่ควรตรวจเพียง forward ครั้งเดียว

---

## PF6 — Environment Capture

เก็บ:

```text
Python version
PyTorch version
Transformers version
PEFT version
TRL version
CUDA version
GPU info
pip freeze
Git commit
```

ไฟล์ที่ควรสร้าง:

```text
environment.txt
```

---

# 10. Phase 8 — Model Compatibility

**Priority: P1**

## ปัญหา

Loss masking / chat format ถูก hard-code เช่น:

```text
<|im_start|>assistant
```

แต่แต่ละ model อาจใช้ chat template ต่างกัน

## แนวทางใหม่

ให้ tokenizer/model เป็น source of truth:

```text
Model
  ↓
Tokenizer
  ↓
Chat Template
  ↓
Token Mask
```

## ต้องทดสอบอย่างน้อย

- Qwen
- Hermes
- Custom Hugging Face Model

## Definition of Done

เปลี่ยน Base Model แล้ว:

- Preflight ตรวจ chat template ใหม่
- Loss masking ถูกต้อง
- Training ใช้ template ของ model
- ไม่มี magic string ที่ผูกกับ model เดียว

---

# 11. Phase 9 — Security และ Input Validation

**Priority: P1**

## `run_name`

ควรจำกัด:

```regex
^[a-zA-Z0-9_-]{1,32}$
```

ห้ามส่ง path จาก user โดยตรง

แนะนำให้แยก:

```text
internal_run_id = UUID
display_name = user-defined name
```

---

## Base Model

แนะนำ Model Registry:

```python
ALLOWED_MODELS = {
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    ...
}
```

หรือ trusted model list

---

## `trust_remote_code=True`

ต้องกำหนด policy ให้ชัดเจน

ทางเลือก:

```text
Trusted Model
    ↓
allow remote code

Untrusted Model
    ↓
reject
```

ไม่ควรเปิดให้ model repository อิสระ execute code ใน production โดยไม่มีการควบคุม

---

# 12. Phase 10 — แก้ Run ID และ Concurrency

**Priority: P1**

## ปัญหา

การหา run ID แบบ:

```text
MAX(run_id) + 1
```

มี race condition

ตัวอย่าง:

```text
Request A → r005
Request B → r005
```

## แนะนำ

ใช้:

```text
UUID / DB-generated ID
```

แล้วมี display ID แยก:

```text
internal_id = 7f04c...
display_id = R-005
```

## Definition of Done

สร้าง Run พร้อมกันหลาย request ได้โดยไม่ชนกัน

---

# 13. Phase 11 — Cleanup Architecture

**Priority: P2**

หลังจาก lifecycle ทำงานถูกต้องแล้ว ให้ cleanup code

ตรวจและลบ:

- Dead function
- Unused import
- Duplicate generator
- Legacy state
- Old training flow
- Debug code
- Duplicate output logic

โดยเฉพาะส่วน:

```text
create_run
start_run
_generate_training_script
preflight
resume
```

ต้องมี Single Source of Truth

---

# 14. Phase 12 — ทำ Test Suite

**Priority: P1**

## Unit Tests

ทดสอบ:

- Config validation
- Dataset validation
- Loss masking
- State transition
- Run ID
- Path validation

---

## Integration Test — Success Flow

```text
Preflight PASS
    ↓
Submit
    ↓
Queue
    ↓
Running
    ↓
Metrics
    ↓
Completed
```

---

## Integration Test — Failed Preflight

```text
Preflight FAIL
    ↓
Submit
    ↓
Rejected
```

---

## Integration Test — Stop / Resume

```text
Start
  ↓
Running
  ↓
Stop
  ↓
Stopped
  ↓
Resume
  ↓
Running
  ↓
Completed
```

---

## Integration Test — Crash

```text
Running
   ↓
Training Process Crash
   ↓
Failed
```

---

# 15. ลำดับการลงมือแก้จริง

ไม่ควรแก้ทั้งหมดพร้อมกัน

```text
STEP 01
แยก Preflight Flow
       ↓
STEP 02
สร้าง State Machine
       ↓
STEP 03
แก้ Start / Stop
       ↓
STEP 04
แก้ Resume
       ↓
STEP 05
จัด Artifact Directory
       ↓
STEP 06
แก้ Metrics API
       ↓
STEP 07
ทำ PF3 / PF5 ให้ตรวจจริง
       ↓
STEP 08
แก้ Model Compatibility
       ↓
STEP 09
Security / Validation
       ↓
STEP 10
Run ID / DB Concurrency
       ↓
STEP 11
Cleanup
       ↓
STEP 12
Unit + Integration Tests
```

---

# 16. Definition of Done — โปรเจกต์ทั้งระบบ

เมื่อแก้เสร็จ ผู้ใช้ต้องสามารถทำ flow นี้ได้โดยไม่ต้อง workaround:

```text
เลือก Model
     ↓
เลือก Dataset
     ↓
ตั้ง Training Config
     ↓
กด Preflight
     ↓
PF1 ✅
PF2 ✅
PF3 ✅
PF4 ✅
PF5 ✅
PF6 ✅
     ↓
Submit
     ↓
Queued
     ↓
Running
     ↓
ดู Loss แบบ Real-time
     ↓
Stop
     ↓
Resume จาก Checkpoint
     ↓
Running ต่อ
     ↓
Completed
     ↓
ดู Adapter / Metrics / Report
```

---

# 17. Final Architecture เป้าหมาย

```text
                 ┌──────────────────┐
                 │ Training Config  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   Preflight      │
                 │ PF1 ... PF6      │
                 └────────┬─────────┘
                          │
                    can_proceed?
                     /          \
                   no            yes
                   │              │
                   ▼              ▼
                Reject       Create Run
                                  │
                                  ▼
                              Queue Run
                                  │
                                  ▼
                              Worker
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
             train.py         metrics.jsonl     checkpoint
                │
                ▼
          DB State Update
                │
                ▼
        Frontend Monitoring
```

---

# 18. สรุป Priority แบบย่อ

| Phase | งาน | Priority |
|---|---|---|
| 1 | Preflight Gate | P0 |
| 2 | State Machine | P0 |
| 3 | Stop | P0 |
| 4 | Resume | P0 |
| 5 | Artifact Structure | P0 |
| 6 | Metrics Contract | P0 |
| 7 | Real Preflight | P1 |
| 8 | Model Compatibility | P1 |
| 9 | Security / Validation | P1 |
| 10 | Run ID / Concurrency | P1 |
| 11 | Cleanup | P2 |
| 12 | Test Suite | P1 |

---

# 19. Suggested GitHub Issues

แนะนำแยกเป็น Issue ดังนี้:

- [ ] `P0: Separate Preflight from Training Start`
- [ ] `P0: Implement Training Run State Machine`
- [ ] `P0: Fix Async Training Process Stop`
- [ ] `P0: Implement Real Checkpoint Resume`
- [ ] `P0: Normalize Run Artifact Structure`
- [ ] `P0: Fix Metrics API Contract`
- [ ] `P1: Implement Real Loss Mask Validation`
- [ ] `P1: Implement Real 3-10 Step Dry Run`
- [ ] `P1: Add Environment Capture`
- [ ] `P1: Make Chat Template Model-Aware`
- [ ] `P1: Add Run Name / Model Input Validation`
- [ ] `P1: Replace Run ID MAX+1 Logic`
- [ ] `P1: Add Training Lifecycle Integration Tests`
- [ ] `P2: Remove Legacy Training Code`
- [ ] `P2: Make `run_server.sh` Portable`
