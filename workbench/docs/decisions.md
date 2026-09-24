# บันทึกการตัดสินใจ (Decision Log)

## D1. Base model เริ่มต้น: Hermes 2 Pro Llama-3 8B — ตัดสินใจ 2026-09-20

**บริบท:** ต้องล็อก tokenizer + chat template ตัวจริงก่อนสร้าง validator และข้อมูล

**ตัวเลือก:**
- Hermes 2 Pro Llama-3 8B (`NousResearch/Hermes-2-Pro-Llama-3-8B`) — ถูกฝึก function calling มาแล้ว, llama.cpp รองรับ native, template ชัดเจน
- Qwen2.5-7B-Instruct — รองรับ tool calling, นิยมใช้ แต่ template คนละแบบ
- Llama-3.1-8B-Instruct — ต้องใช้ tool template แยก, คุณภาพ tool use ต่ำกว่า

**ตัดสินใจ:** เริ่มที่ Hermes 2 Pro Llama-3 8B และรองรับ Qwen2.5-7B เป็น adapter ที่สอง

**เหตุผลที่เลือก 8B แทน 7B:** แผนเดิมระบุ 7B แต่ D1 เลือก 8B เพราะ (1) Hermes-2-Pro-Llama-3-8B มี training data ที่เน้น function calling มากกว่า ซึ่งเป็นเป้าหมายหลักของโปรเจค; (2) สถาปัตยกรรม ModelAdapter แยกทุกอย่างที่ขึ้นกับโมเดลออกจาก validator/API/pipeline ทำให้เปลี่ยนขนาดโมเดลได้โดยไม่แก้โค้ดส่วนอื่น; (3) VRAM 12 GB พอเพียงสำหรับ QLoRA fine-tune 8B ร่วมกับ gradient checkpointing. หากต้องการ 7B จริง ๆ ให้ใช้ `Hermes-2-Pro-Mistral-7B` หรือ `Qwen2.5-7B-Instruct` เป็น adapter ตัวใหม่ ไม่ต้องแก้ pipeline.
**เหตุผล:** ผู้ใช้ต้องการระบบที่เปลี่ยนโมเดลได้ ไม่ผูกกับโมเดลใดโมเดลหนึ่ง — สถาปัตยกรรม
`ModelAdapter` แยกทุกอย่างที่ขึ้นกับโมเดล (template, รูปแบบ tool call, tokenizer) ออกจาก
validator/API/pipeline ทั้งหมด การเพิ่มโมเดลใหม่ = เพิ่มไฟล์ adapter 1 ไฟล์ + template

**หมายเหตุ:** ถ้าอยากได้ขนาด 7B จริงๆ มี `Hermes-2-Pro-Mistral-7B` — เพิ่มเป็น adapter ได้
ด้วยขั้นตอนเดียวกัน (template ChatML + `<tool_call>` tags ชุดเดียวกัน)

## D2. รูปแบบ tool call: ฝังแท็ก `<tool_call>` ใน content (T1.0) — ตัดสินใจ 2026-09-20

**ข้อสรุป:** เก็บ tool call เป็นแท็กฝังใน `content` ของ assistant message:

```
<tool_call>
{"name": "read_file", "arguments": {"path": "a.py"}}
</tool_call>
```

ผลลัพธ์ tool เก็บเป็น content ดิบใน message role `tool` (template ครอบ `<tool_response>` เอง)

**การทดสอบที่ยืนยันแล้ว (tests/test_render_parity.py):**
1. `render_tool_call()` ของเรา render ผ่าน template แล้ว **ตรงทุก byte** กับ path ที่ใช้
   ฟิลด์ `tool_calls` แบบ structured (ทดสอบทั้ง single call, multi call, nested args,
   ข้อความไทย/unicode)
2. Render ทั้งหมด **ตรงทุก byte** กับ `tokenizer.apply_chat_template()` ตัวจริงของ
   `NousResearch/Hermes-2-Pro-Llama-3-8B` (ทดสอบเมื่อ tokenizer โหลดแล้ว)

**ทำไมวิธีนี้:** เก็บแบบฝังแท็ก = รูปแบบที่โมเดลเปล่งออกจริงตอน inference (llama-server
คืนข้อความดิบ) ทำให้ train/serve ใช้ข้อความชุดเดียวกันโดยไม่มีการแปลงกลับไปกลับมา

**ข้อจำกัดที่ทราบ:** ถ้า assistant มีทั้งข้อความบรรยายและ tool call ปนกัน path structured
จะตัดข้อความบรรยายทิ้ง (template ไม่ render `content` เมื่อมี `tool_calls`) — เราจึงใช้
path ฝังแท็กเป็นทางเดียวทั้งระบบ การทดสอบ D2 ครอบเฉพาะกรณี tag-only ซึ่งเป็นรูปแบบหลัก

**Template ที่ล็อก (vendored ใน `backend/adapters/templates/`):**
- `hermes2pro_default.jinja` — ChatML เปล่า ใช้เมื่อไม่มี tools (plan, sec)
- `hermes2pro_tool_use.jinja` — มี tools preamble ใช้เมื่อมี tools (tool, loop)
- ที่มา: `tokenizer_config.json` ของ HF repo ตรงๆ (ตรวจแล้วว่า llama.cpp copy เหมือนกันเป๊ะ)

## D3. ค่า status ของหมวด plan — ตัดสินใจ 2026-09-20

`APPROVED` | `NEEDS_REVISION`

**เหตุผล:** agent สถานะ Review ตัดสินว่าแผนผ่านหรือต้องแก้ — สองค่าพอสำหรับ gate
(เก็บใน `config.PLAN_STATUS_VALUES`, กฎ P3)

## D4. โครงสร้าง output ของหมวด sec — ตัดสินใจ 2026-09-20

```json
{"status": "PASS" | "REJECT",
 "issues": [{"file": "...", "line": 123, "type": "...", "severity": "critical|high|medium|low", "fix": "..."}]}
```

**เหตุผล:** ตรงตามที่แผนแนะนำ (status + issues[]) และตรวจได้จริงทุกฟิลด์ (กฎ S3)
`line` คือเลขบรรทัดในไฟล์ใหม่ของ diff ที่ให้มา (กฎ S4 ตรวจกับ hunk จริง)

**S7 type families ที่ต้องมี:** `sql_injection`, `resource_leak`, `insecure_deserialization`,
`owasp` (จับคู่ type ที่ขึ้นต้นด้วย `owasp_` หรือ `a0`) — ปรับได้ใน config เมื่อทำ gold set

## D5. max sequence length เริ่มต้น: 8192 tokens — ตัดสินใจ 2026-09-20

ตัวอย่าง agent ต้องแบก system + tools preamble + tool results หลายรอบ; 8192 คือค่าที่
ทดสอบ C4 กับข้อมูลจริงได้ ปรับผ่าน `WORKBENCH_MAX_SEQ_LEN` (แผน 00 เสี่ยงเรื่อง context
32k บน VRAM 12GB — ค่านี้จะถูกวัดจริงในระยะ 3/4)

## D6. UI v1: หน้า HTML เดียว serve โดย FastAPI — ตัดสินใจ 2026-09-20

**ข้อสรุป:** `frontend/index.html` (+ styles/app js) ธรรมดา ไม่มี build step FastAPI
mount ที่ `/ui` (StaticFiles) เรียก API แบบ same-origin — ตามค่าตั้งต้นของแผน 07 §15.3
(UI v1 จนถึง G4 ค่อยย้าย Vite+React)

**เหตุผล:**
- ต้นแบบ `finetune-workbench.html` ที่แผน 07 อ้างถึงไม่พบในเครื่อง จึงสร้าง UI v1 ใหม่
  โดยเก็บหลักการเดิม: ตัวตรวจอยู่ backend ที่เดียว (UI เรียก `/validate` debounce 300 ms
  และ `/render` สำหรับพรีวิว+นับ token จริง) ไม่มีตรรกะตรวจซ้ำฝั่ง JS
- serve ผ่านตัว server เดียวกันจบเรื่อง CORS ตอนใช้งานปกติ เปิด CORS เฉพาะ localhost
  ไว้สำหรับเปิดไฟล์ตรง ๆ (file://) ระหว่างพัฒนา
- ดีไซน์ Warm Notebook (paper/terracotta, Fraunces + IBM Plex Sans Thai) + dark mode
  ผ่าน prefers-color-scheme; สถานะแสดงข้อความเสมอ (ไม่ใช้สีอย่างเดียว), แผงผลตรวจใช้
  aria-live ตามแผน 07 §9

## D7. Gold set: 108 ตัวอย่าง 4 หมวด — ตัดสินใจ 2026-09-20

**ข้อสรุป:** เขียนตัวอย่างต้นแบบมือ 108 ตัวอย่าง (tool:30, loop:28, plan:30, sec:20)
ผ่าน validator ทุกกฎระดับ err, import เข้า DB, สร้าง dataset v0001 สำเร็จ

**สถานะ validator:**
- tool: 0 err / 30 warn (T5 call count)
- loop: 0 err / 12 warn (L1 has-failure, L4 giveup ratio)
- plan: 0 err / 0 warn
- sec: 0 err / 2 warn (S6 PASS share 30%, S7 type coverage)

**Dataset v0001:** seed=42, 97 train / 11 val, ไม่มี group leakage, seed reproducible

**สัดส่วน sec:** PASS 6/20 = 30% (ต่ำกว่า S6 window 35-65% แต่เป็น warn ไม่ block)
→ จะปรับเพิ่ม PASS examples เมื่อทำ active learning (ระยะ 6)

**หมายเหตุ:** ตัวอย่าง sec PASS/REJECT สร้างเป็นคู่ (diff เดียวกัน, ผลลัพธ์ต่าง)
เพื่อไม่ให้โมเดลเรียนแค่ลักษณะผิวเผิน (ตามแนวทาง §8 ของแผน 01)

## D8. Eval Harness: ชุดทดสอบ frozen + runners — ตัดสินใจ 2026-09-20

**ข้อสรุป:** สร้าง eval harness ครบตามระยะ 4 — config gates, test suites 5 ชุด, runners, report, tests

**Frozen suites (eval/suites/):**
- tool_syntax: 100 cases (เป้าหมาย >=100 ผ่าน)
- anti_loop: 40 cases (เป้าหมาย >=90% เปลี่ยนพฤติกรรม/หยุดรายงาน)
- security_vuln: 100 cases (10 types x 10 cases, เป้า >=85% REJECT ถูกต้อง)
- security_clean: 100 cases (เป้า FP <=10%)
- plan_json: 50 cases (เป้า JSON valid + status ถูก)

**Gates (locked):**
- tool_syntax >=100% | anti_loop >=90% | security_catch >=85%
- security_fp <=10% | speed_8k >=30 tok/s | regression within 3 pts

**สถานะ:** harness ทำงานได้, เทสต์หน่วยผ่าน 98 tests
การรัน baseline จริง (T4.10-T4.11) ต้องมี llama-server รันอยู่
(ตอนนี้ยัมไม่มี GGUF บนเครื่อง — เป็นงานของระยะ 5)

**รัน baseline เมื่อพร้อม:**
```bash
PYTHONPATH=. .venv/bin/python eval/runners/run_baseline.py --model "Hermes-2-Pro-Llama-3-8B" --run-id baseline_hermes2pro
```

## D9. Execution Plan (08) — ตัดสินใจ 2026-09-20

**ข้อสรุป:** จัดทำ `docs/execution-plan.md` สรุปตารางสัปดาห์, go/no-go gates, critical path, กฎตัดขอบเขต จาก 6 Phase ที่ทำเสร็จ

**Key gates:** G0 (W0 hardware check), G1 (W5 baseline), G2 (W7 pilot), G3 (W10 full), G4 (W12 AL loop)

**Critical path:** D1-D11 → T1.0 → T1.2 → T2.2 → T4.3-T4.6 → baseline → pilot → full train

**Scope cut order:** UI v2 → Runner B/dashboard → AL semi-manual → smaller v002 → never cut frozen eval/loss mask/template parity/gate/rollback

## D10. Corrective implementation pass — 2026-09-23

ตรวจโค้ดเทียบ Plan/00–08 และแก้จุดที่ทำให้ gate/การเทรน/การ rollback ให้ผลไม่จริง:

- **Training lifecycle:** แก้ coroutine ที่ถูกส่งเข้า `threading.Thread` โดยไม่ await; การหยุด/จบ run ไม่เขียนทับสถานะ `stopped`; `resume` เปิด process ใหม่จาก checkpoint ได้
- **Pre-flight:** ปุ่มตรวจไม่สร้าง run; train จะไม่เริ่มหาก PF1–PF5 ที่เป็น critical ไม่ผ่าน; PF2 เทียบ template กับ tokenizer จริง, PF3 สร้าง label mask จาก assistant turns จริง, PF4 ตรวจ train+val, PF5 ต้องผ่าน dry run 10 optimizer steps บน CUDA
- **Training script:** ใช้ `Trainer` กับ custom assistant-only collator เพื่อให้ mask ได้หลาย assistant turnsและไม่ hardcode ChatML marker; ปรับ eval/save steps ให้มี eval กับ dataset pilot ขนาดเล็ก, ใช้ early stopping, resume checkpoints, เขียน metrics และ manifests
- **Eval:** แยก judge ออกจาก runner; strict JSON ไม่รับ code fence; ตรวจ nested tool JSON/argument values, REJECT ที่ไม่มี issues จะตก; `--server` ถูกส่งถึง runner; false-positive rate คำนวณเป็นจำนวน FP จริง; report fail-closed เมื่อ regression/E2E/speed หรือ gate อื่นหาย
- **Model registry/deploy:** ลงทะเบียน candidate ได้เฉพาะ GGUF จริงที่ hash/magic ตรง, เก็บ manifest ข้างไฟล์, แนบรายงาน eval ภายหลังได้, จำกัด report path ไว้ใต้ `eval/reports`, promote/rollback สลับ `models/current` แบบ atomic และตรวจ artifact ก่อน
- **Active Learning:** ตรวจ validator/secret อีกครั้งก่อน approve, บังคับ second reviewer คนละคนสำหรับ sec, จำกัด AL ไม่เกิน 30%, hash ไฟล์ dataset และ holdout regression เป็นกลุ่ม session/project ทั้งกลุ่ม
- **Sandbox/pipeline:** ป้องกัน symlink path escape; generator ตรวจ exact/near duplicate, secrets และ overlap กับ frozen eval suites ทั้งในฐานข้อมูลและใน batch เดียวกัน

**ข้อจำกัดที่ยังเปิดอยู่ (ห้ามตีความว่าพร้อม production):** regression runner (HumanEval+/MBPP+), E2E runner และ UI หน้าประเมิน/deploy/case queue ยังไม่ครบ; sandbox ยังไม่ใช่ Docker/no-network container; การรัน baseline จริงต้องใช้ llama-server และ GPU; ต้องทำ template-parity และ deployment บนฮาร์ดแวร์เป้าหมายก่อน G3

## D11. Intel Arc GPU Runtime vertical slice — 2026-09-24

- Host kernel driver อยู่บน host; ใช้ image ที่ pin ไว้ `intel/pytorch:xpu-2.13.0-ubuntu24.04` สำหรับ user-space XPU stack
- Compose map เฉพาะ `/dev/dri/renderD128`, ใช้ user/group ของ host, bind API ที่ `127.0.0.1:8300`, ไม่ใช้ `privileged` หรือ Docker socket
- เพิ่ม `GET /gpu/devices`, `GET /gpu/runtimes`, PF0/PF7/PF8 และแสดง GPU ใน training UI; Intel XPU เป็น runtime เดียวที่เปิดใช้, NVIDIA/AMD/CPU training ยัง fail-closed/planned
- Generated training script เลือก XPU ก่อน CUDA, เก็บ runtime/driver/PyTorch ใน run manifest และ parse config ผ่าน JSON เพื่อรักษา boolean; profile UI เริ่มที่ NF4, batch=1, grad_accum=8, seq=2048
- ตรวจบน Intel Arc B580: PyTorch `2.13.0+xpu`, bitsandbytes `0.50.2`, Level Zero driver/runtime `1.17.39395+13`; 4-bit NF4 double-quant kernel ผ่าน และ tiny local Llama QLoRA สร้าง metrics/checkpoint/adapter ได้จริงบน XPU
- ยังไม่ถือว่าแผนเสร็จ: ยังไม่มี GPU service แยก, job allocation/per-run container/stop-resume API, NVIDIA/AMD, และยังไม่ทดสอบ full Hermes-2-Pro-8B; PF/data gates อื่นยังอาจ block run จริง

**ผลทดสอบล่าสุด (2026-09-24):** `253 passed` ด้วย `workbench/.venv/bin/python -m pytest workbench/tests/` (มี Starlette/httpx deprecation warning 1 รายการ)
