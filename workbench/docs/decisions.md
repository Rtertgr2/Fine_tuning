# บันทึกการตัดสินใจ (Decision Log)

## D1. Base model เริ่มต้น: Hermes 2 Pro Llama-3 8B — ตัดสินใจ 2026-09-20

**บริบท:** ต้องล็อก tokenizer + chat template ตัวจริงก่อนสร้าง validator และข้อมูล

**ตัวเลือก:**
- Hermes 2 Pro Llama-3 8B (`NousResearch/Hermes-2-Pro-Llama-3-8B`) — ถูกฝึก function calling มาแล้ว, llama.cpp รองรับ native, template ชัดเจน
- Qwen2.5-7B-Instruct — รองรับ tool calling, นิยมใช้ แต่ template คนละแบบ
- Llama-3.1-8B-Instruct — ต้องใช้ tool template แยก, คุณภาพ tool use ต่ำกว่า

**ตัดสินใจ:** เริ่มที่ Hermes 2 Pro Llama-3 8B และรองรับ Qwen2.5-7B เป็น adapter ที่สอง
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
