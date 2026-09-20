# ระยะที่ 3: Training Launcher

**ประมาณการ:** ประมาณ 7 วันทำงาน (รอบนำร่องโหมด A ใช้ราว 4 วัน) **ต้องทำก่อน:** ระยะ 1 (ชุดข้อมูล) และระยะ 4 (baseline ของ base model) **ส่งผลให้:** ระยะ 5

## 1. เป้าหมาย

สั่งเทรน QLoRA จาก UI ด้วย config ที่ผ่านการตรวจก่อนเริ่ม ดูความคืบหน้าได้ และได้ adapter พร้อมข้อมูลย้อนรอยครบ โดยไม่ต้องแก้สคริปต์ด้วยมือ

## 2. ขอบเขต

**ทำ**
- ฟอร์ม config และการสร้างสคริปต์เทรน
- ตรวจก่อนเริ่มเทรน (pre-flight) และ dry run
- Runner สำหรับสั่งรันบนเครื่อง GPU
- แสดง log และกราฟ loss พร้อมตรวจ overfit
- บันทึกผลลัพธ์พร้อม manifest

**ไม่ทำ**
- merge และแปลง GGUF (ระยะ 5)
- ตัดสินว่าโมเดลใช้ได้หรือไม่ (ระยะ 4)

## 3. ข้อกำหนดเบื้องต้น

- ชุดข้อมูลเวอร์ชันที่ล็อกแล้วจากระยะ 1
- คะแนน baseline ของ base model จากระยะ 4
- เลือกรูปแบบ runner (ดูหัวข้อ 4)
- ตัดสินใจ base model และตรวจว่าตระกูลโมเดลรองรับ tool calling ผ่าน chat template

## 4. ตัวเลือกการรัน (Runner)

| ตัวเลือก | วิธี | ข้อดี | ข้อเสีย | แนะนำ |
|---|---|---|---|---|
| A. ส่งออกโน้ตบุ๊ก/สคริปต์ | UI สร้างไฟล์ ผู้ใช้อัปโหลดไปรันบน Colab หรือเซิร์ฟเวอร์เอง แล้วอัปโหลด `trainer_state.json` กลับ | ทำเร็วที่สุด ไม่ต้องมี runner | ไม่มี log สด | เริ่มที่นี่ (MVP) |
| B. Runner บนเครื่อง GPU | เอเจนต์เล็กๆ (FastAPI) บนเครื่องที่มี GPU รับงาน รันเทรน และส่ง log ผ่าน SSE หรือ WebSocket | ครบวงจร มี log สด | ต้องตั้งเครื่องและความปลอดภัยของ endpoint | ทำเป็นขั้นถัดไป |
| C. เทรนบนเครื่อง Intel XPU | Docker หรือ torch.xpu ตามแผนเดิม | ไม่ต้องเช่า GPU | เสี่ยงเรื่องไดรเวอร์และความเข้ากันของไลบรารี | ทดลองแยก ไม่ควรเป็นเส้นทางหลัก |

หมายเหตุ: ความพอดีของ VRAM ที่ 7B + 4-bit + ความยาว 8k ขึ้นกับการ์ด ให้ยืนยันด้วย dry run ในหัวข้อ 6 เสมอ ไม่ควรสรุปจากสเปกอย่างเดียว

## 5. Config เทรน (YAML)

```yaml
run_name: r001
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
dataset_version: v0001
seed: 3407
quantization: {load_in_4bit: true, quant_type: nf4, double_quant: true}
lora:
  r: 16
  alpha: 32
  dropout: 0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
train:
  epochs: 2
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.05
  per_device_batch_size: 2
  grad_accum: 4
  max_seq_length: 8192
  eval_steps: 50
  save_steps: 50
  early_stopping_patience: 3
loss_masking: assistant_only
```

ค่าเริ่มต้นต่างจากแผนเดิมเล็กน้อย: 2 epochs (แทน 3) และ 8,192 tokens (แทน 4,096) เพราะข้อมูลหลักพันเสี่ยง overfit และตัวอย่างเขียนไฟล์เต็มยาวเกิน 4k ได้ง่าย ถ้า eval loss ยังลดต่อ ค่อยเพิ่ม epoch

## 6. Pre-flight checks (บังคับก่อนกดเริ่ม)

| รหัส | ตรวจอะไร | เกณฑ์ผ่าน |
|---|---|---|
| PF1 | hash ของไฟล์ชุดข้อมูลตรงกับ manifest | ตรงทุกไฟล์ |
| PF2 | เรนเดอร์ 5 ตัวอย่างด้วย chat template จริงและแสดงให้ดู | รูปแบบตรงกับที่ตั้งใจ รวม tool call และ tool response |
| PF3 | **ทดสอบ loss mask:** ถอดรหัส label ที่ไม่ถูก mask ของตัวอย่างสุ่ม | มีเฉพาะข้อความ assistant ไม่มี system, user หรือ tool response |
| PF4 | ตรวจความยาว token | ไม่มีตัวอย่างถูกตัด (truncate) |
| PF5 | Dry run 10 steps บนเครื่องจริง | ไม่ OOM และบันทึกการใช้ VRAM สูงสุด |
| PF6 | บันทึกเวอร์ชันไลบรารี (pip freeze) | เขียนลง manifest |

ถ้าไม่ผ่าน PF3 ห้ามเริ่มเทรน เพราะโมเดลจะเรียนจากข้อความที่ไม่ควรเรียน (เช่น ผลลัพธ์ของ tool)

## 7. การรันและการติดตาม

- ระหว่างเทรน runner เขียน `metrics.jsonl` (step, train_loss, eval_loss, learning_rate, grad_norm, เวลา)
- Backend อ่านแล้วส่งให้ UI ผ่าน SSE หรือ WebSocket (โหมด A ใช้การอัปโหลดไฟล์แทน)
- UI แสดง: กราฟ train/eval loss, ความเร็ว (steps/วินาที), เวลาที่เหลือโดยประมาณ, การใช้ VRAM
- ปุ่มหยุด, ปุ่มต่อจาก checkpoint เมื่อหลุดระหว่างทาง (สำคัญกับ Colab ที่มีข้อจำกัดเวลา session)

**กฎตรวจ overfit อัตโนมัติ:** ถ้า eval loss สูงขึ้นติดต่อกัน 3 ครั้งที่วัด ขณะที่ train loss ลดลง ให้แจ้งเตือนและใช้ checkpoint ที่ eval loss ต่ำสุด

**ข้อควรระวัง:** eval loss ต่ำไม่ได้แปลว่าทำงานจริงได้ดี ต้องผ่าน Eval Harness ของระยะ 4 ก่อนทุกครั้ง

## 8. การทดลองหลายค่า (ไม่บังคับ)

ทดลองกริดเล็กๆ เช่น learning rate {1e-4, 2e-4} × epochs {2, 3} แล้วเลือกจากผล gate ของระยะ 4 ไม่ใช่จาก loss อย่างเดียว บันทึกทุกรันไว้เปรียบเทียบ

## 9. ผลลัพธ์ของแต่ละรัน

```
runs/r001/
  config.yaml
  dataset_manifest.json    # เวอร์ชันและ hash ของข้อมูล
  env.txt                  # pip freeze, GPU, driver
  metrics.jsonl
  adapter/                 # adapter_model.safetensors, adapter_config.json
  tokenizer/
  best_checkpoint.txt
  report.md                # สรุปอัตโนมัติ
```

## 10. งานย่อย

| รหัส | งาน | รายละเอียด | เวลา |
|---|---|---|---|
| T3.1 | ตัวสร้างสคริปต์จาก config | ปรับจากสคริปต์ต้นแบบ ใช้ YAML เป็นแหล่งความจริง | 0.5 วัน |
| T3.2 | Pre-flight PF1-PF6 | เขียนเป็นสคริปต์และแสดงผลใน UI | 1.5 วัน |
| T3.3 | โหมด A | ส่งออกสคริปต์/โน้ตบุ๊ก และนำเข้า trainer_state.json | 0.5 วัน |
| T3.4 | Runner (โหมด B) | รับงาน รันเทรน สตรีม metrics หยุด/ต่อได้ | 2 วัน |
| T3.5 | หน้าติดตามผล | กราฟ loss, สถานะ, คำเตือน overfit | 1 วัน |
| T3.6 | Manifest และรายงาน | รวบรวมผลลัพธ์ตามหัวข้อ 9 | 0.5 วัน |
| T3.7 | ทดลองเทรนเต็มรอบแรก | ใช้ชุดข้อมูลจริง ตรวจปัญหาที่เจอ | 1 วัน |

## 11. การทดสอบ

- ทดสอบ pre-flight ด้วยข้อมูลที่ตั้งใจทำให้ผิด (mask ผิด, template ผิด, ตัวอย่างยาวเกิน) ต้องถูกจับได้
- ทดสอบรันสั้น (10-20 steps) ตลอดวงจร: สั่งเริ่ม → ดู log → หยุด → ต่อ → จบ
- ทดสอบว่ารันซ้ำด้วย seed เดิมได้ loss ช่วงต้นใกล้เคียงเดิม
- ทดสอบการอ่าน metrics ที่ไฟล์ขาดหายหรือไม่สมบูรณ์

## 12. เกณฑ์ตรวจรับ

- [ ] กดเทรนแล้วได้ adapter กลับมาโดยไม่แก้สคริปต์ด้วยมือ
- [ ] Pre-flight ทั้ง 6 ข้อรันอัตโนมัติ และจับกรณีผิดในการทดสอบได้
- [ ] มีกราฟ train/eval loss และคำเตือน overfit ทำงาน
- [ ] ทุกรันมี manifest ที่ผูกกับเวอร์ชันข้อมูล config และไลบรารี
- [ ] หยุดและต่อจาก checkpoint ได้

## 13. ความเสี่ยง

| ความเสี่ยง | วิธีรับมือ |
|---|---|
| Loss mask ผิด | PF3 บังคับ ไม่ผ่านห้ามเทรน |
| Template ไม่ตรงกับตอน deploy | PF2 และการทดสอบเทียบในระยะ 5 |
| OOM ที่ความยาว 8k | PF5 dry run ลด batch เพิ่ม grad accumulation หรือใช้ GPU ใหญ่ขึ้น |
| Colab หลุดกลางทาง | บันทึก checkpoint ทุก 50 steps และรองรับการต่อ |
| Overfit | validation, early stopping, 2 epochs เป็นค่าเริ่มต้น |
