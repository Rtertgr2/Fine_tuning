# ระยะที่ 5: Merge, Quantize, Deploy

**ประมาณการ:** ประมาณ 4 วันทำงาน **ต้องทำก่อน:** ระยะ 3 (adapter) และระยะ 4 (gate) **ส่งผลให้:** ใช้งานจริง และระยะ 6

## 1. เป้าหมาย

แปลง adapter เป็นไฟล์ GGUF ที่รันบน llama-server (Vulkan) ได้ ผ่าน gate ก่อนใช้งาน และย้อนกลับเวอร์ชันได้ใน 1 คลิก

## 2. ขอบเขต

**ทำ**
- Merge adapter เข้า base 16-bit
- แปลงเป็น GGUF, สร้าง imatrix, quantize หลายระดับ
- ทดสอบความเข้ากันของ chat template
- ทะเบียนเวอร์ชันโมเดล, promote และ rollback

**ไม่ทำ**
- การเทรนและการประเมินเต็มรูปแบบ (ใช้ผลจากระยะ 3 และ 4)

## 3. ข้อกำหนดเบื้องต้น

- adapter จากระยะ 3 ที่มี manifest ครบ
- llama.cpp ที่ build แล้ว (มี `convert_hf_to_gguf.py`, `llama-imatrix`, `llama-quantize`, `llama-server`) และ **บันทึก commit ที่ใช้** เพราะรูปแบบ GGUF และตัวแปลงเปลี่ยนตามเวอร์ชัน
- พื้นที่ดิสก์พอสำหรับโมเดล 16-bit ราว 15 GB ต่อเวอร์ชัน (7B) บวกไฟล์ GGUF

## 4. ขั้นตอน (Pipeline)

```
adapter -> merge (16-bit) -> convert (f16 GGUF) -> imatrix -> quantize (Q4_K_M / Q5_K_M / Q8_0)
        -> template parity test -> smoke eval -> register version -> promote / rollback
```

**4.1 Merge**
- โหลด base ที่ความแม่นยำ 16-bit (ไม่ใช่ 4-bit) แล้วรวม adapter เข้าไป จากนั้นบันทึกโมเดลและ tokenizer
- ตัวอย่างด้วย Unsloth: `model.save_pretrained_merged("merged", tokenizer, save_method="merged_16bit")`

**4.2 แปลงเป็น GGUF**
```
python convert_hf_to_gguf.py merged --outfile model-f16.gguf --outtype f16
```

**4.3 imatrix**
- เตรียมข้อความสอบเทียบ (`calib.txt`) จากตัวอย่างชุดเทรนผสมโค้ดทั่วไป เพื่อให้ครอบคลุมทั้งรูปแบบ tool call และโค้ดปกติ
```
llama-imatrix -m model-f16.gguf -f calib.txt -o imatrix.dat
```

**4.4 Quantize**
```
llama-quantize --imatrix imatrix.dat model-f16.gguf model-Q4_K_M.gguf Q4_K_M
llama-quantize --imatrix imatrix.dat model-f16.gguf model-Q5_K_M.gguf Q5_K_M
llama-quantize --imatrix imatrix.dat model-f16.gguf model-Q8_0.gguf   Q8_0
```

| ระดับ | ขนาดโดยประมาณ (7B) | ใช้เมื่อ |
|---|---|---|
| Q4_K_M | ราว 4.7 GB | ต้องการความเร็วสูงสุดและเหลือ VRAM ให้ context ยาว |
| Q5_K_M | ราว 5.4 GB | ตัวกลางที่แนะนำให้ลองก่อน |
| Q8_0 | ราว 7.8 GB | ต้องการความแม่นยำสูงสุด แต่ต้องเผื่อ KV cache ของ context 32k บน VRAM 12 GB |

เลือกจากผล eval จริงของระยะ 4 ไม่ใช่จากขนาดอย่างเดียว

**4.5 ทดสอบความเข้ากันของ chat template (บังคับ)**
- รันบทสนทนาตัวอย่างชุดเดียวกันผ่าน `tokenizer.apply_chat_template` (ฝั่งเทรน) และผ่านการเรนเดอร์ของ llama-server (ฝั่งใช้งาน เช่น endpoint apply-template ถ้าเวอร์ชันที่ใช้มี หรือเทียบ prompt ที่บันทึกใน log ของเซิร์ฟเวอร์) แล้วเทียบข้อความที่ได้ต้องตรงกัน
- ครอบคลุมกรณีมี tools, tool_call, tool response และหลายรอบ
- ต้องรัน llama-server ด้วย `--jinja` ให้ใช้ template ที่ฝังในโมเดล

**4.6 Smoke eval**
- รันด่าน 1 (Tool syntax) แบบย่อและด่านความเร็วบนไฟล์ที่ quantize แล้วทุกระดับ เพื่อตัดตัวที่เสียหายออกก่อนรัน eval เต็ม

**4.7 คำสั่งรัน**
```
llama-server -m models/current/model-Q5_K_M.gguf --jinja -c 32768 -ngl 99 --port 8080
```
ปรับ `-ngl`, ขนาด KV cache และ batch ตามผลวัดบนเครื่องจริง

## 5. ทะเบียนเวอร์ชันและ rollback

**model_versions**

| ฟิลด์ | ความหมาย |
|---|---|
| version | เช่น v0.1.0 |
| dataset_version | ชุดข้อมูลที่ใช้เทรน |
| train_run | รหัสรันเทรน |
| llama_cpp_commit | commit ของเครื่องมือแปลงและ quantize |
| quant | ระดับ quantization |
| gguf_sha256 | hash ของไฟล์ |
| eval_report | ลิงก์รายงานจากระยะ 4 |
| status | candidate, staging, production, retired |

**โครงสร้างไฟล์**
```
models/
  v0.1.0/  manifest.json, model-Q5_K_M.gguf, imatrix.dat, eval/
  v0.2.0/  ...
  current -> v0.2.0        # symlink ที่สคริปต์ปล่อยบริการอ่าน
```

- **Promote:** ทำได้เมื่อผ่าน gate ของระยะ 4 เท่านั้น แล้วสลับ symlink `current` และรีสตาร์ต llama-server
- **Rollback:** ชี้ `current` กลับเวอร์ชันก่อนหน้าและรีสตาร์ต ต้องใช้เวลาไม่เกิน 1 นาทีและมีปุ่มเดียวใน UI
- เก็บเวอร์ชัน production ล่าสุดอย่างน้อย 3 เวอร์ชัน ลบไฟล์ merged 16-bit ได้หลังตรวจว่า GGUF ผ่าน gate แล้ว เพราะสร้างใหม่ได้จาก adapter

## 6. งานย่อย

| รหัส | งาน | รายละเอียด | เวลา |
|---|---|---|---|
| T5.1 | สคริปต์ merge | รับ run id แล้วสร้าง merged 16-bit | 0.5 วัน |
| T5.2 | สคริปต์แปลงและ quantize | f16, imatrix, ทุกระดับ พร้อมบันทึก commit ของ llama.cpp | 0.5 วัน |
| T5.3 | ชุดสอบเทียบ imatrix | รวมจากชุดเทรนและโค้ดทั่วไป | 0.5 วัน |
| T5.4 | ทดสอบ template parity | เทียบผลเรนเดอร์ฝั่งเทรนกับฝั่งเซิร์ฟเวอร์ | 0.5 วัน |
| T5.5 | Smoke eval อัตโนมัติ | เรียกด่านย่อยของระยะ 4 | 0.5 วัน |
| T5.6 | ทะเบียนเวอร์ชันและ UI | ตาราง, ปุ่ม promote, ปุ่ม rollback, สลับ symlink | 1 วัน |
| T5.7 | ซ้อม rollback | สลับไปกลับระหว่างสองเวอร์ชันจริง | 0.5 วัน |

## 7. การทดสอบ

- Template parity ผ่านทุกกรณีตัวอย่าง (tools, หลายรอบ)
- ไฟล์ GGUF ทุกระดับโหลดและตอบได้ และผ่าน smoke eval
- Rollback: สลับไปกลับ 3 รอบ และโมเดลที่ทำงานตรงกับที่ตั้งใจทุกครั้ง (ตรวจ hash หรือชื่อไฟล์ที่โหลด)
- ทดสอบ promote ที่ถูกปฏิเสธเมื่อ eval ไม่ผ่าน

## 8. เกณฑ์ตรวจรับ

- [ ] จาก adapter ได้ GGUF ทุกระดับที่ต้องการด้วยคำสั่งเดียว
- [ ] ผ่านการทดสอบ template parity
- [ ] Promote ถูกบล็อกเมื่อไม่ผ่าน gate
- [ ] Rollback กลับเวอร์ชันก่อนหน้าได้ใน 1 คลิก และเสร็จภายใน 1 นาที
- [ ] ทุกเวอร์ชันมี manifest ครบ (ข้อมูล, รันเทรน, commit ของ llama.cpp, hash, รายงาน eval)

## 9. ความเสี่ยง

| ความเสี่ยง | วิธีรับมือ |
|---|---|
| Template ต่างกันระหว่างเทรนกับใช้งาน | ทดสอบ parity ก่อนทุกครั้งและใช้ `--jinja` |
| GGUF จากเครื่องมือคนละเวอร์ชันได้ผลไม่เหมือนกัน | บันทึกและล็อก commit ของ llama.cpp ต่อเวอร์ชันโมเดล |
| Quantize ต่ำเกินไปทำให้ tool call พัง | เลือกระดับจากผล eval และมี Q5_K_M/Q8_0 เป็นทางสำรอง |
| VRAM ไม่พอที่ Q8_0 + context 32k | วัดจริง ลดระดับ quantize หรือลด context หรือขนาด KV cache |
| พื้นที่ดิสก์เต็มจากไฟล์หลายเวอร์ชัน | นโยบายเก็บ 3 เวอร์ชัน และลบ merged 16-bit หลังผ่าน gate |
