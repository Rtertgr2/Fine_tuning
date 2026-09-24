# Fine-tuning Workbench — GPU Runtime / Driver Container Plan

## สถานะ implementation (2026-09-24)

ทำ vertical slice สำหรับเครื่อง Intel Arc B580 แล้ว: `GET /gpu/devices`, `GET /gpu/runtimes`, PF0/PF7/PF8, Docker Compose ที่ map เฉพาะ `/dev/dri/renderD128` และ training script ที่เลือก PyTorch XPU ได้ ทดสอบ QLoRA ด้วย tiny local Llama จนได้ metrics/checkpoint/adapter บน GPU จริง

ขอบเขตที่ยังไม่เสร็จ: ยังไม่มี service แยก, `/gpu/jobs`/allocate/release, per-run container orchestration หรือ stop/resume ผ่าน container; NVIDIA/AMD ยังเป็น `planned` และ full Hermes-2-Pro-8B run ยังไม่ได้ทดสอบ

## 1. เป้าหมาย

เพิ่มชั้น GPU Runtime สำหรับโปรเจกต์ Fine-tuning Workbench เพื่อให้ระบบสามารถตรวจพบ GPU ของเครื่อง Host แล้วเลือก Runtime/Container ที่เหมาะสมโดยอัตโนมัติ จากนั้น Training Workbench เรียก API เดียวเพื่อสั่งงาน โดยไม่ต้องรู้รายละเอียดของ NVIDIA / AMD / Intel มากนัก

แนวคิดหลัก:

```text
Fine-tuning Workbench
        │
        ▼
GPU Runtime API
        │
        ├── Detect GPU
        ├── Validate Driver
        ├── Select Runtime
        ├── Select Docker Image
        ├── Allocate GPU
        └── Launch Training Container
        │
        ├───────────────┬───────────────┐
        ▼               ▼               ▼
    NVIDIA            AMD             Intel
    CUDA              ROCm            oneAPI/XPU
```

---

# 2. ข้อกำหนดสำคัญ: อย่าเอา Kernel Driver ใส่ Docker แบบตรง ๆ

Docker container ใช้ Linux kernel ของ Host ดังนั้น kernel-level GPU driver ไม่ควรถูกออกแบบให้ติดตั้งและแทนที่จากใน container

สำหรับ NVIDIA เอกสาร NVIDIA ระบุว่าต้องติดตั้ง NVIDIA GPU Driver บน Host และใช้ NVIDIA Container Toolkit เพื่อให้ container เข้าถึง GPU ได้

สำหรับ AMD ROCm เอกสาร AMD ระบุว่าต้องมี `amdgpu-dkms` บน Host แล้วจึง expose GPU ให้ container ผ่าน ROCm runtime หรือ device passthrough

สำหรับ Intel ต้องติดตั้ง Intel GPU driver บน Host และ container ใช้ GPU ผ่าน device interface เช่น `/dev/dri`

ดังนั้น architecture ที่ถูกต้องคือ:

```text
HOST
├── Linux Kernel
├── GPU Kernel Driver
│   ├── NVIDIA Driver
│   ├── AMD amdgpu
│   └── Intel GPU Driver
│
└── Docker
      │
      ├── NVIDIA Container Runtime / CDI
      ├── AMD Container Runtime / CDI
      └── Intel /dev/dri
             │
             ▼
         USER SPACE
             │
             ├── CUDA
             ├── ROCm / HIP
             └── oneAPI / Level Zero
```

แหล่งอ้างอิง:
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- AMD ROCm Docker Containers: https://rocm.docs.amd.com/en/develop/install/docker-containers.html
- Intel oneAPI Containers: https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-containers.html

---

# 3. เป้าหมายของคำว่า Universal GPU

ไม่ควรพยายามสร้าง:

```text
Universal Driver
      ↓
NVIDIA + AMD + Intel
```

เพราะ driver stack เป็น vendor-specific

ให้สร้าง:

```text
Universal GPU Runtime API
            ↓
       GPU Abstraction
            ↓
   ┌────────┼─────────┐
   ▼        ▼         ▼
 NVIDIA    AMD      Intel
 CUDA      ROCm     oneAPI/XPU
```

ดังนั้น Training Project จะเรียก API แบบเดียว:

```text
POST /gpu/jobs
```

โดยไม่ต้องเขียนโค้ดแยกเป็น NVIDIA/AMD/Intel ในส่วน UI

---

# 4. GPU Runtime Layer

สร้าง service ใหม่:

```text
gpu-runtime/
├── api/
├── detector/
├── drivers/
│   ├── nvidia/
│   ├── amd/
│   └── intel/
├── containers/
│   ├── nvidia/
│   ├── amd/
│   ├── intel/
│   └── cpu/
├── scheduler/
├── compatibility/
└── health/
```

หน้าที่ของ service:

1. Detect GPU
2. Detect vendor
3. Detect GPU model
4. Detect driver version
5. Detect available VRAM
6. Detect runtime support
7. Select training image
8. Prepare container
9. Expose GPU
10. Start/Stop/Monitor training process

---

# 5. GPU Detection API

สร้าง:

```http
GET /gpu/devices
```

ตัวอย่าง response:

```json
{
  "devices": [
    {
      "id": "gpu-0",
      "vendor": "nvidia",
      "name": "NVIDIA RTX ...",
      "driver_version": "...",
      "memory_total_mb": 24576,
      "memory_free_mb": 22000,
      "runtime": "cuda",
      "status": "ready"
    }
  ]
}
```

AMD:

```json
{
  "vendor": "amd",
  "runtime": "rocm"
}
```

Intel:

```json
{
  "vendor": "intel",
  "runtime": "xpu"
}
```

---

# 6. Runtime Registry

สร้าง registry กลาง:

```yaml
runtimes:

  nvidia-cuda:
    vendor: nvidia
    runtime: cuda
    image: fine-tuning-nvidia:...
    detection:
      command: nvidia-smi

  amd-rocm:
    vendor: amd
    runtime: rocm
    image: fine-tuning-amd:...
    detection:
      command: rocminfo

  intel-xpu:
    vendor: intel
    runtime: xpu
    image: fine-tuning-intel:...
    detection:
      command: sycl-ls
```

ห้าม hard-code runtime กระจายทั่ว project

---

# 7. NVIDIA Container

## Host

ต้องมี:

```text
NVIDIA GPU Driver
Docker
NVIDIA Container Toolkit
```

NVIDIA ระบุว่า Container Toolkit ใช้สำหรับ expose GPU ให้ container และรองรับ Docker/Containerd/CRI-O/Podman

ตัวอย่าง:

```bash
docker run --gpus all ...
```

Docker รองรับการระบุ GPU รายตัวด้วย:

```bash
docker run --gpus '"device=0"'
```

และสามารถใช้ CDI เพื่อทำ hardware injection แบบมาตรฐานได้

## Container

ภายใน container เก็บ:

```text
CUDA runtime
PyTorch CUDA build
Transformers
PEFT
TRL
bitsandbytes
Training application
```

ไม่ต้องติดตั้ง kernel driver ซ้ำใน image

---

# 8. AMD ROCm Container

## Host

ต้องมี AMD GPU driver stack เช่น:

```text
amdgpu
```

และ Docker

AMD ROCm รองรับการ expose GPU เข้า container ผ่าน AMD Container Runtime Toolkit หรือการส่ง device nodes โดยตรง

ตัวอย่าง manual passthrough:

```bash
docker run \
  --device /dev/kfd \
  --device /dev/dri \
  image
```

## Container

เก็บ:

```text
ROCm
HIP
PyTorch ROCm
Transformers
PEFT
TRL
Training application
```

---

# 9. Intel GPU Container

## Host

ต้องมี Intel GPU driver และ GPU device access

โดยทั่วไป GPU compute จะ expose ผ่าน:

```text
/dev/dri
```

Intel oneAPI มี official container images และสามารถใช้ `/dev/dri` เพื่อให้ container เข้าถึง GPU

## Container

เก็บ:

```text
oneAPI
Level Zero
OpenCL
PyTorch XPU
Transformers
PEFT
TRL
Training application
```

---

# 10. Container Images

ไม่ควรสร้าง Docker image เดียวสำหรับทุก vendor

แนะนำ:

```text
registry/
├── fine-tuning-base:cpu
├── fine-tuning-base:nvidia
├── fine-tuning-base:amd
└── fine-tuning-base:intel
```

แล้วแบ่ง version:

```text
fine-tuning:nvidia-cuda12.8
fine-tuning:amd-rocm7.x
fine-tuning:intel-xpu2026
```

หมายเหตุ: version จริงควร pin ตาม compatibility matrix ของ model/framework/GPU ที่รองรับในช่วง deploy ไม่ควรใช้ `latest` เป็น dependency หลักใน production

---

# 11. Common Training Environment

สิ่งที่ควรเหมือนกันทุก container:

```text
Python
PyTorch
Transformers
PEFT
TRL
Datasets
Accelerate
Training code
Config schema
Logging
Metrics format
Checkpoint format
```

สิ่งที่ต่างกันตาม vendor:

```text
NVIDIA → CUDA
AMD    → ROCm/HIP
Intel  → oneAPI/Level Zero/XPU
```

ทำให้ Training Logic ส่วนใหญ่ยังเป็นชุดเดียวกัน:

```text
train.py
```

แต่ device backend ถูก inject ตอน runtime

---

# 12. PyTorch Device Abstraction

Training code ควรเลิก assume:

```python
device = "cuda"
```

เปลี่ยนเป็น:

```python
device = runtime.detect_device()
```

ตัวอย่าง abstraction:

```python
class GPUBackend:
    vendor: str
    device_type: str

    def detect(self):
        ...

    def memory_info(self):
        ...

    def prepare_environment(self):
        ...

    def validate(self):
        ...
```

Implement:

```text
NvidiaBackend
AMDBackend
IntelBackend
CPUBackend
```

---

# 13. Device Selection API

โปรเจกต์หลักเรียก:

```http
POST /gpu/allocate
```

request:

```json
{
  "vendor": "auto",
  "device_id": "auto",
  "memory_required_mb": 16000,
  "framework": "pytorch",
  "precision": "bf16"
}
```

runtime service ตอบ:

```json
{
  "allocation_id": "gpualloc-001",
  "vendor": "nvidia",
  "device_id": "gpu-0",
  "runtime": "cuda",
  "container": "fine-tuning:nvidia-cuda12.8",
  "status": "allocated"
}
```

---

# 14. Container Launch API

ตัว Training Workbench ไม่ควรสร้าง Docker command เอง

ให้เรียก:

```http
POST /containers/training
```

request:

```json
{
  "runtime": "auto",
  "gpu_id": "gpu-0",
  "image": "auto",
  "workspace": "/workspace/r001",
  "command": [
    "python",
    "train.py",
    "--config",
    "config.json"
  ]
}
```

runtime service เป็นคนตัดสินใจ:

```text
GPU
 ↓
Vendor
 ↓
Runtime
 ↓
Container Image
 ↓
Device Mapping
 ↓
Container
```

---

# 15. ใช้ CDI เป็น Abstraction Layer

Docker มี Container Device Interface (CDI) สำหรับ standardizing hardware device exposure ให้กับ containers

CDI เหมาะกับ architecture นี้ เพราะช่วยให้ GPU device configuration ไม่ต้อง hard-code รายละเอียดทั้งหมดไว้ใน Training Workbench

เป้าหมาย:

```text
Training Workbench
        ↓
GPU Runtime Service
        ↓
CDI / Vendor Runtime
        ↓
Container
        ↓
GPU
```

อ้างอิง:
https://docs.docker.com/build/building/cdi/

NVIDIA Container Toolkit รองรับการสร้าง CDI specification:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html

---

# 16. Compatibility Matrix

ต้องสร้าง database/file สำหรับ mapping:

```text
GPU
↓
Driver
↓
Runtime
↓
PyTorch
↓
CUDA/ROCm/oneAPI
↓
Container
↓
Training Features
```

ตัวอย่าง schema:

```yaml
gpu:
  vendor: nvidia
  architecture: ...
  minimum_driver: ...
  runtime: cuda
  runtime_version: ...
  pytorch: ...
  image: ...
  features:
    bf16: true
    fp16: true
    quantization: true
    flash_attention: true
```

AMD:

```yaml
gpu:
  vendor: amd
  architecture: ...
  runtime: rocm
```

Intel:

```yaml
gpu:
  vendor: intel
  runtime: xpu
```

---

# 17. Preflight ที่ต้องเพิ่ม

จาก Preflight เดิม PF1-PF6 ให้เพิ่ม GPU Runtime checks:

```text
PF0 GPU Detection
PF1 Configuration
PF2 Dataset
PF3 Loss Mask
PF4 GPU / Memory
PF5 Dry Run
PF6 Environment
PF7 Container Runtime
PF8 Framework Compatibility
```

## PF0 — GPU Detection

ตรวจ:

```text
Vendor
Model
Driver
VRAM
Runtime
```

## PF7 — Container Runtime

ตรวจ:

```text
Docker installed
Runtime available
GPU visible
Device mapped correctly
```

## PF8 — Framework Compatibility

ตรวจ:

```text
GPU ↔ Driver
GPU ↔ CUDA/ROCm/XPU
Runtime ↔ PyTorch
Model ↔ dtype
Quantization ↔ backend
```

---

# 18. Training Flow ใหม่

```text
User
 ↓
Fine-tuning Workbench
 ↓
GET /gpu/devices
 ↓
Detect GPU
 ↓
Select GPU Runtime
 ↓
Preflight
 ├── GPU
 ├── Driver
 ├── Container
 ├── Framework
 ├── Dataset
 └── Loss Mask
 ↓
PASS
 ↓
Allocate GPU
 ↓
Launch Container
 ↓
train.py
 ↓
Metrics
 ↓
Checkpoint
 ↓
Complete
```

---

# 19. Stop / Resume กับ Container

## Stop

```text
POST /training/runs/{id}/stop
        ↓
GPU Runtime Service
        ↓
docker stop
        ↓
Container stopped
        ↓
DB = stopped
```

## Resume

```text
POST /training/runs/{id}/resume
        ↓
Find checkpoint
        ↓
Allocate GPU
        ↓
Launch new container
        ↓
resume_from_checkpoint
        ↓
Running
```

Container ไม่ควรถูกมองว่าเป็น checkpoint

Checkpoint ต้องอยู่บน persistent storage:

```text
runs/r001/checkpoints/
```

ไม่ใช่เฉพาะใน container filesystem

---

# 20. Volume Architecture

แนะนำ:

```text
Host Storage
    │
    ├── datasets/
    ├── runs/
    │    ├── r001/
    │    └── r002/
    ├── cache/
    └── models/
            │
            ▼
        Docker Volume
```

Mount เข้า container:

```text
/workspace/datasets
/workspace/run
/workspace/cache
/workspace/models
```

หลักการ:

- Code อยู่ใน image
- Runtime อยู่ใน image
- Dataset อยู่ persistent storage
- Checkpoint อยู่ persistent storage
- Metrics อยู่ persistent storage
- Container ทิ้งได้
- Run data ห้ามหาย

---

# 21. Security Model

อย่าให้ API จาก frontend สั่ง Docker โดยตรง

ไม่ควร:

```text
Frontend
   ↓
Docker socket
```

ควร:

```text
Frontend
   ↓
Training Backend
   ↓
GPU Runtime Service
   ↓
Docker
```

Docker daemon / socket ถือเป็น privileged capability

Runtime Service ควร whitelist:

```text
Allowed Images
Allowed Commands
Allowed Volumes
Allowed GPU IDs
Allowed Resources
```

และ sanitize ทุก input

---

# 22. API ที่แนะนำ

```text
GET  /gpu/devices
GET  /gpu/runtimes
GET  /gpu/compatibility

POST /gpu/preflight
POST /gpu/allocate
POST /gpu/release

POST /containers/training
GET  /containers/{id}
POST /containers/{id}/stop

POST /training/runs
POST /training/runs/{id}/stop
POST /training/runs/{id}/resume
```

---

# 23. Project Structure ที่แนะนำ

```text
Fine_tuning/
├── workbench/
│   ├── backend/
│   ├── frontend/
│   └── ...
│
├── gpu-runtime/
│   ├── api/
│   ├── detector/
│   ├── scheduler/
│   ├── compatibility/
│   ├── drivers/
│   │   ├── nvidia/
│   │   ├── amd/
│   │   └── intel/
│   └── main.py
│
├── containers/
│   ├── nvidia/
│   │   └── Dockerfile
│   ├── amd/
│   │   └── Dockerfile
│   ├── intel/
│   │   └── Dockerfile
│   └── cpu/
│       └── Dockerfile
│
├── runtime-config/
│   ├── runtimes.yaml
│   ├── compatibility.yaml
│   └── security.yaml
│
└── docker-compose.yml
```

---

# 24. Development Roadmap

## Phase A — Foundation

- [ ] สร้าง GPU Runtime Service แยกจาก Workbench (ตอนนี้เป็น router/service ภายใน backend)
- [x] GPU detection (Linux DRM sysfs + PyTorch runtime)
- [ ] NVIDIA detection/runtime profile (ยังไม่รองรับ training)
- [ ] AMD detection/runtime profile (ยังไม่รองรับ training)
- [x] Intel detection (Arc B580 / XPU)
- [ ] CPU fallback training profile (ยังคง fail-closed หากไม่มี GPU)
- [x] `/gpu/devices`
- [x] `/gpu/runtimes` (Intel XPU enabled; NVIDIA/AMD planned)

## Phase B — NVIDIA

- [ ] NVIDIA Container Toolkit
- [ ] CUDA image
- [ ] GPU allocation
- [ ] `nvidia-smi` health check
- [ ] PyTorch CUDA test
- [ ] Fine-tuning smoke test

## Phase C — AMD

- [ ] ROCm image
- [ ] `/dev/kfd`
- [ ] `/dev/dri`
- [ ] `rocminfo` health check
- [ ] PyTorch ROCm test
- [ ] Fine-tuning smoke test

## Phase D — Intel

- [x] Intel PyTorch/XPU image (`intel/pytorch:xpu-2.13.0-ubuntu24.04`)
- [x] `/dev/dri/renderD128` device mapping; host kernel driver remains on host
- [x] Level Zero runtime visible through PyTorch XPU device properties
- [ ] `sycl-ls` test (not required by the current image/profile)
- [x] PyTorch XPU test on Intel Arc B580
- [x] 4-bit NF4 + double-quant LoRA training smoke test on a tiny local Llama model
- [ ] Full Hermes-2-Pro-8B training run (requires full model/dataset validation)

## Phase E — Common API

- [ ] Device allocation
- [ ] Container launch
- [ ] Stop
- [ ] Release
- [ ] Metrics
- [ ] Logs
- [ ] Error handling

## Phase F — Integration

- [x] Integrate PF0/PF7/PF8 with current Preflight
- [ ] Integrate device allocation/container launch with Training Run State Machine
- [ ] Integrate stop/resume and per-run container lifecycle
- [x] Keep GPU run/driver metadata with training artifacts
- [x] Add frontend GPU information
- [ ] Verify full Hermes-2-Pro-8B run against current dataset gates
- [ ] Complete per-run isolation/resource limits before production use

Current execution uses one localhost-only Workbench container with a single Intel render node; it does not expose the Docker socket or launch containers per run.

---

# 25. Testing Matrix

อย่างน้อยควรมี:

| GPU Vendor | Detection | Container | PyTorch | Dry Run | LoRA | Resume |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AMD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Intel | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CPU fallback | ✅ | ✅ | ✅ | ✅ | N/A | ✅ |

สำหรับแต่ละ vendor ต้องแยกทดสอบตาม GPU architecture ที่รองรับจริง ไม่ควรใช้ผลจาก GPU รุ่นเดียวแล้วสรุปว่า vendor ทั้งหมดรองรับ

---

# 26. Definition of Done

ระบบถือว่า GPU Runtime Layer พร้อมเมื่อ:

```text
Install Host Driver
        ↓
Install Docker
        ↓
Install Vendor Runtime
        ↓
Start GPU Runtime Service
        ↓
GET /gpu/devices
        ↓
เห็น GPU
        ↓
Preflight
        ↓
Compatibility Check
        ↓
เลือก Container อัตโนมัติ
        ↓
Launch Training
        ↓
GPU ทำงานจริง
        ↓
Metrics
        ↓
Checkpoint
        ↓
Stop
        ↓
Resume
        ↓
Complete
```

---

# 27. สิ่งที่ไม่ควรทำ

## ห้ามทำ

```text
Docker Image
   ↓
ติดตั้ง kernel GPU driver
   ↓
หวังให้ container เปลี่ยน driver ของ Host
```

## ห้ามทำ

```text
Training Backend
   ↓
docker run ...
```

โดยไม่มี GPU Runtime abstraction

## ห้ามทำ

```text
latest CUDA
latest ROCm
latest PyTorch
```

เป็น production dependency

## ห้ามทำ

เก็บ checkpoint ไว้เฉพาะใน container filesystem

---

# 28. Recommended Final Architecture

```text
                      ┌─────────────────────┐
                      │ Fine-tuning UI      │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Training Backend    │
                      │                     │
                      │ Config              │
                      │ Preflight           │
                      │ Run State           │
                      │ Metrics             │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ GPU Runtime API     │
                      │                     │
                      │ Detect              │
                      │ Compatibility       │
                      │ Allocate            │
                      │ Launch              │
                      │ Stop / Release      │
                      └──────────┬──────────┘
                                 │
                         CDI / Vendor Runtime
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
       │ NVIDIA      │   │ AMD         │   │ Intel       │
       │ CUDA        │   │ ROCm        │   │ oneAPI/XPU  │
       └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
              │                 │                  │
              ▼                 ▼                  ▼
       ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
       │ Docker      │   │ Docker      │   │ Docker      │
       │ Container   │   │ Container   │   │ Container   │
       └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                         Persistent Storage
                     ┌────────────────────────┐
                     │ dataset                │
                     │ model cache            │
                     │ checkpoints            │
                     │ metrics                │
                     │ logs                   │
                     └────────────────────────┘
```

---

# 29. แผนที่ควรทำต่อจากแผนเดิม

หลังจากแก้ Training Lifecycle ใน `Fine_tuning_fix_plan.md` แล้ว ให้ต่อด้วย:

```text
TRAINING LIFECYCLE
       ↓
GPU RUNTIME ABSTRACTION
       ↓
NVIDIA
       ↓
AMD
       ↓
Intel
       ↓
Compatibility Matrix
       ↓
Container Orchestration
       ↓
Training Integration
       ↓
Multi-GPU
       ↓
Production Hardening
```

ไม่ควรเริ่มจาก Multi-GPU ก่อนที่ Single-GPU ของแต่ละ vendor จะทำงานผ่าน abstraction เดียวกันได้

---

# 30. Official References

- NVIDIA Container Toolkit — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NVIDIA CDI — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
- Docker CDI — https://docs.docker.com/build/building/cdi/
- AMD ROCm Docker Containers — https://rocm.docs.amd.com/en/develop/install/docker-containers.html
- Intel oneAPI Containers — https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-oneapi-toolkit-with-containers.html
- Intel GPU Driver Installation — https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-linux/latest/install-intel-gpu-drivers.html
