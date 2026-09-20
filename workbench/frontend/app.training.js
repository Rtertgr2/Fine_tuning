/* Workbench UI v1 — training tab (T3): dataset selection, config, submit, metrics.
   Training backend handles Mode A (export script) and Mode B (runner subprocess). */
"use strict";
/* global h, $, api, toast, setView, markHealth */

const DEFAULT_CFG = {
  run_name: "r001",
  base_model: "Qwen/Qwen2.5-Coder-7B-Instruct",
  dataset_version: "v0001",
  seed: 3407,
  quantization: { load_in_4bit: true, quant_type: "nf4", double_quant: true },
  lora: { r: 16, alpha: 32, dropout: 0.0, target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] },
  train: {
    epochs: 2, learning_rate: 2e-4, lr_scheduler: "cosine", warmup_ratio: 0.05,
    per_device_batch_size: 2, grad_accum: 4, max_seq_length: 8192,
    eval_steps: 50, save_steps: 50, early_stopping_patience: 3,
  },
  loss_masking: "assistant_only",
};

async function renderTraining() {
  const view = $("#view");
  view.replaceChildren(h("div", { class: "view-head" },
    h("h1", { text: "เทรนโมเดล" }),
    h("span", { class: "hint", text: "เลือกชุดข้อมูล ตั้งค่า → สั่งเทรน → ดู loss และ metrics แบบ real-time" }),
    h("span", { class: "spacer" })));

  const box = h("div", { class: "empty", text: "กำลังโหลดรายการ..." });
  view.append(box);

  try {
    const [datasets, runs] = await Promise.all([
      api("/datasets"),
      api("/training/runs"),
    ]);
    box.remove();
    view.append(buildTrainingLayout(datasets.items || [], runs.items || []));
  } catch (e) {
    box.className = "empty";
    box.replaceChildren(
      h("div", { class: "big", text: "โหลดข้อมูลไม่สำเร็จ" }),
      h("div", { text: e.message }));
  }
}

function buildTrainingLayout(datasets, runs) {
  const wrap = h("div", { class: "train-layout" });

  /* ---- dataset + run selector ---- */
  const dsOptions = datasets.map(d =>
    h("option", { value: d.version, selected: d.version === "v0001", text: `${d.version} — ${d.total_examples} ตัวอย่าง · seed=${d.seed}` }));

  const runOptions = runs.map(r =>
    h("option", { value: r.run_id, text: `${r.run_id} · ${r.status} · ${r.dataset_version}` }));

  const form = h("div", { class: "card" },
    h("h2", { text: "สร้างรันใหม่" }),
    h("div", { class: "grid-2" },
      fieldSelect("dataset", "ชุดข้อมูล", dsOptions, "เลือก v0001 (หมวด tool/loop/plan/sec)"),
      fieldText("run_name", "ชื่อรัน", "r001", "เช่น r001, r002..."),
    ),
    h("div", { class: "grid-2" },
      fieldText("base_model", "Base Model", "Qwen/Qwen2.5-Coder-7B-Instruct", "HF repo path"),
      fieldSelect("loss_masking", "Loss Masking", [
        h("option", { value: "assistant_only", selected: true, text: "assistant_only (แนะนำ)" }),
        h("option", { value: "all", text: "all" }),
      ], "เทรนเฉพาะส่วนที่ตอบ"),
    ),
    h("h3", { text: "LoRA", style: "margin-top:16px" }),
    h("div", { class: "grid-2" },
      fieldNumber("lora_r", "r", 16, "power of 2"),
      fieldNumber("lora_alpha", "alpha", 32, ""),
      fieldNumber("lora_dropout", "dropout", 0, "0-1"),
    ),
    h("h3", { text: "Training", style: "margin-top:16px" }),
    h("div", { class: "grid-2" },
      fieldNumber("epochs", "epochs", 2, ""),
      fieldNumber("learning_rate", "learning_rate", 0.0002, ""),
      fieldSelect("lr_scheduler", "scheduler", [
        h("option", { value: "cosine", selected: true, text: "cosine" }),
        h("option", { value: "linear", text: "linear" }),
        h("option", { value: "constant", text: "constant" }),
      ], ""),
      fieldNumber("batch_size", "batch_size", 2, "per device"),
      fieldNumber("grad_accum", "grad_accum", 4, ""),
      fieldNumber("max_seq", "max_seq_length", 8192, ""),
      fieldNumber("eval_steps", "eval_steps", 50, ""),
      fieldNumber("save_steps", "save_steps", 50, ""),
      fieldNumber("early_stop", "early_stopping_patience", 3, ""),
    ),
    h("h3", { text: "Quantization", style: "margin-top:16px" }),
    h("div", { class: "grid-2" },
      fieldCheck("load_4bit", "load_in_4bit", true, "4-bit QLoRA"),
      fieldSelect("quant_type", "quant_type", [
        h("option", { value: "nf4", selected: true, text: "nf4" }),
        h("option", { value: "fp4", text: "fp4" }),
      ], ""),
    ),
    h("div", { class: "row2", style: "margin-top:16px" },
      h("button", { class: "btn", onclick: () => runPreflightFromForm(), text: "🔍 ตรวจสอบ (Preflight)" }),
      h("button", { class: "btn btn-primary", onclick: () => submitTraining(), text: "🚀 สั่งเทรน" }),
    ),
    h("div", { id: "preflight-result", style: "margin-top:12px" }));

  /* ---- runs list ---- */
  const runsBox = h("div", { class: "card", style: "margin-top:16px" },
    h("h2", { text: "รันที่มีอยู่" }),
    runs.length === 0
      ? h("p", { class: "hint", text: "ยังไม่มีรัน — สร้างรันแรกด้านบน" })
      : h("table", { class: "mini" },
          h("thead", {}, h("tr", {},
            h("th", { text: "รัน" }),
            h("th", { text: "dataset" }),
            h("th", { text: "status" }),
            h("th", { text: "เริ่ม" }),
            h("th", { text: "" }))),
          h("tbody", {}, runs.map(r =>
            h("tr", { class: "clickable", onclick: () => showRunDetail(r.run_id) },
              h("td", {}, h("code", { class: "k", text: r.run_id })),
              h("td", { text: r.dataset_version }),
              h("td", {}, statusBadge(r.status)),
              h("td", { text: fmtTime(r.created_at) }),
              h("td", {}, h("button", { class: "btn btn-sm", text: "ดู", onclick: (e) => { e.stopPropagation(); showRunDetail(r.run_id); } })))))));

  wrap.append(form, runsBox);
  return wrap;
}

function fieldText(id, label, val, hint) {
  return h("div", { class: "fld" },
    h("label", { class: "fld", text: label }),
    h("input", { type: "text", id: "f-" + id, value: val }),
    hint ? h("small", { class: "hint", text: hint }) : null);
}

function fieldNumber(id, label, val, hint) {
  return h("div", { class: "fld" },
    h("label", { class: "fld", text: label }),
    h("input", { type: "number", id: "f-" + id, value: val, step: "any" }),
    hint ? h("small", { class: "hint", text: hint }) : null);
}

function fieldSelect(id, label, options, hint) {
  return h("div", { class: "fld" },
    h("label", { class: "fld", text: label }),
    h("select", { id: "f-" + id }, options),
    hint ? h("small", { class: "hint", text: hint }) : null);
}

function fieldCheck(id, label, checked, hint) {
  return h("div", { class: "fld" },
    h("label", { class: "fld" },
      h("input", { type: "checkbox", id: "f-" + id, checked: checked }),
      " " + label),
    hint ? h("small", { class: "hint", text: hint }) : null);
}

function statusBadge(status) {
  const cls = status === "running" ? "chip-ok" : status === "completed" ? "chip-ok" : status === "failed" ? "chip-err" : status === "stopped" ? "chip-warn" : "chip-neutral";
  return h("span", { class: `chip ${cls}`, text: status });
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("th-TH", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function readForm() {
  const num = (id, def) => { const v = parseFloat($("#f-" + id).value); return isNaN(v) ? def : v; };
  const str = id => $("#f-" + id).value.trim();
  const chk = id => $("#f-" + id).checked;
  return {
    run_name: str("run_name") || "r001",
    base_model: str("base_model") || "Qwen/Qwen2.5-Coder-7B-Instruct",
    dataset_version: str("dataset") || "v0001",
    seed: 3407,
    quantization: { load_in_4bit: chk("load_4bit"), quant_type: str("quant_type") || "nf4", double_quant: true },
    lora: { r: num("lora_r", 16), alpha: num("lora_alpha", 32), dropout: num("lora_dropout", 0), target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] },
    train: {
      epochs: num("epochs", 2), learning_rate: num("learning_rate", 2e-4),
      lr_scheduler: str("lr_scheduler") || "cosine", warmup_ratio: 0.05,
      per_device_batch_size: num("batch_size", 2), grad_accum: num("grad_accum", 4),
      max_seq_length: num("max_seq", 8192), eval_steps: num("eval_steps", 50),
      save_steps: num("save_steps", 50), early_stopping_patience: num("early_stop", 3),
    },
    loss_masking: str("loss_masking") || "assistant_only",
  };
}

async function runPreflightFromForm() {
  const cfg = readForm();
  const box = $("#preflight-result");
  box.replaceChildren(h("div", { class: "hint", text: "กำลังตรวจสอบ..." }));
  try {
    // Create a run first, then preflight
    const run = await api("/training/runs", { method: "POST", body: { config: cfg } });
    const report = await api(`/training/runs/${run.run_id}/preflight`, { method: "POST" });
    renderPreflight(box, report);
  } catch (e) {
    box.replaceChildren(h("div", { class: "chip chip-err", text: "Preflight ล้มเหลว: " + e.message }));
  }
}

function renderPreflight(box, report) {
  const checks = report.checks || [];
  const failed = checks.filter(c => !c.passed);
  box.replaceChildren(
    h("h3", { text: `ผลตรวจสอบ (${checks.length - failed.length}/${checks.length} ผ่าน)` }),
    h("div", { class: "grid-2" }, checks.map(c =>
      h("div", { class: "card", style: "padding:8px 12px" },
        h("div", { class: "cat-head" },
          h("span", { class: c.passed ? "chip chip-ok" : "chip chip-err", text: c.passed ? "✓" : "✗" }),
          h("strong", { text: c.name })),
        h("div", { class: "hint", text: c.message })))),
    report.can_proceed
      ? h("div", { class: "chip chip-ok", style: "margin-top:8px", text: "✅ พร้อมเทรน" })
      : h("div", { class: "chip chip-err", style: "margin-top:8px", text: "❌ ไม่พร้อมเทรน — แก้ข้อผิดพลาดก่อน" }));
}

async function submitTraining() {
  const cfg = readForm();
  try {
    const run = await api("/training/runs", { method: "POST", body: { config: cfg } });
    toast(`สร้างรัน ${run.run_id} แล้ว — กำลังเริ่มเทรน`, "ok");
    setView("training"); // refresh
  } catch (e) {
    toast("สั่งเทรนไม่สำเร็จ: " + e.message, "err");
  }
}

async function showRunDetail(runId) {
  const view = $("#view");
  view.replaceChildren(h("div", { class: "view-head" },
    h("h1", { text: `รัน ${runId}` }),
    h("button", { class: "btn btn-sm", onclick: () => setView("training"), text: "← กลับ" })));

  const box = h("div", { class: "empty", text: "กำลังโหลด..." });
  view.append(box);

  try {
    const [run, metrics] = await Promise.all([
      api(`/training/runs/${runId}`),
      api(`/training/runs/${runId}/metrics`),
    ]);
    box.remove();
    view.append(buildRunDetail(run, metrics.metrics || []));
    // Auto-refresh if running
    if (run.status === "running") {
      setTimeout(() => { if ($("#view")) showRunDetail(runId); }, 5000);
    }
  } catch (e) {
    box.replaceChildren(h("div", { class: "big", text: "โหลดไม่สำเร็จ" }), h("div", { text: e.message }));
  }
}

function buildRunDetail(run, metrics) {
  const wrap = h("div");
  wrap.append(h("div", { class: "grid-2", style: "margin-bottom:16px" },
    h("div", { class: "card" },
      h("h2", { text: "สถานะ" }),
      h("div", { class: "cat-head" },
        statusBadge(run.status),
        h("span", { class: "spacer" }),
        run.status === "running"
          ? h("button", { class: "btn btn-sm", onclick: () => stopRun(run.run_id), text: "⏸ หยุด" })
          : run.status === "stopped"
            ? h("button", { class: "btn btn-sm", onclick: () => resumeRun(run.run_id), text: "▶ ต่อ" })
            : null),
      h("div", { class: "hint", text: `สร้าง ${fmtTime(run.created_at)} · อัปเดต ${fmtTime(run.completed_at || run.created_at)}` })),
    h("div", { class: "card" },
      h("h2", { text: "ตั้งค่า" }),
      h("pre", { class: "mono", style: "font-size:12px;max-height:200px;overflow:auto", text: JSON.stringify(JSON.parse(run.config_json || "{}"), null, 2) }))));

  // Metrics chart (text-based sparkline)
  const trainLoss = metrics.filter(m => "train_loss" in m);
  const evalLoss = metrics.filter(m => "eval_loss" in m);

  if (trainLoss.length > 0) {
    const lossCard = h("div", { class: "card", style: "margin-top:16px" },
      h("h2", { text: "Train Loss" }),
      h("div", { class: "spark" }, sparkline(trainLoss.map(m => m.train_loss), 60, 12)),
      h("div", { class: "hint", text: `จุดสุดท้าย: ${trainLoss[trainLoss.length - 1]?.train_loss?.toFixed(4) || "—"} ที่ step ${trainLoss[trainLoss.length - 1]?.step || "—"}` }));
    wrap.append(lossCard);
  }

  if (evalLoss.length > 0) {
    const evalCard = h("div", { class: "card", style: "margin-top:16px" },
      h("h2", { text: "Eval Loss" }),
      h("div", { class: "spark" }, sparkline(evalLoss.map(m => m.eval_loss), 60, 12)),
      h("div", { class: "hint", text: `ดีที่สุด: ${Math.min(...evalLoss.map(m => m.eval_loss)).toFixed(4)} ที่ step ${evalLoss.find(m => m.eval_loss === Math.min(...evalLoss.map(e => e.eval_loss)))?.step || "—"}` }));
    wrap.append(evalCard);
  }

  if (metrics.length === 0) {
    wrap.append(h("div", { class: "card", style: "margin-top:16px" },
      h("p", { class: "hint", text: "ยังไม่มี metrics — รอเทรนเริ่มทำงาน" })));
  }

  return wrap;
}

function sparkline(values, w, h) {
  if (values.length < 2) return h("div", { class: "hint", text: "รอข้อมูล..." });
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = w / (values.length - 1);
  const points = values.map((v, i) => `${i * step},${h - ((v - min) / range) * h}`).join(" ");
  const svg = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block"><polyline points="${points}" fill="none" stroke="#C8553D" stroke-width="1.5"/></svg>`;
  return h("div", { innerHTML: svg });
}

async function stopRun(runId) {
  try {
    await api(`/training/runs/${runId}/stop`, { method: "POST" });
    toast(`หยุดรัน ${runId} แล้ว`, "ok");
    showRunDetail(runId);
  } catch (e) { toast("หยุดไม่ได้: " + e.message, "err"); }
}

async function resumeRun(runId) {
  try {
    await api(`/training/runs/${runId}/resume`, { method: "POST" });
    toast(`กลับมาเทรน ${runId} ต่อแล้ว`, "ok");
    showRunDetail(runId);
  } catch (e) { toast("กลับมาไม่ได้: " + e.message, "err"); }
}

// Export for app.core.js
window.renderTraining = renderTraining;
