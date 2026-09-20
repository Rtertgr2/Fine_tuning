/* Workbench UI v1 — core: DOM helpers, API client, tabs, dashboard (T1.9).
   Validation/rendering truth lives in the backend; this file only displays. */
"use strict";

/* ---- API base: same-origin when served by FastAPI; override via ?api= or localStorage ---- */
const params = new URLSearchParams(location.search);
const API = params.get("api") || localStorage.getItem("wb_api") ||
  (location.origin.startsWith("http") ? "" : "http://127.0.0.1:8300");

/* ---- tiny DOM helper: h(tag, attrs, children) — textContent for user data (XSS-safe) ---- */
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (sel, root = document) => root.querySelector(sel);

/* ---- API client ---- */
class ApiError extends Error {
  constructor(status, body) {
    const detail = (body && (body.detail || body.error)) || `HTTP ${status}`;
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status; this.body = body;
  }
}
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(API + path, {
      headers: opts.body && !(opts.body instanceof Blob)
        ? { "Content-Type": "application/json" } : undefined,
      ...opts,
      body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
    });
  } catch (e) {
    markHealth(false);
    throw new ApiError(0, { detail: "เชื่อมต่อ backend ไม่ได้ — ตรวจว่า uvicorn รันอยู่แล้วกดเชื่อมต่อใหม่" });
  }
  if (!res.ok) {
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    throw new ApiError(res.status, body);
  }
  markHealth(true);
  return res.status === 204 ? null : res.json();
}

/* ---- health + adapter chip ---- */
function markHealth(ok) {
  const dot = $("#health-dot"), txt = $("#health-text");
  dot.classList.toggle("on", ok); dot.classList.toggle("off", !ok);
  txt.textContent = ok ? "เชื่อมต่อ backend แล้ว" : "เชื่อมต่อ backend ไม่ได้";
}
async function refreshHealth() {
  try {
    const info = await api("/health");
    markHealth(true);
    const ad = await api("/adapters").catch(() => null);
    const chip = $("#adapter-chip");
    if (ad) chip.textContent = `${ad.active} · ${ad.tokenizer_available ? "tokenizer พร้อม" : "tokenizer ไม่พร้อม"} · max ${ad.max_seq_len.toLocaleString()} tok`;
    else chip.textContent = info.adapter || "…";
    chip.title = `adapter: ${info.adapter} · db: ${info.db}`;
  } catch { markHealth(false); }
}

/* ---- toast ---- */
function toast(msg, kind = "", ms = 3500) {
  const root = $("#toast-root");
  const t = h("div", { class: `toast ${kind}`, role: "status", text: msg });
  root.append(t);
  setTimeout(() => t.remove(), ms);
}

/* ---- fmt ---- */
const fmtN = (n) => Number(n ?? 0).toLocaleString("th-TH");
const fmtPct = (num, den) => den > 0 ? Math.round((num / den) * 100) : 0;

/* ---- tabs + routing ---- */
/* renderExamples อยู่ใน app.editor.js และ renderTraining อยู่ใน app.training.js
   ทั้งสองถูกโหลดหลังจากไฟล์นี้ ใช้ getter เพื่อไม่ให้ ReferenceError ตอน parse
   ถ้ายังไม่โหลด return stub แสดง "กำลังโหลด..." แทนการ throw */
const VIEWS = {};
VIEWS.dashboard = renderDashboard;
Object.defineProperty(VIEWS, "examples", {
  get() { return window.renderExamples || (() => { $("#view").replaceChildren(h("div", { class: "empty", text: "กำลังโหลด..." })); }); },
});
Object.defineProperty(VIEWS, "training", {
  get() { return window.renderTraining || (() => { $("#view").replaceChildren(h("div", { class: "empty", text: "กำลังโหลด..." })); }); },
});
let currentView = params.get("view") === "examples" ? "examples" : params.get("view") === "training" ? "training" : "dashboard";
function setView(name) {
  currentView = VIEWS[name] ? name : "dashboard";
  for (const t of ["dashboard", "examples", "training"]) {
    $("#tab-" + t).setAttribute("aria-selected", String(t === currentView));
  }
  history.replaceState(null, "", `?view=${currentView}`);
  VIEWS[currentView]();
}

/* =====================================================================
   Dashboard (T1.9): progress per category, pass rate, sec PASS/REJECT,
   token histogram, validator-code statistics.
   ===================================================================== */
const CATEGORY_LABELS = { tool: "tool", loop: "loop", plan: "plan", sec: "sec" };
const CATEGORY_HINTS = { tool: "ใช้ tool จริง (git/read/write)", loop: "ล้มแล้วลุก หาทางแก้", plan: "วิเคราะห์แผน → JSON", sec: "รีวิวความปลอดภัย diff" };
const STATUS_TH = { draft: "ฉบับร่าง", approved: "อนุมัติ", rejected: "ปฏิเสธ" };
const SOURCE_TH = { manual: "เขียนมือ", generated: "สร้างอัตโนมัติ", active_learning: "เคสจริง" };

async function renderDashboard() {
  const view = $("#view");
  view.replaceChildren(h("div", { class: "view-head" },
    h("h1", { text: "ภาพรวม" }),
    h("span", { class: "hint", text: "ความคืบหน้าชุดข้อมูล · อัตราผ่านตรวจ · สัดส่วน PASS/REJECT · ความยาว token" }),
    h("span", { class: "spacer" }),
    h("button", { class: "btn btn-sm", onclick: () => renderDashboard(), text: "รีเฟรช" })));
  const box = h("div", { class: "empty", text: "กำลังโหลดสถิติ…" });
  view.append(box);

  let s;
  try { s = await api("/stats"); }
  catch (e) { box.replaceChildren(h("div", { class: "big", text: "โหลดสถิติไม่สำเร็จ" }), h("div", { text: e.message })); return; }

  if (!s.total) {
    box.className = "empty";
    box.replaceChildren(
      h("div", { class: "big", text: "ยังไม่มีตัวอย่างในระบบ" }),
      h("div", { text: "เริ่มเขียนตัวอย่างแรกจากแท็บ ตัวอย่าง — ผ่านตรวจครบ 100 ตัวอย่างคือถึงเป้า M0" }),
      h("p", {}, h("button", { class: "btn btn-primary", text: "ไปเขียนตัวอย่าง", onclick: () => setView("examples") })));
    return;
  }
  box.remove();
  const val = s.validation;
  const passed = Object.values(val.by_category).reduce((a, c) => a + (c.pass || 0), 0);
  const M0_TARGET = 100;

  /* top progress */
  const passPct = fmtPct(passed, s.total);
  view.append(h("div", { class: "card", style: "margin-bottom:16px" },
    h("div", { class: "cat-head" },
      h("div", { class: "cat-name", text: `ตัวอย่างทั้งหมด ${fmtN(s.total)} รายการ` },
        h("small", { text: `ผ่านตรวจ (ไม่มี err) ${fmtN(passed)} รายการ` })),
      h("span", { class: `chip ${passed >= M0_TARGET ? "chip-ok" : "chip-neutral"}`, text: `เป้า M0: ผ่านตรวจ 100 ตัวอย่าง` })),
    h("div", { class: "bar ok" }, h("i", { style: `width:${Math.min(passPct, 100)}%` })),
    h("div", { class: "row2", style: "justify-content:space-between" },
      h("span", { text: `ผ่านตรวจ ${passPct}% ของทั้งหมด` }),
      h("span", { text: `อนุมัติ ${fmtN(s.by_status.approved)} · ร่าง ${fmtN(s.by_status.draft)} · ปฏิเสธ ${fmtN(s.by_status.rejected)}` })),
    !val.tokenizer_available
      ? h("div", { class: "dirty-note", style: "margin-top:8px", text: "tokenizer ไม่พร้อม — C4 (จำกัด token) และ C6 (ซ้ำ) ถูกข้ามบางส่วน ดู /health" })
      : null));

  /* per-category cards */
  const cats = h("div", { class: "stat-grid", style: "margin-bottom:16px" });
  for (const [cat, n] of Object.entries(s.by_category)) {
    const v = (val.by_category[cat] || { pass: 0, fail: 0 });
    const pct = fmtPct(v.pass, v.pass + v.fail);
    cats.append(h("div", { class: "card stat-card" },
      h("div", { class: "cat-head" },
        h("div", { class: "cat-name", text: `${CATEGORY_LABELS[cat] || cat} × ${fmtN(n)}` }),
        h("span", { class: "lbl", text: CATEGORY_HINTS[cat] || "" })),
      h("div", { class: "bar " + (pct >= 100 ? "ok" : pct >= 60 ? "" : "warn") }, h("i", { style: `width:${pct}%` })),
      h("div", { class: "row2" },
        h("span", { text: `ผ่านตรวจ ${fmtN(v.pass)} · ต้องแก้ ${fmtN(v.fail)}` }),
        h("span", { class: "n", text: `${pct}%` }))));
  }
  view.append(cats);

  /* middle row: sources + sec share */
  const secCounts = s.sec.status_counts || {};
  const secTotal = Object.values(secCounts).reduce((a, b) => a + b, 0);
  const secPass = secCounts["PASS"] || 0;
  const secPct = secTotal ? fmtPct(secPass, secTotal) : 0;
  const secInWindow = secPct >= 35 && secPct <= 65;
  view.append(h("div", { class: "grid-2", style: "margin-bottom:16px" },
    h("div", { class: "card" },
      h("h2", { text: "สถานะและแหล่งที่มา" }),
      h("div", { style: "display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px" },
        ["draft", "approved", "rejected"].map((k) =>
          h("span", { class: `chip ${k === "approved" ? "chip-ok" : k === "rejected" ? "chip-err" : "chip-warn"}`, text: `${STATUS_TH[k]} ${fmtN(s.by_status[k])}` }))),
      h("div", { style: "display:flex;flex-wrap:wrap;gap:8px" },
        ["manual", "generated", "active_learning"].map((k) =>
          h("span", { class: "chip", text: `${SOURCE_TH[k]} ${fmtN(s.by_source[k])}` })))),
    h("div", { class: "card" },
      h("h2", { text: "หมวด sec — สัดส่วน PASS/REJECT" },
        h("span", { class: "sub", text: "เป้า ~50% (กฎ S6, หน้าต่าง 35–65%)" })),
      secTotal
        ? [
            h("div", { class: "cat-head" }, h("span", { class: "num", style: "font-size:24px", text: `${secPct}% PASS` }),
              h("span", { class: `chip ${secInWindow ? "chip-ok" : "chip-warn"}`, text: secInWindow ? "อยู่ในหน้าต่างเป้า" : "เยื้องเป้า — ปรับชุดให้สมดุล" })),
            h("div", { class: "bar" + (secInWindow ? "" : " warn") }, h("i", { style: `width:${secPct}%` })),
            h("div", { class: "row2" },
              h("span", { text: `PASS ${fmtN(secPass)} · REJECT ${fmtN(secCounts["REJECT"] || 0)} · แยก JSON ไม่ได้ ${fmtN(secCounts["unparsed"] || 0)}` })),
          ]
        : h("div", { class: "hint", text: "ยังไม่มีตัวอย่างหมวด sec" }))));

  /* token histogram + validator code stats */
  const histMax = Math.max(1, ...Object.values(s.token_histogram));
  const histCard = h("div", { class: "card" },
    h("h2", { text: "ความยาว token หลัง render" },
      h("span", { class: "sub", text: "เพดาน 8,192 (C4)" + (s.token_count_unknown ? ` · ไม่ทราบ ${fmtN(s.token_count_unknown)} รายการ` : "") })));
  for (const [bucket, n] of Object.entries(s.token_histogram)) {
    histCard.append(
      h("div", { class: "hist-row" },
        h("span", { text: bucket }),
        h("div", { class: "bar" }, h("i", { style: `width:${(n / histMax) * 100}%` })),
        h("span", { class: "n", text: fmtN(n) })));
  }

  function codeTable(title, counts, emptyText) {
    const rows = Object.entries(counts);
    if (!rows.length) return h("p", { class: "hint", text: emptyText });
    const body = h("tbody");
    for (const [code, n] of rows) {
      body.append(h("tr", {},
        h("td", {}, h("code", { class: "k", text: code })),
        h("td", { class: "num", text: fmtN(n) })));
    }
    return h("table", { class: "mini" },
      h("thead", {}, h("tr", {},
        h("th", { text: title }),
        h("th", { style: "text-align:right", text: "ครั้ง" }))),
      body);
  }

  const statCard = h("div", { class: "card" },
    h("h2", { text: "สถิติกฎตรวจ" },
      h("span", { class: "sub", text: "กฎที่ตัดข้อมูลมากผิดปกติควรทบทวน (แผน 01 §11)" })));
  statCard.append(codeTable("กฎ (err)", val.err_counts_by_code, "ไม่มี err เลย 🎉"));
  const warnTable = codeTable("กฎ (warn)", val.warn_counts_by_code, "");
  if (warnTable.tagName === "TABLE") warnTable.style.marginTop = "8px";
  statCard.append(warnTable);

  view.append(h("div", { class: "grid-2" }, histCard, statCard));
}

/* ---- boot ---- */
window.addEventListener("DOMContentLoaded", () => {
  $("#api-label").textContent = API ? `API: ${API}` : "API: same-origin";
  $("#brand-link").addEventListener("click", (e) => { e.preventDefault(); setView("dashboard"); });
  $("#tab-dashboard").addEventListener("click", () => setView("dashboard"));
  $("#tab-examples").addEventListener("click", () => setView("examples"));
  $("#tab-training").addEventListener("click", () => setView("training"));
  $("#btn-reconnect").addEventListener("click", () => { refreshHealth(); setView(currentView); });
  refreshHealth();
  setView(currentView);
});
