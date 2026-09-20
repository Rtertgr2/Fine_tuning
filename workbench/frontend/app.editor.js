/* Workbench UI v1 — examples list + multi-turn editor (T1.5).
   Validation happens ONLY on the backend: POST /validate + POST /render,
   debounced 300 ms (plan 07 principle 6 + §10). */
"use strict";
/* global h, $, api, toast, setView, markHealth */

const CATS = ["tool", "loop", "plan", "sec"];
const ROLES = ["system", "user", "assistant", "tool"];
const STATUS_TH = { draft: "ฉบับร่าง", approved: "อนุมัติ", rejected: "ปฏิเสธ" };
const SOURCE_TH = { manual: "เขียนมือ", generated: "สร้างอัตโนมัติ", active_learning: "เคสจริง" };
let toolSchemasCache = null;
let ed = null; // editor state
const listState = { category: "", status: "", q: "", offset: 0, limit: 20 };

/* =====================================================================
   List (S1): filters, table, pager
   ===================================================================== */
async function renderExamples() {
  const view = $("#view");
  view.replaceChildren(h("div", { class: "view-head" },
    h("h1", { text: "ตัวอย่าง" }),
    h("span", { class: "hint", text: "คลิกแถวเพื่อแก้ · บันทึกได้เสมอ แต่เข้าชุดข้อมูลได้เฉพาะที่ผ่านตรวจ (ไม่มี err)" }),
    h("span", { class: "spacer" }),
    h("button", { class: "btn btn-primary", text: "+ ตัวอย่างใหม่", onclick: () => openEditor(null) })));

  const tools = h("div", { class: "list-tools" },
    h("div", { class: "fld" }, h("label", { class: "fld", text: "หมวด" }), categorySelect()),
    h("div", { class: "fld" }, h("label", { class: "fld", text: "สถานะ" }), statusSelect()),
    h("div", { class: "fld grow" }, h("label", { class: "fld", text: "ค้นหา" }),
      h("input", { type: "search", id: "f-q", value: listState.q, placeholder: "คำในข้อความ / id / group_id", oninput: (e) => { listState.q = e.target.value; listState.offset = 0; listSearchDebounced(); } })));
  view.append(tools);

  const wrap = h("div", { class: "card" }, h("div", { text: "กำลังโหลดรายการ…" }));
  view.append(wrap);
  await refreshList(wrap);
}

function categorySelect(id = "f-cat", value = listState.category, onChange) {
  const sel = h("select", { id, "aria-label": "หมวด", onchange: onChange || ((e) => { listState.category = e.target.value; listState.offset = 0; renderExamples(); }) },
    h("option", { value: "", text: "ทั้งหมด" }),
    CATS.map((c) => h("option", { value: c, text: c, ...(listState.category === c ? { selected: true } : {}) })));
  return sel;
}
function statusSelect(id = "f-status", value = listState.status, onChange) {
  const sel = h("select", { id, "aria-label": "สถานะ", onchange: onChange || ((e) => { listState.status = e.target.value; listState.offset = 0; renderExamples(); }) },
    h("option", { value: "", text: "ทั้งหมด" }),
    ["draft", "approved", "rejected"].map((s) => h("option", { value: s, text: STATUS_TH[s], ...(listState.status === s ? { selected: true } : {}) })));
  return sel;
}

let listSearchTimer = null;
function listSearchDebounced() {
  clearTimeout(listSearchTimer);
  listSearchTimer = setTimeout(() => {
    const wrap = $("#list-wrap");
    if (wrap) refreshList(wrap);
  }, 300);
}

async function refreshList(wrap) {
  const qs = new URLSearchParams({ limit: String(listState.limit), offset: String(listState.offset) });
  if (listState.category) qs.set("category", listState.category);
  if (listState.status) qs.set("status", listState.status);
  if (listState.q) qs.set("q", listState.q);
  let total, items;
  try { ({ total, items } = await api(`/examples?${qs}`)); }
  catch (e) { wrap.replaceChildren(h("div", { class: "big", text: "โหลดรายการไม่สำเร็จ" }), h("div", { text: e.message })); return; }
  wrap.id = "list-wrap";
  if (!items.length) {
    wrap.replaceChildren(h("div", { class: "empty" },
      h("div", { class: "big", text: listState.q || listState.category || listState.status ? "ไม่พบตัวอย่างที่ตรงกับตัวกรอง" : "ยังไม่มีตัวอย่าง" }),
      listState.q || listState.category || listState.status
        ? h("button", { class: "btn btn-sm", text: "ล้างตัวกรอง", onclick: () => { listState.category = ""; listState.status = ""; listState.q = ""; listState.offset = 0; renderExamples(); } })
        : h("button", { class: "btn btn-primary", text: "+ เขียนตัวอย่างแรก", onclick: () => openEditor(null) })));
    return;
  }
  const rows = h("tbody");
  for (const it of items) {
    const tr = h("tr", { class: "rowlink", tabindex: "0", onclick: () => openEditor(it), onkeydown: (e) => { if (e.key === "Enter") openEditor(it); } },
      h("td", { class: "id-cell" }, h("code", { title: it.id, text: it.id.length > 18 ? it.id.slice(0, 15) + "…" : it.id })),
      h("td", {}, h("span", { class: "chip", text: it.category })),
      h("td", {}, h("span", { class: `chip ${it.status === "approved" ? "chip-ok" : it.status === "rejected" ? "chip-err" : "chip-warn"}`, text: STATUS_TH[it.status] })),
      h("td", { text: SOURCE_TH[it.source] || it.source }),
      h("td", { class: "num", text: it.token_count == null ? "—" : it.token_count.toLocaleString("th-TH") }),
      h("td", { text: it.group_id || "—" }),
      h("td", { text: (it.updated_at || "").replace("T", " ").slice(0, 16) }),
      h("td", { class: "actions" },
        h("button", { class: "btn btn-sm", text: "แก้", onclick: (e) => { e.stopPropagation(); openEditor(it); } }),
        h("button", { class: "btn btn-sm btn-danger", text: "ลบ", onclick: async (e) => { e.stopPropagation(); if (!confirm(`ลบตัวอย่าง ${it.id}? ลบแล้วกู้ไม่ได้`)) return; try { await api(`/examples/${it.id}`, { method: "DELETE" }); toast("ลบแล้ว", "ok"); refreshList(wrap); } catch (err) { toast(`ลบไม่สำเร็จ: ${err.message}`, "err"); } } })));
    rows.append(tr);
  }
  const start = listState.offset + 1, end = Math.min(listState.offset + items.length, total);
  const pager = h("div", { class: "pager" },
    h("span", { text: `แสดง ${start}–${end} จาก ${total.toLocaleString("th-TH")} รายการ` }),
    h("button", { class: "btn btn-sm", text: "← ก่อนหน้า", disabled: listState.offset <= 0, onclick: () => { listState.offset = Math.max(0, listState.offset - listState.limit); refreshList(wrap); } }),
    h("button", { class: "btn btn-sm", text: "ถัดไป →", disabled: end >= total, onclick: () => { listState.offset += listState.limit; refreshList(wrap); } }));
  wrap.replaceChildren(
    h("table", { class: "rows" },
      h("thead", {}, h("tr", {},
        h("th", { text: "id" }), h("th", { text: "หมวด" }), h("th", { text: "สถานะ" }), h("th", { text: "แหล่ง" }),
        h("th", { text: "token" }), h("th", { text: "group_id" }), h("th", { text: "อัปเดต" }), h("th", { text: "" }))),
      rows),
    pager);
}

/* =====================================================================
   Editor (S2): multi-turn messages, tools JSON, live validation panel,
   token meter, template preview. Layout: form left, panel right.
   ===================================================================== */
function openEditor(existing) {
  ed = existing ? {
    exId: existing.id,
    original: existing,
    category: existing.category,
    messages: existing.messages.map((m) => ({ role: m.role, content: m.content })),
    toolsText: existing.tools ? JSON.stringify(existing.tools, null, 2) : "",
    source: existing.source,
    groupId: existing.group_id || "",
    status: existing.status,
    dirty: false, timer: null, abort: null,
  } : {
    exId: null, original: null,
    category: "tool",
    messages: [
      { role: "system", content: "" },
      { role: "user", content: "" },
    ],
    toolsText: "", source: "manual", groupId: "",
    status: "draft", dirty: false, timer: null, abort: null,
  };
  window.addEventListener("beforeunload", beforeUnloadGuard);

  const view = $("#view");
  const head = h("div", { class: "view-head" },
    h("button", { class: "btn btn-ghost", text: "← รายการ", onclick: () => backToList() }),
    h("h1", { text: ed.exId ? `แก้ตัวอย่าง ${ed.exId}` : "ตัวอย่างใหม่" }),
    h("span", { class: "chip", id: "hash-chip", title: "content_hash (คำนวณตอนบันทึก)", text: ed.exId ? `hash ${ed.original.content_hash.slice(0, 10)}…` : "ยังไม่มี hash" }));

  const form = h("div", { class: "card" },
    h("div", { class: "field-inline" }, h("label", { class: "fld", text: "หมวด" },
      h("select", { id: "e-cat", onchange: onCategoryChange },
        CATS.map((c) => h("option", { value: c, text: c, ...(ed.category === c ? { selected: true } : {}) })))),
      h("div", { class: "hint", id: "cat-hint", style: "font-size:12px;color:var(--muted)", text: CATEGORY_HINTS[ed.category] || "" })),
    h("div", { class: "meta-row", style: "margin-bottom:12px" },
      h("div", {}, h("label", { class: "fld", text: "group_id (กันข้อมูลรั่วตอนแบ่ง train/val)" }),
        h("input", { type: "text", id: "e-group", value: ed.groupId, placeholder: "เช่น repo/owner-name", oninput: (e) => { ed.groupId = e.target.value; markDirty(); } })),
      h("div", {}, h("label", { class: "fld", text: "แหล่งที่มา" }),
        h("select", { id: "e-source", onchange: (e) => { ed.source = e.target.value; markDirty(); } },
          ["manual", "generated", "active_learning"].map((s) => h("option", { value: s, text: SOURCE_TH[s], ...(ed.source === s ? { selected: true } : {}) }))))),
    h("div", { id: "msg-list", class: "msg-list", "aria-label": "ข้อความในตัวอย่าง" }),
    h("div", { class: "add-row" },
      h("button", { class: "btn btn-sm", text: "+ ข้อความ", onclick: () => addMessage() }),
      h("button", { class: "btn btn-sm", text: "+ tool_call", onclick: () => addToolCall() }),
      h("button", { class: "btn btn-sm", text: "+ ผลลัพธ์ tool", onclick: () => addToolResult() })),
    h("details", { class: "tools-box", id: "tools-box", style: "margin-top:12px" },
      h("summary", { text: "นิยาม tools (JSON) — ใช้กับหมวด tool/loop" }),
      h("textarea", { class: "mono", id: "e-tools", rows: "8", spellcheck: "false",
        placeholder: "เว้นว่าง = ให้ระบบเติมชุด tools มาตรฐานให้ (git_checkout, read_file, write_file)",
        oninput: (e) => { ed.toolsText = e.target.value; markDirty(); scheduleValidate(); } },
        ed.toolsText)));

  const side = h("div", { class: "editor-sticky" },
    h("div", { class: "card" },
      h("h2", { text: "ผลตรวจ", }, h("span", { class: "sub", text: "ฝั่ง backend · อัปเดตอัตโนมัติทุก 300 ms" })),
      h("div", { class: "vpanel-status", id: "vpanel", "aria-live": "polite" }, h("span", { text: "กำลังตรวจ…" })),
      h("div", { class: "vlist", id: "vlist" }),
      h("div", { class: "client-note", id: "client-note", style: "display:none" })),
    h("div", { class: "card" },
      h("h2", { text: "ความยาว + ตัวอย่างหลัง render" }),
      h("div", { class: "tokmeter", id: "tokmeter" }, h("span", { class: "n", text: "…" })),
      h("details", { class: "render-prev" },
        h("summary", { text: "ดูตามที่เรนเดอร์ด้วย template" }),
        h("div", { class: "hint", id: "prev-meta", style: "font-size:12px;color:var(--muted);margin-top:6px" }),
        h("pre", { id: "prev-rendered" }))),
    h("div", { class: "card" },
      h("h2", { text: "บันทึก" }),
      h("div", { class: "field-inline" }, h("label", { class: "fld", text: "สถานะ" },
        h("select", { id: "e-status", onchange: (e) => { ed.status = e.target.value; markDirty(); } },
          ["draft", "approved", "rejected"].map((s) => h("option", { value: s, text: STATUS_TH[s], ...(ed.status === s ? { selected: true } : {}) })))),
        h("div", { class: "hint", style: "font-size:12px;color:var(--muted)", text: "อนุมัติได้เมื่อผ่านตรวจ (ไม่มี err)" })),
      h("div", { class: "edit-actions" },
        h("button", { class: "btn btn-primary", id: "btn-save", text: ed.exId ? "บันทึกการแก้ไข" : "เพิ่มเข้าชุดข้อมูล", onclick: saveExample }),
        h("button", { class: "btn btn-ghost", text: "ยกเลิก", onclick: () => backToList() }),
        h("span", { class: "spacer" }),
        h("span", { class: "dirty-note", id: "dirty-note", hidden: "true", text: "● ยังไม่บันทึก" })),
      h("div", { class: "error-box", id: "error-box", role: "alert" })));

  view.replaceChildren(head, h("div", { class: "editor-grid" }, form, side));
  rebuildMsgList();
  scheduleValidate(0);
  if ((ed.category === "tool" || ed.category === "loop") && !ed.toolsText.trim()) {
    ensureToolSchemas().then(() => {
      if (ed && !ed.toolsText.trim()) { ed.toolsText = defaultToolsText(); const ta = $("#e-tools"); if (ta) ta.value = ed.toolsText; markDirty(); scheduleValidate(); }
    });
  }
}

function markDirty() {
  if (!ed) return;
  ed.dirty = true;
  const n = $("#dirty-note");
  if (n) n.hidden = false;
}

function rebuildMsgList() {
  const list = $("#msg-list");
  if (!list) return;
  list.replaceChildren();
  ed.messages.forEach((m, i) => {
    const roleSel = h("select", { "aria-label": "role ข้อความที่ " + (i + 1), onchange: (e) => { m.role = e.target.value; row.classList.toggle("role-" + e.target.value, true); markDirty(); scheduleValidate(); } },
      ROLES.map((r) => h("option", { value: r, text: r, ...(m.role === r ? { selected: true } : {}) })));
    const ta = h("textarea", { spellcheck: "false", "aria-label": "เนื้อหาข้อความที่ " + (i + 1), oninput: (e) => { m.content = e.target.value; markDirty(); scheduleValidate(); } },
      m.content);
    const row = h("div", { class: `msg role-${m.role}`, dataset: { idx: String(i) } },
      h("div", { class: "msg-head" },
        h("span", { class: "idx", text: `ข้อ ${i + 1}` }),
        roleSel,
        h("span", { class: "spacer" }),
        h("button", { class: "btn btn-sm btn-danger", text: "ลบ", "aria-label": `ลบข้อความที่ ${i + 1}`,
          onclick: () => { ed.messages.splice(i, 1); markDirty(); rebuildMsgList(); scheduleValidate(); } })),
      ta);
    list.append(row);
  });
}

function flashMessage(idx) {
  const row = document.querySelector(`#msg-list .msg[data-idx="${idx}"]`);
  if (!row) return;
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.remove("flash"); void row.offsetWidth; row.classList.add("flash");
}

function addMessage() {
  if (!ed) return;
  ed.messages.push({ role: "user", content: "" });
  markDirty(); rebuildMsgList(); scheduleValidate();
}

const TC_OPEN = "\u003Ctool_call\u003E";
const TC_CLOSE = "\u003C/tool_call\u003E";

function addToolCall() {
  if (!ed) return;
  const skeleton = TC_OPEN + "\n" + JSON.stringify({ name: "read_file", arguments: { path: "src/a.py" } }, null, 2) + "\n" + TC_CLOSE;
  ed.messages.push({ role: "assistant", content: skeleton });
  markDirty(); rebuildMsgList(); scheduleValidate();
}

function addToolResult() {
  if (!ed) return;
  ed.messages.push({ role: "tool", content: "(ผลลัพธ์ tool เป็นข้อความดิบ เช่น Error: no such file or directory: src/a.py)" });
  markDirty(); rebuildMsgList(); scheduleValidate();
}

async function ensureToolSchemas() {
  if (toolSchemasCache) return;
  try { toolSchemasCache = await api("/tools"); } catch { /* offline: leave empty */ }
}

function defaultToolsText() {
  return toolSchemasCache ? JSON.stringify(toolSchemasCache.schemas, null, 2) : "";
}

async function onCategoryChange() {
  if (!ed) return;
  ed.category = $("#e-cat").value;
  const hint = $("#cat-hint");
  if (hint) hint.textContent = CATEGORY_HINTS[ed.category] || "";
  markDirty();
  if ((ed.category === "tool" || ed.category === "loop") && !ed.toolsText.trim()) {
    await ensureToolSchemas();
    if (ed && !ed.toolsText.trim()) {
      ed.toolsText = defaultToolsText();
      const ta = $("#e-tools");
      if (ta) ta.value = ed.toolsText;
      const box = $("#tools-box");
      if (box) box.open = true;
    }
  }
  scheduleValidate();
}

function scheduleValidate(delay) {
  if (!ed) return;
  clearTimeout(ed.timer);
  ed.timer = setTimeout(validateNow, delay == null ? 300 : delay);
}

function draftPayload() {
  let tools = null;
  const text = ed.toolsText.trim();
  if (text) {
    try { tools = JSON.parse(text); }
    catch (err) { return { error: "JSON ของ tools ไม่ถูกต้อง: " + err.message }; }
  }
  return {
    payload: {
      category: ed.category,
      messages: ed.messages.map((m) => ({ role: m.role, content: m.content })),
      tools: tools, source: ed.source,
      group_id: ed.groupId.trim() || null,
    },
  };
}

function validateNow() {
  if (!ed) return;
  const panel = $("#vpanel"), vlist = $("#vlist"), note = $("#client-note");
  if (!panel) return;
  const result = draftPayload();
  if (result.error) {
    note.style.display = "block";
    note.textContent = result.error + " — ยังไม่ส่งตรวจ/บันทึกจนกว่าจะแก้ JSON";
    panel.className = "vpanel-status st-err";
    panel.replaceChildren(h("span", { text: "แก้ JSON ของ tools ก่อน" }));
    vlist.replaceChildren();
    return;
  }
  note.style.display = "none";
  if (ed.abort) ed.abort.abort();
  const ctrl = new AbortController();
  ed.abort = ctrl;
  panel.className = "vpanel-status";
  panel.replaceChildren(h("span", { text: "กำลังตรวจ…" }));
  Promise.all([
    api("/validate", { method: "POST", body: result.payload, signal: ctrl.signal }),
    api("/render", { method: "POST", body: result.payload, signal: ctrl.signal }),
  ]).then((pair) => {
    if (ed && ed.abort === ctrl) { drawValidation(pair[0]); drawPreview(pair[1]); }
  }).catch((e) => {
    if (e && e.name === "AbortError") return;
    markHealth(false);
    panel.className = "vpanel-status st-err";
    panel.replaceChildren(h("span", { text: "ตรวจไม่สำเร็จ: " + e.message }));
  });
}

function drawValidation(report) {
  const panel = $("#vpanel"), vlist = $("#vlist");
  if (!panel) return;
  const errs = report.violations.filter((v) => v.level === "err");
  const warns = report.violations.filter((v) => v.level === "warn");
  if (!report.ok) {
    panel.className = "vpanel-status st-err";
    panel.replaceChildren(h("span", { text: "ต้องแก้ " + errs.length + " ข้อ (err) — ยังเข้าชุดข้อมูลไม่ได้" }));
  } else if (warns.length) {
    panel.className = "vpanel-status st-warn";
    panel.replaceChildren(h("span", { text: "ผ่านตรวจ · ควรดู " + warns.length + " ข้อ (warn)" }));
  } else {
    panel.className = "vpanel-status st-pass";
    panel.replaceChildren(h("span", { text: "ผ่านตรวจทุกกฎ" }));
  }
  vlist.replaceChildren();
  for (const v of report.violations) {
    const item = h("div", { class: "vitem lv-" + v.level },
      h("span", { class: "code", text: v.code }),
      h("span", { class: "text", text: v.message }));
    const m = /messages\[(\d+)\]/.exec(v.location || "");
    if (m) item.append(h("button", {
      class: "loc", text: "ข้อ " + (Number(m[1]) + 1),
      onclick: () => flashMessage(Number(m[1])),
    }));
    vlist.append(item);
  }
  if (report.skipped && report.skipped.length) {
    vlist.append(h("div", { class: "hint", style: "font-size:12px;color:var(--muted)" },
      "ข้าม: " + report.skipped.join(", ")));
  }
}

function drawPreview(p) {
  const meter = $("#tokmeter");
  if (!meter) return;
  const max = p.max_seq_len;
  if (p.token_count == null) {
    meter.className = "tokmeter";
    meter.replaceChildren(
      h("span", { class: "n", text: "—" }),
      h("span", { text: " / tokenizer ไม่พร้อม — C4 ข้าม (เพดาน " + max.toLocaleString("th-TH") + ")" }));
  } else {
    const pct = Math.min(100, (p.token_count / max) * 100);
    meter.className = "tokmeter " + (pct > 90 ? "hot" : pct > 75 ? "warm" : "");
    meter.replaceChildren(
      h("span", { class: "n", text: p.token_count.toLocaleString("th-TH") }),
      h("span", { text: " / " + max.toLocaleString("th-TH") + " tokens" }),
      h("div", {
        class: "bar " + (pct > 90 ? "err" : pct > 75 ? "warn" : "ok"),
        style: "flex:1;margin:0 0 0 8px",
      }, h("i", { style: "width:" + pct + "%" })));
  }
  const meta = $("#prev-meta");
  if (meta) {
    meta.textContent = p.adapter + " · template " + p.template_kind
      + (p.tools_auto_injected ? " · auto-inject ชุด tools มาตรฐานแล้ว" : "");
  }
  const pre = $("#prev-rendered");
  if (pre) pre.textContent = p.rendered;
}

async function saveExample() {
  if (!ed) return;
  const result = draftPayload();
  const box = $("#error-box");
  if (result.error) {
    box.style.display = "block";
    box.textContent = result.error;
    toast("JSON ของ tools ยังไม่ถูกต้อง", "err");
    return;
  }
  box.style.display = "none";
  const btn = $("#btn-save");
  btn.disabled = true;
  try {
    let savedId = ed.exId;
    if (ed.exId) {
      const patch = {
        category: result.payload.category,
        messages: result.payload.messages,
        tools: result.payload.tools,
        source: result.payload.source,
        group_id: result.payload.group_id,
      };
      const out = await api("/examples/" + ed.exId, { method: "PUT", body: patch });
      if (ed.status !== ed.original.status) {
        await api("/examples/" + ed.exId + "/status", { method: "POST", body: { status: ed.status } });
      }
      savedId = out.example.id;
    } else {
      const out = await api("/examples", { method: "POST", body: result.payload });
      savedId = out.example.id;
      if (ed.status !== "draft") {
        await api("/examples/" + savedId + "/status", { method: "POST", body: { status: ed.status } });
      }
    }
    ed.dirty = false;
    toast("บันทึกแล้ว · " + savedId + (result.payload.group_id ? " · group " + result.payload.group_id : ""), "ok");
    backToList(true);
  } catch (e) {
    box.style.display = "block";
    box.textContent = e.status ? "HTTP " + e.status + ": " + e.message : e.message;
    toast("บันทึกไม่สำเร็จ", "err");
    btn.disabled = false;
  }
}

function beforeUnloadGuard(e) {
  if (ed && ed.dirty) { e.preventDefault(); e.returnValue = ""; }
}

function backToList(force) {
  if (!force && ed && ed.dirty && !confirm("มีการแก้ไขที่ยังไม่บันทึก — ออกโดยไม่บันทึก?")) return;
  ed = null;
  window.removeEventListener("beforeunload", beforeUnloadGuard);
  setView("examples");
}

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s" && ed) {
    e.preventDefault();
    saveExample();
  }
});
