import { createMap } from "./map.js";
import { createGroupCard } from "./group-card.js";

const byId = (id) => document.getElementById(id);
const words = (value) => String(value ?? "unavailable").replaceAll("_", " ");
const percent = (value) => (value == null ? "unavailable" : `${Math.round(value * 100)}%`);
const TONES = ["none", "low", "medium", "high"];
const toneColor = (level) => `var(--concern-${TONES.includes(level) ? level : "none"})`;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}

const state = { meta: null, rows: new Map(), history: [], files: [], busy: false };
const map = createMap(byId("stage"), { onSelect: (id) => selectCloud(id) });
const selectCloud = createGroupCard(byId("card"), (id) => map.select(id), (visible) => {
  byId("thread").hidden = visible;
  byId("chips").hidden = visible;
  if (!visible) byId("thread").scrollTop = byId("thread").scrollHeight;
});

function notice(message, command) {
  const box = byId("notice");
  box.replaceChildren(node("p", message));
  if (command) box.append(node("code", command));
  box.hidden = false;
}

function renderChips() {
  const chips = byId("chips");
  chips.replaceChildren();
  state.files.forEach((file, index) => {
    const chip = node("span", file.name, "chip");
    const remove = node("button", "×", "plain");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.addEventListener("click", () => { state.files.splice(index, 1); renderChips(); });
    chip.append(remove);
    chips.append(chip);
  });
}

function addFiles(list) {
  const limits = state.meta.chat.limits;
  for (const file of list) {
    if (state.files.length >= limits.attachments) {
      say(`At most ${limits.attachments} attachments.`, "error");
      break;
    }
    if (file.size > limits.attachment_mb * 1024 * 1024) say(`${file.name} is larger than ${limits.attachment_mb} MB.`, "error");
    else state.files.push(file);
  }
  renderChips();
}

const toBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
  reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
  reader.readAsDataURL(file);
});

function say(text, kind) {
  const thread = byId("thread");
  const message = node("div", text, `msg ${kind}`);
  thread.append(message);
  thread.scrollTop = thread.scrollHeight;
  return message;
}

function renderVerdict(result) {
  const box = node("div", null, "verdict");
  const { verdict, judge, placement } = result;
  const content = result.content_assessment;
  if (content) {
    const head = node("p");
    const dot = node("i", null, "dot");
    dot.style.setProperty("--label-color", toneColor(content.concern_level));
    const category = content.category === "propaganda" ? "propaganda-style persuasion" : words(content.category);
    head.append(dot, node("strong", category), ` · content concern ${words(content.concern_level)} · confidence ${percent(content.confidence)}`);
    box.append(head, node("p", content.explanation));
    if (content.techniques.length) box.append(node("p", content.techniques.map(words).join(" · ")));
    for (const evidence of content.evidence) box.append(node("blockquote", evidence, "content-evidence"));
  } else {
    box.append(node("p", `Content assessment unavailable: ${result.content_review?.reason || "no result returned"}.`));
  }
  if (verdict) {
    box.append(node("p", `Separate spread review: ${words(verdict.concern_level)}. One item does not establish coordination.`, "spread-review"));
  } else {
    box.append(node("p", `Spread review unavailable: ${judge.reason}. This does not change the content assessment.`, "spread-review"));
  }
  const cloud = placement.nearest_cloud;
  const where = node("p");
  if (cloud) {
    const row = state.rows.get(cloud.cluster_id);
    const hidden = row?.discarded;
    const link = node(hidden ? "span" : "button", cloud.label || `group ${cloud.cluster_id}`, hidden ? "" : "plain");
    if (!hidden) {
      link.type = "button";
      link.addEventListener("click", () => selectCloud(cloud.cluster_id));
    }
    const qualifiers = [
      hidden ? "below display threshold" : null,
      row?.analysis_stale ? "assessment needs refresh" : null,
      row && row.status !== "ok" ? `assessment ${words(row.status)}` : null,
    ].filter(Boolean);
    where.append("Closest group: ", link, ` · group review priority ${cloud.review_priority}/100${qualifiers.length ? ` (${qualifiers.join("; ")})` : ""}.`);
    box.append(where, node("p", "The group priority describes the existing group, not this submission. A semantic match alone does not establish shared framing or coordination.", "small"));
  } else {
    where.textContent = "No matching group among the closest posts.";
    box.append(where);
  }
  box.append(node("p", `content: ${result.content_review?.model || result.models.chat} · spread: ${judge.model}`, "small"));
  return box;
}

async function send() {
  const input = byId("prompt");
  const text = input.value.trim();
  if (state.busy || (!text && !state.files.length)) return;
  state.busy = true;
  const files = state.files.splice(0);
  renderChips();
  say([text, ...files.map((file) => `＋ ${file.name}`)].filter(Boolean).join("\n"), "user");
  input.value = "";
  resize();
  const pending = say("", "assistant");
  const working = node("span", null, "working");
  const spinner = node("span", null, "spinner");
  spinner.setAttribute("aria-hidden", "true");
  working.setAttribute("role", "status");
  working.append(spinner, "Nemotron is assessing content and matching groups…");
  pending.append(working);
  try {
    const attachments = await Promise.all(files.map(async (file) => ({ name: file.name, type: file.type, data: await toBase64(file) })));
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, attachments, history: state.history.slice(-8) }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`;
      throw new Error(detail);
    }
    pending.replaceChildren(node("span", body.reply), renderVerdict(body));
    state.history.push(
      { role: "user", content: (text || `Attached: ${files.map((file) => file.name).join(", ")}`).slice(0, 4000) },
      { role: "assistant", content: body.reply.slice(0, 4000) },
    );
    map.setMarker({
      label: files[0]?.name || "your text",
      cloudId: body.placement.nearest_cloud?.cluster_id,
    });
  } catch (error) {
    pending.className = "msg assistant error";
    pending.textContent = error.message;
  } finally {
    state.busy = false;
    byId("thread").scrollTop = byId("thread").scrollHeight;
  }
}

function resize() {
  const input = byId("prompt");
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

function setupChat() {
  const input = byId("prompt");
  const composer = byId("composer");
  composer.addEventListener("focusin", () => selectCloud(null));
  composer.addEventListener("click", () => {
    if (!byId("card").hidden) selectCloud(null);
  });
  if (!state.meta.chat.available) {
    input.disabled = true;
    byId("files").disabled = true;
    input.placeholder = state.meta.chat.error;
    return;
  }
  input.addEventListener("input", resize);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send();
    }
  });
  input.addEventListener("paste", (event) => {
    if (event.clipboardData?.files.length) {
      event.preventDefault();
      addFiles(event.clipboardData.files);
    }
  });
  composer.addEventListener("submit", (event) => { event.preventDefault(); send(); });
  byId("files").addEventListener("change", (event) => { addFiles(event.target.files); event.target.value = ""; });
  for (const name of ["dragenter", "dragover"]) {
    document.addEventListener(name, (event) => { event.preventDefault(); composer.classList.add("dragging"); });
  }
  for (const name of ["dragleave", "drop"]) {
    document.addEventListener(name, (event) => { event.preventDefault(); composer.classList.remove("dragging"); });
  }
  document.addEventListener("drop", (event) => {
    if (event.dataTransfer?.files.length) {
      selectCloud(null);
      addFiles(event.dataTransfer.files);
    }
  });
}

// On wide screens the panel shares the right edge, leaving the remaining area for bubbles.
const docked = matchMedia("(min-width: 1000px)");
const dock = () => map.setInset(docked.matches ? byId("composer").getBoundingClientRect().width + 40 : 0, docked.matches ? 0 : 210);
docked.addEventListener("change", dock);
dock();

function renderInfo() {
  const { meta } = state;
  byId("info").textContent = [
    `${meta.n_posts.toLocaleString()} posts`,
    `${meta.n_clusters - meta.n_discarded} groups`,
    `${meta.n_discarded} dropped at 0/100`,
    "size = posts · colour = concern",
  ].filter(Boolean).join(" · ");
  if (meta.dataset?.burstiness_reliable === false) byId("info").textContent += " · burst timing excluded (short snapshot)";
  if (meta.n_stale) byId("info").textContent += ` · ${meta.n_stale} model assessments need refreshing`;
}

async function start() {
  try {
    const [metaResponse, rowsResponse] = await Promise.all([fetch("/api/meta"), fetch("/api/clusters?include_hidden=true")]);
    if (metaResponse.status === 503) return notice("No processed posts yet. Run this, then restart the server:", "python -m detector all --posts 10000");
    if (!metaResponse.ok || !rowsResponse.ok) throw new Error("the API returned an error");
    state.meta = await metaResponse.json();
    const rows = await rowsResponse.json();
    setupChat();
    for (const row of rows) state.rows.set(row.cluster_id, row);
    map.setClouds(rows, { preloaded: true });
    renderInfo();
    if (rows.every((row) => row.discarded)) notice("No groups above the review threshold. You can still assess material in the chat.");
    else byId("notice").hidden = true;
  } catch (error) {
    notice(`Could not load the map: ${error.message}. Reload to retry.`);
  }
}
start();
