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

export function createChat({ meta, rows, selectCloud, map, live = false }) {
const state = { meta, rows, history: [], files: [], busy: false };

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
  if (state.busy) {
    say("Wait for the current request before adding attachments.", "error");
    return;
  }
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
  const section = (title) => {
    const item = node("section", null, "verdict-section");
    item.append(node("h3", title));
    box.append(item);
    return item;
  };
  const contentSection = section("Content assessment");
  if (content) {
    const head = node("p");
    const dot = node("i", null, "dot");
    dot.style.setProperty("--label-color", toneColor(content.concern_level));
    const category = content.category === "propaganda" ? "propaganda-style persuasion" : words(content.category);
    head.append(dot, node("strong", category), ` · content concern ${words(content.concern_level)} · confidence ${percent(content.confidence)}`);
    contentSection.append(head, node("p", content.explanation));
    if (content.techniques.length) contentSection.append(node("p", `Techniques: ${content.techniques.map(words).join(" · ")}`));
    if (content.evidence.length) {
      const evidence = section("Evidence");
      for (const item of content.evidence) evidence.append(node("blockquote", item, "content-evidence"));
    }
  } else {
    contentSection.append(node("p", `Unavailable: ${result.content_review?.reason || "no result returned"}.`));
  }
  const spread = section("Spread review");
  if (verdict) {
    spread.append(node("p", `${words(verdict.concern_level)} concern. ${verdict.explanation}`));
  } else {
    spread.append(node("p", `Unavailable: ${judge.reason}. The content assessment is unaffected.`));
  }
  spread.append(node("p", "One item does not establish coordination.", "small"));
  const cloud = placement.nearest_cloud;
  const group = section(live ? "Related group in the live window" : "Related group");
  if (cloud) {
    const row = state.rows.get(cloud.cluster_id);
    const hidden = row?.discarded || (live && !row);
    const link = node(hidden ? "span" : "button", cloud.label || `group ${cloud.cluster_id}`, hidden ? "" : "plain");
    if (!hidden) {
      link.type = "button";
      link.addEventListener("click", () => {
        if (live && !state.rows.has(cloud.cluster_id)) {
          link.replaceWith(node("span", `${link.textContent} (no longer visible in the live window)`));
        } else {
          selectCloud(cloud.cluster_id);
        }
      });
    }
    const qualifiers = [
      hidden ? (live ? "no longer visible in the live window" : "below display threshold") : null,
      row?.analysis_stale ? "assessment needs refresh" : null,
      row && row.status !== "ok" ? `assessment ${words(row.status)}` : null,
    ].filter(Boolean);
    const where = node("p");
    where.append(link, ` · group concern ${words(cloud.concern_level)}${qualifiers.length ? ` (${qualifiers.join("; ")})` : ""}`);
    group.append(where, node("p", "This is the existing group's concern, not the submitted item's content concern. A match does not establish shared framing.", "small"));
  } else {
    group.append(node("p", "No matching group among the closest posts."));
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
    const response = await fetch(live ? "/api/live/chat" : "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, attachments, history: state.history.slice(-8) }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`;
      throw new Error(detail);
    }
    const summary = node("section", null, "reply-summary");
    summary.append(node("h3", "Summary"), node("p", body.reply));
    pending.replaceChildren(summary, renderVerdict(body));
    state.history.push(
      { role: "user", content: (text || `Attached: ${files.map((file) => file.name).join(", ")}`).slice(0, 4000) },
      { role: "assistant", content: body.reply.slice(0, 4000) },
    );
    if (!live || state.rows.has(body.placement.nearest_cloud?.cluster_id)) {
      map.setMarker({
        label: files[0]?.name || "your text",
        cloudId: body.placement.nearest_cloud?.cluster_id,
      });
    }
  } catch (error) {
    pending.className = "msg assistant error";
    pending.textContent = error.message;
    input.value = [text, input.value].filter(Boolean).join("\n");
    state.files.unshift(...files);
    renderChips();
    resize();
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
  input.disabled = false;
  byId("files").disabled = false;
  input.placeholder = "Ask Nemotron 3.5 Super — paste a post, or attach a file or article";
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

setupChat();
}
