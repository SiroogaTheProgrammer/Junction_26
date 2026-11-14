const PAGE = 25;
const byId = (id) => document.getElementById(id);
const TEXT_FIELDS = ["text", "texto"];
let datasets = [];
let current = null;
let offset = 0;
let total = 0;
let request = 0;
let runPending = false;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderTabs() {
  const nav = byId("datasets");
  nav.replaceChildren(...datasets.map((item) => {
    const button = el("button", "plain", item.live ? `${item.name} · live` : item.name);
    button.type = "button";
    button.disabled = runPending;
    button.setAttribute("aria-pressed", String(item.id === current?.id));
    button.addEventListener("click", () => select(item.id, 0));
    return button;
  }));
}

function renderRow(row, index) {
  const item = el("li", "row");
  const field = TEXT_FIELDS.find((name) => name in row);
  item.append(el("p", "row-index small", `#${(offset + index + 1).toLocaleString()}`));
  if (field) item.append(el("p", "row-text", String(row[field] ?? "")));
  const meta = el("dl", "row-meta small");
  for (const [key, value] of Object.entries(row)) {
    if (key === field) continue;
    meta.append(el("dt", "", key), el("dd", "", value === null ? "—" : String(value)));
  }
  item.append(meta);
  return item;
}

async function load() {
  const id = ++request;
  byId("range").textContent = "loading…";
  for (const name of ["prev", "next", "random"]) byId(name).disabled = true;
  try {
    const response = await fetch(`/api/datasets/${current.id}/rows?offset=${offset}&length=${PAGE}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    if (id !== request) return;
    total = body.total;
    byId("rows").replaceChildren(...body.rows.map(renderRow));
    const end = offset + body.rows.length;
    byId("range").textContent = `rows ${(offset + 1).toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()}`;
    byId("prev").disabled = offset === 0;
    byId("next").disabled = end >= total;
    byId("random").disabled = false;
  } catch (error) {
    if (id !== request) return;
    byId("rows").replaceChildren();
    byId("range").textContent = `Could not load rows (${error.message}).`;
    byId("random").disabled = false;
  }
}

function select(id, start) {
  current = datasets.find((item) => item.id === id) ?? datasets[0];
  offset = Math.max(0, start);
  byId("about").textContent = `${current.about} Source: ${current.repo}`;
  history.replaceState(null, "", `?dataset=${current.id}&offset=${offset}`);
  renderTabs();
  byId("run").disabled = runPending;
  load();
}

function go(next) {
  offset = Math.max(0, next);
  history.replaceState(null, "", `?dataset=${current.id}&offset=${offset}`);
  load();
  window.scrollTo({ top: 0 });
}

byId("prev").addEventListener("click", () => go(offset - PAGE));
byId("next").addEventListener("click", () => go(offset + PAGE));
byId("random").addEventListener("click", () => go(Math.floor(Math.random() * Math.max(1, total - PAGE))));
byId("run").addEventListener("click", async () => {
  if (runPending) return;
  const chosen = current;
  if (!chosen.live && !window.confirm(
    `Run ${chosen.name}? This switches Live for everyone, clears the current window and resumes this dataset's saved progress. Chat will use the new window.`,
  )) return;
  runPending = true;
  renderTabs();
  byId("run").disabled = true;
  byId("run-status").textContent = "Starting live analysis…";
  try {
    const response = await fetch(`/api/datasets/${chosen.id}/run`, { method: "POST" });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    location.assign(body.url);
  } catch (error) {
    byId("run-status").textContent = `Could not start live analysis (${error.message}).`;
    runPending = false;
    renderTabs();
    byId("run").disabled = false;
  }
});

try {
  const response = await fetch("/api/datasets");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  datasets = await response.json();
  const params = new URLSearchParams(location.search);
  select(params.get("dataset"), Number.parseInt(params.get("offset") ?? "0", 10) || 0);
} catch (error) {
  byId("range").textContent = `Could not reach the server (${error.message}).`;
}
