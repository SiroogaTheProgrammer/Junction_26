import { createMap } from "./map.js";
import { createGroupCard } from "./group-card.js";
import { createChat } from "./chat.js";

const byId = (id) => document.getElementById(id);
const liveRows = new Map();
let selected = null;
let runningDataset = null;
const map = createMap(byId("stage"), { onSelect: (id) => selectCloud(id) });
const showCard = createGroupCard(
  byId("card"),
  (id) => { selected = map.select(id); return selected; },
  (visible) => {
    byId("thread").hidden = visible;
    byId("chips").hidden = visible;
    if (!visible) byId("thread").scrollTop = byId("thread").scrollHeight;
  },
  "/api/live/clusters",
);
const selectCloud = (id) => showCard(id);
const docked = matchMedia("(min-width: 1000px)");
const narrow = matchMedia("(max-width: 700px)");
const dock = () => map.setInset(
  docked.matches ? byId("composer").getBoundingClientRect().width + 40 : 0,
  docked.matches ? 200 : 320, 0, narrow.matches ? 45 : 0,
);
docked.addEventListener("change", dock);
narrow.addEventListener("change", dock);
dock();

async function setupChat() {
  try {
    const response = await fetch("/api/live/chat/status");
    if (!response.ok) {
      const body = await response.json();
      throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
    }
    const meta = await response.json();
    createChat({ meta, rows: liveRows, selectCloud, map, live: true });
  } catch (error) {
    byId("prompt").placeholder = `Chat unavailable: ${error.message}`;
  }
}
setupChat();

function notice(message) {
  const box = byId("notice");
  box.replaceChildren(Object.assign(document.createElement("p"), { textContent: message }));
  box.hidden = false;
}

function render(snapshot) {
  runningDataset = snapshot.dataset;
  const rows = snapshot.groups;
  liveRows.clear();
  for (const row of rows) liveRows.set(row.cluster_id, row);
  map.setClouds(rows, { preloaded: true });
  if (selected != null && !rows.some((row) => row.cluster_id === selected)) selectCloud(null);
  byId("arrived").textContent = `${snapshot.processed.toLocaleString()} processed`;
  byId("rate").textContent = `${snapshot.retained.toLocaleString()} / ${snapshot.window_limit.toLocaleString()} in window`;
  byId("batch-size").textContent = `batch: ${snapshot.batch_size.toLocaleString()} source records`;
  const batch = snapshot.last_batch;
  const speed = batch?.posts_per_second;
  byId("batch-speed").textContent = !batch ? "speed: awaiting results"
    : speed == null ? "speed unavailable" : `${speed.toFixed(2)} posts/s`;
  if (speed == null) {
    byId("batch-speed").style.removeProperty("--speed-hue");
    byId("batch-speed").classList.remove("speed-coloured");
  } else {
    byId("batch-speed").style.setProperty("--speed-hue", String(Math.max(0, Math.min(speed / 20, 1)) * 120));
    byId("batch-speed").classList.add("speed-coloured");
  }
  byId("dataset").textContent = `Historical dataset: ${snapshot.dataset}`;
  byId("last-batch").textContent = batch
    ? `Last batch: ${batch.posts.toLocaleString()} posts / ${batch.source_rows.toLocaleString()} records · ${batch.grouped_posts.toLocaleString()} grouped · ${batch.seconds.toFixed(1)} s`
    : "Last batch: awaiting results";
  byId("clock").textContent = `${snapshot.scanned.toLocaleString()} source records read (including filtered records)${snapshot.resumed_from ? ` · resumed at ${snapshot.resumed_from.toLocaleString()}` : ""} · ${snapshot.total_groups.toLocaleString()} groups in window${snapshot.hidden_groups ? ` · ${snapshot.hidden_groups} below threshold` : ""}${snapshot.failed_groups ? ` · ${snapshot.failed_groups} analyses failed` : ""}`;
  byId("failures").hidden = !snapshot.failed_groups;
  byId("failures").textContent = snapshot.failed_groups ? `${snapshot.failed_groups} group analyses failed` : "";
  const emptyReason = snapshot.total_groups
    ? "Groups were found, but all are below the review threshold."
    : "No groups found in the current window; posts may be too dissimilar or batches too small.";
  if (snapshot.status === "starting" || snapshot.status === "idle") {
    notice(snapshot.resumed_from ? `Resuming after ${snapshot.resumed_from.toLocaleString()} source rows…` : "Starting dataset stream and embedding model…");
  } else if (snapshot.status === "error") {
    notice(`Live processing stopped: ${snapshot.error}. Fix the cause and restart the server to retry.`);
  } else if (snapshot.status === "complete") {
    if (rows.length) byId("notice").hidden = true;
    else if (snapshot.resumed_from && !snapshot.retained) notice("The dataset is exhausted. The previous rolling window was not saved across the server restart.");
    else notice(`The dataset is exhausted. ${emptyReason}`);
    byId("arrived").textContent += " · dataset exhausted";
  } else if (!rows.length) {
    notice(snapshot.last_batch ? `${emptyReason} Processing continues.` : "Processing the first batch…");
  } else {
    byId("notice").hidden = true;
  }
}

const stream = new EventSource("/api/live/processed");
stream.addEventListener("dataset_changed", () => location.reload());
stream.addEventListener("message", (event) => {
  const snapshot = JSON.parse(event.data);
  render(snapshot);
  if (snapshot.status === "complete" || snapshot.status === "error") stream.close();
});
stream.addEventListener("error", () => {
  stream.close();
  notice("The live connection was interrupted. Reload to reconnect to the current run.");
});
// Completed streams are closed; those viewers still need to notice a source switch.
setInterval(async () => {
  if (!runningDataset) return;
  try {
    const response = await fetch("/api/datasets");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const sources = await response.json();
    const active = sources.find((source) => source.live);
    if (active && active.repo !== runningDataset) location.reload();
  } catch (error) {
    console.warn("Could not check the active Live dataset", error);
  }
}, 5000);
