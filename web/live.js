import { createMap } from "./map.js";
import { createGroupCard } from "./group-card.js";

const byId = (id) => document.getElementById(id);
let selected = null;
const map = createMap(byId("stage"), { onSelect: (id) => selectCloud(id) });
const showCard = createGroupCard(
  byId("card"),
  (id) => { selected = map.select(id); return selected; },
  () => {},
  "/api/live/clusters",
);
const selectCloud = (id) => showCard(id);
map.setInset(0, 230);

function notice(message) {
  const box = byId("notice");
  box.replaceChildren(Object.assign(document.createElement("p"), { textContent: message }));
  box.hidden = false;
}

function render(snapshot) {
  const rows = snapshot.groups;
  map.setClouds(rows, { preloaded: true });
  if (selected != null && !rows.some((row) => row.cluster_id === selected)) selectCloud(null);
  byId("info").textContent = `live test · ${snapshot.dataset} · new groups per batch · not real-time social traffic`;
  byId("arrived").textContent = `${snapshot.processed.toLocaleString()} processed`;
  byId("rate").textContent = `${snapshot.retained.toLocaleString()} / ${snapshot.window_limit.toLocaleString()} posts in window`;
  byId("batch-size").textContent = `batch size: ${snapshot.batch_size.toLocaleString()} source rows`;
  const batch = snapshot.last_batch;
  const speed = batch?.posts_per_second;
  byId("batch-speed").textContent = batch
    ? `last batch: ${batch.posts.toLocaleString()} posts / ${batch.source_rows.toLocaleString()} rows · ${batch.grouped_posts.toLocaleString()} grouped · ${batch.seconds.toFixed(1)} s · ${speed == null ? "speed unavailable" : `${speed.toFixed(2)} posts/s`}`
    : "last batch: awaiting results";
  byId("clock").textContent = `${snapshot.scanned.toLocaleString()} source rows scanned${snapshot.resumed_from ? ` · resumed at ${snapshot.resumed_from.toLocaleString()}` : ""} · ${snapshot.total_groups.toLocaleString()} groups in window${snapshot.hidden_groups ? ` · ${snapshot.hidden_groups} below threshold` : ""}${snapshot.failed_groups ? ` · ${snapshot.failed_groups} analyses failed` : ""}`;
  const emptyReason = snapshot.total_groups
    ? "Groups were found, but all are below the review threshold."
    : "No groups found in the current window; posts may be too dissimilar or batches too small.";
  byId("ranking").replaceChildren(...rows.slice(0, 6).map((row) => {
    const item = document.createElement("div");
    const score = Object.assign(document.createElement("b"), { textContent: row.review_priority });
    const name = Object.assign(document.createElement("span"), { textContent: row.label || `Group ${row.cluster_id}` });
    const count = Object.assign(document.createElement("i"), { textContent: `${row.size} posts` });
    item.append(score, name, count);
    return item;
  }));
  if (snapshot.status === "starting" || snapshot.status === "idle") {
    notice(snapshot.resumed_from ? `Resuming after ${snapshot.resumed_from.toLocaleString()} source rows…` : "Starting dataset stream and embedding model…");
  } else if (snapshot.status === "error") {
    notice(`Live processing stopped: ${snapshot.error}. Fix the cause and restart the server to retry.`);
  } else if (snapshot.status === "complete") {
    if (rows.length) byId("notice").hidden = true;
    else if (snapshot.resumed_from && !snapshot.retained) notice("The dataset is exhausted. The previous rolling window was not saved across the server restart.");
    else notice(`The dataset is exhausted. ${emptyReason}`);
    byId("rate").textContent += " · dataset exhausted";
  } else if (!rows.length) {
    notice(snapshot.last_batch ? `${emptyReason} Processing continues.` : "Processing the first batch…");
  } else {
    byId("notice").hidden = true;
  }
}

const stream = new EventSource("/api/live/processed");
stream.addEventListener("message", (event) => {
  const snapshot = JSON.parse(event.data);
  render(snapshot);
  if (snapshot.status === "complete" || snapshot.status === "error") stream.close();
});
stream.addEventListener("error", () => {
  stream.close();
  notice("The live connection was interrupted. Reload to reconnect to the current run.");
});
