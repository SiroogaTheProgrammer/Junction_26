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
  byId("info").textContent = "live test · streamed dataset · new groups per batch · rolling window · not real-time social traffic";
  byId("arrived").textContent = `${snapshot.processed.toLocaleString()} processed`;
  byId("rate").textContent = `${snapshot.retained.toLocaleString()} / ${snapshot.window_limit.toLocaleString()} posts in window`;
  byId("clock").textContent = `${snapshot.scanned.toLocaleString()} source rows scanned · ${snapshot.total_groups.toLocaleString()} groups in window${snapshot.failed_groups ? ` · ${snapshot.failed_groups} analyses failed` : ""}`;
  byId("ranking").replaceChildren(...rows.slice(0, 6).map((row) => {
    const item = document.createElement("div");
    const score = Object.assign(document.createElement("b"), { textContent: row.review_priority });
    const name = Object.assign(document.createElement("span"), { textContent: row.label || `Group ${row.cluster_id}` });
    const count = Object.assign(document.createElement("i"), { textContent: `${row.size} posts` });
    item.append(score, name, count);
    return item;
  }));
  if (snapshot.status === "starting" || snapshot.status === "idle") {
    notice("Starting dataset stream and embedding model…");
  } else if (snapshot.status === "error") {
    notice(`Live processing stopped: ${snapshot.error}. Fix the cause and restart the server to retry.`);
  } else if (snapshot.status === "complete") {
    if (rows.length) byId("notice").hidden = true;
    else notice("The dataset is exhausted. No groups above the review threshold remain in the window.");
    byId("rate").textContent += " · dataset exhausted";
  } else if (!rows.length) {
    notice(snapshot.processed ? "No groups above the review threshold yet; processing continues." : "Processing the first batch…");
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
