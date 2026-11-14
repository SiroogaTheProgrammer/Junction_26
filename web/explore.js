import { createMap } from "./map.js";
import { createGroupCard } from "./group-card.js";
import { createChat } from "./chat.js";

const byId = (id) => document.getElementById(id);
const map = createMap(byId("stage"), { onSelect: (id) => selectCloud(id) });
const selectCloud = createGroupCard(byId("card"), (id) => map.select(id), (visible) => {
  byId("thread").hidden = visible;
  byId("chips").hidden = visible;
  if (!visible) byId("thread").scrollTop = byId("thread").scrollHeight;
});

function notice(message, command) {
  const box = byId("notice");
  box.replaceChildren(Object.assign(document.createElement("p"), { textContent: message }));
  if (command) box.append(Object.assign(document.createElement("code"), { textContent: command }));
  box.hidden = false;
}

const docked = matchMedia("(min-width: 1000px)");
const dock = () => map.setInset(docked.matches ? byId("composer").getBoundingClientRect().width + 40 : 0, docked.matches ? 0 : 210);
docked.addEventListener("change", dock);
dock();

async function start() {
  try {
    const [metaResponse, rowsResponse] = await Promise.all([fetch("/api/meta"), fetch("/api/clusters?include_hidden=true")]);
    if (metaResponse.status === 503) return notice("No processed posts yet. Run this, then restart the server:", "python -m detector all --posts 10000");
    if (!metaResponse.ok || !rowsResponse.ok) throw new Error("the API returned an error");
    const meta = await metaResponse.json();
    const rows = await rowsResponse.json();
    createChat({ meta, rows: new Map(rows.map((row) => [row.cluster_id, row])), selectCloud, map });
    map.setClouds(rows, { preloaded: true });
    byId("info").textContent = [
      `${meta.n_posts.toLocaleString()} posts`,
      `${meta.n_clusters - meta.n_discarded} groups`,
      `${meta.n_discarded} dropped at 0/100`,
      "size = posts · colour = concern",
    ].join(" · ");
    if (meta.dataset?.burstiness_reliable === false) byId("info").textContent += " · burst timing excluded (short snapshot)";
    if (meta.n_stale) byId("info").textContent += ` · ${meta.n_stale} model assessments need refreshing`;
    if (rows.every((row) => row.discarded)) notice("No groups above the review threshold. You can still assess material in the chat.");
    else byId("notice").hidden = true;
  } catch (error) {
    notice(`Could not load the map: ${error.message}. Reload to retry.`);
  }
}
start();
