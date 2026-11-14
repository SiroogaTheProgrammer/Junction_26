import { createMap } from "./map.js";
import { createGroupCard } from "./group-card.js";

const byId = (id) => document.getElementById(id);
const TONES = ["none", "low", "medium", "high"];
const RANKED = 6;
const LABEL_AFTER = 1;
const map = createMap(byId("stage"), { labelAfter: LABEL_AFTER, onSelect: (id) => selectCloud(id) });
const selectCloud = createGroupCard(byId("card"), (id) => map.select(id));
map.setInset(0, 230);
let lastRanking = 0;

function notice(message) {
  const box = byId("notice");
  box.replaceChildren(Object.assign(document.createElement("p"), { textContent: message }));
  box.hidden = false;
}

function renderRanking(force = false) {
  const now = performance.now();
  if (!force && now - lastRanking < 250) return;
  lastRanking = now;
  // Sorted as posts arrive: review priority first, then how many posts the cloud has collected so far.
  const active = [...map.clouds.values()].filter((cloud) => cloud.shown && cloud.n >= LABEL_AFTER)
    .sort((a, b) => b.row.review_priority - a.row.review_priority || b.n - a.n).slice(0, RANKED);
  byId("ranking").replaceChildren(...active.map((cloud) => {
    const row = document.createElement("div");
    const score = Object.assign(document.createElement("b"), { textContent: cloud.row.review_priority });
    const name = Object.assign(document.createElement("span"), { textContent: cloud.row.label || `Group ${cloud.id}` });
    const size = Object.assign(document.createElement("i"), { textContent: `${cloud.n} posts` });
    const level = cloud.row.status === "ok" && TONES.includes(cloud.row.concern_level) ? cloud.row.concern_level : "none";
    size.style.setProperty("--label-color", `var(--concern-${level})`);
    row.append(score, name, size);
    return row;
  }));
}

async function start() {
  try {
    const [metaResponse, rowsResponse] = await Promise.all([fetch("/api/meta"), fetch("/api/clusters?include_hidden=true")]);
    if (metaResponse.status === 503) return notice("No processed posts yet. Run python -m detector all --posts 10000, then restart the server.");
    if (!metaResponse.ok || !rowsResponse.ok) throw new Error("the API returned an error");
    const meta = await metaResponse.json();
    if (!meta.map_available) return notice(meta.map_error);
    map.setClouds(await rowsResponse.json());
    byId("info").textContent = `live test · replay through ${location.host} · groups from the offline run · size = posts · colour = concern`;
    byId("rate").textContent = `${meta.live.posts_per_second} posts/s`;

    const stream = new EventSource("/api/live/stream");
    stream.addEventListener("message", (event) => {
      const batch = JSON.parse(event.data);
      byId("notice").hidden = true;
      map.addPoints(batch.points);
      byId("arrived").textContent = `${batch.sent.toLocaleString()} / ${batch.total.toLocaleString()}`;
      byId("clock").textContent = batch.clock ? `post time ${batch.clock.slice(0, 19).replace("T", " ")}` : "";
      renderRanking();
    });
    stream.addEventListener("done", () => {
      stream.close();
      renderRanking(true);
      const again = Object.assign(document.createElement("a"), { textContent: "replay", href: "/live", className: "plain" });
      byId("rate").replaceChildren("replay complete · ", again);
    });
    stream.addEventListener("error", () => {
      // EventSource would reconnect and restart the replay from the first post; stop instead.
      stream.close();
      notice("The stream was interrupted. Reload to restart the replay.");
    });
  } catch (error) {
    notice(`Could not start the live test: ${error.message}. Reload to retry.`);
  }
}
start();
