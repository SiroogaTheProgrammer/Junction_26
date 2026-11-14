import { groupLabel } from "./labels.js";

const TONES = ["none", "low", "medium", "high"];

// Pack final group sizes once, so replay arrivals grow in stable slots.
export function packBubbles(rows) {
  const circles = [];
  const largest = Math.sqrt(Math.max(1, ...rows.map((row) => row.size)));
  const gap = largest * 0.14;
  for (const row of [...rows].sort((a, b) => b.size - a.size || a.cluster_id - b.cluster_id)) {
    const r = Math.sqrt(row.size);
    let x = 0, y = 0;
    for (let step = 0; circles.length; step += 1) {
      const angle = step * 0.15 + row.cluster_id;
      const distance = largest * 0.09 * Math.sqrt(step);
      x = Math.cos(angle) * distance;
      y = Math.sin(angle) * distance;
      if (circles.every((other) => Math.hypot(x - other.x, y - other.y) >= r + other.r + gap)) break;
    }
    circles.push({ id: row.cluster_id, x, y, r });
  }
  return circles;
}

export function createMap(root, { onSelect = () => {}, labelAfter = 1 } = {}) {
  const layer = root.querySelector(".labels");
  const clouds = new Map();
  const marker = document.createElement("p");
  marker.className = "submission-note";
  marker.hidden = true;
  root.append(marker);
  let packed = [];
  let count = 0;
  let selected = null;
  let right = 0;
  let leftInset = 0;
  let topInset = 0;
  let bottom = 0;
  let submission = null;

  function draw() {
    const bounds = root.getBoundingClientRect();
    const width = Math.max(100, bounds.width - leftInset - right - 40);
    const height = Math.max(100, bounds.height - bottom - topInset - 100);
    const left = Math.min(0, ...packed.map((ball) => ball.x - ball.r));
    const top = Math.min(0, ...packed.map((ball) => ball.y - ball.r));
    const spanX = Math.max(1, ...packed.map((ball) => ball.x + ball.r)) - left;
    const spanY = Math.max(1, ...packed.map((ball) => ball.y + ball.r)) - top;
    const scale = Math.min(width / spanX, height / spanY);
    const offsetX = leftInset + 20 + (width - spanX * scale) / 2 - left * scale;
    const offsetY = 60 + topInset + (height - spanY * scale) / 2 - top * scale;
    for (const ball of packed) {
      const cloud = clouds.get(ball.id);
      const radius = Math.sqrt(cloud.n) * scale;
      const el = cloud.el;
      el.hidden = cloud.n < labelAfter;
      el.style.width = `${radius * 2}px`;
      el.style.height = `${radius * 2}px`;
      el.style.left = `${offsetX + ball.x * scale}px`;
      el.style.top = `${offsetY + ball.y * scale}px`;
      el.classList.toggle("compact", radius < 42);
      el.classList.toggle("selected", cloud.id === selected);
      el.classList.toggle("matched", cloud.id === submission?.cloudId);
      el.classList.toggle("dimmed", selected != null && cloud.id !== selected);
      const level = cloud.row.status === "ok" ? cloud.row.concern_level : cloud.row.status;
      const description = `${groupLabel(cloud.row.label, cloud.id)}, ${cloud.n.toLocaleString()} ${cloud.n === 1 ? "post" : "posts"}, concern ${level}`;
      el.setAttribute("aria-label", description);
      el.title = `${description}${cloud.row.signals?.[0] ? `\n${cloud.row.signals[0]}` : ""}`;
      el.querySelector(".bubble-count").textContent = cloud.n.toLocaleString();
    }
    layer.dataset.groups = packed.filter((ball) => clouds.get(ball.id).n >= labelAfter).length;
    layer.dataset.posts = count;
  }

  root.addEventListener("click", (event) => {
    if (event.target === root || event.target === layer) onSelect(null);
  });
  new ResizeObserver(draw).observe(root);

  return {
    clouds,
    setClouds(rows, { preloaded = false } = {}) {
      const incoming = new Set(rows.map((row) => row.cluster_id));
      for (const [id, cloud] of clouds) {
        if (!incoming.has(id)) {
          cloud.el?.remove();
          clouds.delete(id);
        }
      }
      count = 0;
      for (const row of rows) {
        let cloud = clouds.get(row.cluster_id);
        if (!cloud) {
          cloud = { id: row.cluster_id, row, shown: !row.discarded, n: 0, el: null };
          clouds.set(cloud.id, cloud);
        }
        cloud.row = row;
        cloud.shown = !row.discarded;
        if (preloaded) cloud.n = row.size;
        count += cloud.n;
        if (cloud.shown && !cloud.el) {
          const el = document.createElement("button");
          el.type = "button";
          el.className = "group-bubble";
          el.hidden = true;
          const name = document.createElement("span");
          name.className = "bubble-name";
          const size = document.createElement("strong");
          size.className = "bubble-count";
          el.append(name, size);
          el.addEventListener("click", () => onSelect(cloud.id));
          layer.append(el);
          cloud.el = el;
        } else if (!cloud.shown && cloud.el) {
          cloud.el.remove();
          cloud.el = null;
        }
        if (cloud.el) {
          const level = row.status === "ok" && TONES.includes(row.concern_level) ? row.concern_level : "none";
          cloud.el.style.setProperty("--label-color", `var(--concern-${level})`);
          cloud.el.querySelector(".bubble-name").textContent = groupLabel(row.label, row.cluster_id);
        }
      }
      if (submission?.cloudId != null && !incoming.has(submission.cloudId)) {
        marker.textContent = `${submission.label} · matched group no longer visible`;
        submission = null;
      }
      packed = packBubbles(rows.filter((row) => !row.discarded));
      draw();
    },
    addPoints(points) {
      for (const point of points) {
        const cloud = clouds.get(point[1]);
        if (cloud) cloud.n += 1;
        count += 1;
      }
      draw();
    },
    setInset(nextRight, nextBottom = 0, nextLeft = 0, nextTop = 0) {
      right = nextRight;
      bottom = nextBottom;
      leftInset = nextLeft;
      topInset = nextTop;
      draw();
    },
    select(id) {
      selected = clouds.get(id)?.shown ? id : null;
      draw();
      return selected;
    },
    setMarker(next) {
      submission = next;
      marker.hidden = !next;
      if (next) {
        const cloud = clouds.get(next.cloudId);
        marker.textContent = cloud
          ? `${next.label} · closest group: ${cloud.row.label || cloud.id}${cloud.shown ? "" : " (hidden, below review threshold)"}`
          : `${next.label} · no matching group`;
      }
      draw();
    },
    get count() { return count; },
  };
}
