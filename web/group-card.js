const words = (value) => String(value ?? "unavailable").replaceAll("_", " ");
const percent = (value) => (value == null ? "unavailable" : `${Math.round(value * 100)}%`);
const statPercent = (value) => value == null ? "unknown" : `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}

export function createGroupCard(card, select) {
  let request = null;

  function render({ cluster, analysis }) {
    const parsed = analysis.parsed;
    const title = node("h2");
    const dot = node("i", null, "dot");
    const tone = ["none", "low", "medium", "high"].includes(parsed?.concern_level) ? parsed.concern_level : "none";
    dot.style.setProperty("--label-color", `var(--concern-${tone})`);
    title.append(dot, parsed?.label || `Group ${cluster.cluster_id}`);
    const header = node("div", null, "card-header");
    const close = node("button", "×", "plain card-close");
    close.type = "button";
    close.setAttribute("aria-label", "Close group details");
    close.addEventListener("click", () => show(null));
    header.append(title, close);
    card.replaceChildren(header);
    const section = (heading) => {
      const element = node("section", null, "card-section");
      element.append(node("h3", heading));
      card.append(element);
      return element;
    };
    if (cluster.signals?.length) {
      const signals = section("Behaviour signals");
      for (const signal of cluster.signals) signals.append(node("p", signal, "small"));
      signals.append(node("p", "Aggregate indicators for review, not proof of bots or shared control.", "small"));
    }
    const overview = section("Overview");
    if (cluster.analysis_stale) overview.append(node("p", "Saved model assessment predates the current statistics, prompt or sample size. Rerun the analyzer before treating its concern as current.", "small"));
    overview.append(node("p", [
      `${cluster.size.toLocaleString()} posts in the dataset`,
      `review priority ${cluster.review_priority}/100`,
      `concern ${words(parsed?.concern_level || analysis.status)}`,
      `confidence ${percent(parsed?.confidence)}`,
    ].join(" · ")));
    if (parsed) {
      overview.append(node("p", parsed.summary), node("p", parsed.explanation));
      const framing = section("Framing");
      framing.append(node("p", `narrative relevance ${percent(parsed.relevance)} (${words(parsed.relevance_kind)}): ${parsed.relevance_reason}`, "small"));
      const techniques = parsed.techniques.map((item) => words(item.name));
      if (techniques.length) framing.append(node("p", `techniques: ${techniques.join(", ")}`, "small"));
    } else {
      overview.append(node("p", analysis.status === "failed" ? "The analysis of this group failed." : "This group has not been analysed."));
    }
    const spread = section("Spread");
    const stats = cluster.stats || cluster;
    const baseline = cluster.bucket_baseline;
    spread.append(node("p", [
      `${cluster.n_authors.toLocaleString()} authors`,
      stats.dup_authors == null ? null : `${stats.dup_authors} authors of near-duplicates`,
      stats.largest_dup_group == null ? null : `largest repeated group: ${stats.largest_dup_group}`,
    ].filter(Boolean).join(" · "), "small"));
    const baselineCount = baseline?.reference_n_clusters ?? baseline?.n_clusters;
    if (baseline) spread.append(node("p", `Baseline ${baseline.name} · ${baseline.source === "bucket" ? "similar-sized groups" : "global fallback"}${baselineCount == null ? "" : ` · ${baselineCount} groups`}`, "small"));
    const table = node("table", null, "stats-table");
    const head = node("thead");
    const headings = node("tr");
    for (const label of ["Signal", "Value", "p50", "p90"]) {
      const cell = node("th", label);
      cell.scope = "col";
      headings.append(cell);
    }
    head.append(headings);
    table.append(head);
    const body = node("tbody");
    for (const [key, label] of [
      ["dup_ratio", cluster.stats ? "Near-duplicates" : "Repeated text (legacy)"],
      ["exact_dup_ratio", "Exact duplicates"],
      ["authors_per_post", "Authors / post"],
      ["top_author_share", "Top author"],
      ["top5_author_share", "Top 5 authors"],
      ["burstiness", `${cluster.burst_window_min || 10}-minute burst`],
      ["top_domain_share", "Top link domain"],
      ["link_share", "Posts with links"],
      ["top_hashtag_share", "Top hashtag"],
      ["image_share", "Posts with images"],
    ]) {
      if (!(key in stats)) continue;
      const pct = baseline?.pct?.[key];
      const above = pct && stats[key] > pct.p90;
      const row = node("tr", null, above ? "above-baseline" : "");
      row.append(node("th", label), node("td", `${statPercent(stats[key])}${above ? " ↑" : ""}`), node("td", pct ? statPercent(pct.p50) : "—"), node("td", pct ? statPercent(pct.p90) : "—"));
      body.append(row);
    }
    table.append(body);
    spread.append(table);
    if (cluster.burstiness_reliable === false) spread.append(node("p", "Burst timing is unreliable: the dataset spans fewer than six burst windows. Excluded from flags and priority.", "small"));
    if (stats.top_domain) spread.append(node("p", `Top domain: ${stats.top_domain}`, "small"));
    if (stats.top_hashtag) spread.append(node("p", `Top hashtag: #${stats.top_hashtag}`, "small"));
    spread.append(node("p", cluster.behavior_anomaly ? `Above baseline: ${cluster.anomaly_triggered_by.map(words).join(", ")}` : "No above-threshold behaviour flag; this does not establish that activity is organic.", "small"));
    const examples = section("Example posts");
    const posts = (analysis.sample || []).slice(0, 5);
    for (const post of posts) examples.append(node("blockquote", post.text, "post"));
    card.append(node("p", "Group concern describes framing and spread; it does not determine truth or intent. Esc closes.", "small"));
    return { examples, needsExamples: !posts.length };
  }

  async function show(id) {
    request?.abort();
    if (select(id) == null) {
      card.hidden = true;
      return;
    }
    const controller = new AbortController();
    request = controller;
    card.replaceChildren(node("p", "loading…"));
    card.hidden = false;
    try {
      const response = await fetch(`/api/clusters/${id}`, { signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const { examples, needsExamples } = render(await response.json());
      if (needsExamples) {
        examples.append(node("p", "Loading examples…", "small"));
        const page = await fetch(`/api/clusters/${id}/posts?limit=5`, { signal: controller.signal });
        if (!page.ok) throw new Error(`Could not load examples (HTTP ${page.status})`);
        const { posts } = await page.json();
        examples.replaceChildren(node("h3", "Example posts"), ...posts.map((post) => node("blockquote", post.text, "post")));
        if (!posts.length) examples.append(node("p", "No example posts available.", "small"));
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        if (card.querySelector("p")?.textContent === "loading…") card.replaceChildren();
        card.append(node("p", `Could not load group details (${error.message}).`, "small"));
      }
    }
  }

  document.addEventListener("keydown", (event) => { if (event.key === "Escape") show(null); });
  return show;
}
