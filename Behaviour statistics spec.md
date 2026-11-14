# Behaviour statistics spec

Spec for adding the behaviour statistics to the existing pipeline. These are the plain-arithmetic signals of how a cluster spreads (duplicates, dominant accounts, bursts, link and hashtag concentration). The [analyzer spec](https://claude.ai/artifact/AvHNjPiJREQa2wQMeWcNRq) and the dashboard depend on them: the organic cap, the behaviour floor for hidden topics, and the stats shown next to each narrative. Terse on purpose, for the team and for coding agents.

## 1. Goal and principles

- Compute per-cluster spread statistics with **no AI**, so every number can be checked by hand.
- Compare each cluster with the rest of the dataset (a baseline), never with a fixed magic threshold.
- Produce two flags the rest of the system uses: `organic_looking` and `behavior_anomaly`.
- Produce short human-readable signal lines ("41% near-duplicate posts, dataset p90 is 9%") for the LLM prompt and the dashboard.
- Describe accounts only in aggregate. No per-account score, no account labels.

## 2. Where it fits

```
load -> embed -> cluster -> stats -> analyze -> serve
```

- New stage `python -m detector stats`. It reads `data/posts.parquet`, `data/assignments.parquet` and `data/clusters.json`, and **adds** fields to `data/clusters.json` (section 7). It never changes cluster membership.
- `cluster` calls `stats.run()` at its end, so a normal run needs no extra command. The separate command exists so stats can be recomputed on the current clusters without re-clustering, which is how this is added to the existing system.
- `python -m detector all` runs `load, embed, cluster, stats, analyze`. The stats stage is idempotent: rerunning gives the same result.
- Writes are atomic: write `clusters.json.tmp`, then rename.

## 3. Module layout

```
src/detector/stats.py
  run()                                    # stage entry point
  normalize_text(text) -> str
  extract_domains(text) -> list[str]
  extract_hashtags(text) -> list[str]
  near_duplicate_groups(texts) -> list[int]          # group id per post, -1 = no near-duplicate
  densest_window_share(timestamps, minutes) -> float
  cluster_stats(df_cluster) -> ClusterStats
  build_baseline(all_stats) -> Baseline
  baseline_for(size, baseline) -> Percentiles
  organic_looking(stats, pct) -> bool
  behavior_anomaly(stats, pct, size) -> AnomalyResult
  describe_signals(stats, pct) -> list[str]
src/detector/schema.py                     # ClusterStats, Baseline, AnomalyResult (pydantic)
```

Everything except `run()` is a pure function with unit tests. The analyzer and the API import `behavior_anomaly`, `baseline_for` and `describe_signals` from here; nothing is reimplemented elsewhere.

## 4. Text helpers

The loader provides canonical `text`, `created_at`, `author` and `uri`, so links and hashtags come from text. The new `itsmebatuhan/bluesky-10m-posts-15-languages` source supplies `did`/`rkey` instead of author/URI and optional `lang`/`is_reply`; the loader maps them. It does not supply `has_images`: preserve null and report image share as unknown rather than inventing zero.

- **`normalize_text`:** Unicode NFKC, lowercase, links to `<url>`, remove `@mentions`, keep hashtag words without `#`, replace anything that is not a letter, digit or `<url>` with a space, collapse whitespace, strip.
- **Link detection:** match full URLs (`https?://...`) **and** bare domains with a path, because Bluesky clients often shorten links in the text (`example.com/2026/11/...`). Pattern for bare domains: a word made of labels ending in a known top-level domain of 2 to 24 letters, optionally followed by `/...`. Keep a small denylist of false matches (for example file names like `image.png`).
- **`extract_domains`:** lowercase host, strip a leading `www.`, one entry per distinct domain per post.
- **`extract_hashtags`:** `#` followed by Unicode letters, digits or underscores, lowercased, one entry per distinct tag per post.

## 5. Statistics per cluster

Computed for every non-noise cluster, regardless of size.

| Stat | Definition |
| --- | --- |
| `size` | Posts in the cluster |
| `n_authors` | Distinct authors |
| `authors_per_post` | `n_authors / size` |
| `top_author_share` | Share of posts by the most active author |
| `top5_author_share` | Share of posts by the five most active authors |
| `exact_dup_ratio` | Share of posts whose normalised text occurs at least twice in the cluster |
| `dup_ratio` | Share of posts with at least one **near**-duplicate in the cluster (includes exact duplicates). Used for flags |
| `dup_authors` | Distinct authors among posts that have a near-duplicate. Many authors posting the same text is the classic coordination pattern; one author is spam |
| `largest_dup_group` | Size of the biggest near-duplicate group |
| `burstiness` | Share of posts inside the densest `BURST_WINDOW_MIN`-minute sliding window |
| `time_span_min` | Minutes between the first and last post |
| `first_post_at`, `last_post_at` | ISO timestamps |
| `top_domain`, `top_domain_share` | Most common link domain and the share of posts containing it (null and 0 if no links) |
| `link_share` | Share of posts with at least one link |
| `top_hashtag`, `top_hashtag_share` | Most common hashtag and its share (null and 0 if none) |
| `image_share` | Share of posts with `has_images` |

`image_share` is null if image-presence metadata is unavailable for any post in
the group. These aggregate statistics cannot establish client provenance, bot
identity or common control.

**Near-duplicates**

- MinHash with locality-sensitive hashing (`datasketch`, `num_perm = 64`) over the normalised text.
- Shingles: word 3-grams; for texts under 8 words use character 5-grams instead.
- Two posts are near-duplicates when estimated Jaccard similarity is at least `NEAR_DUP_JACCARD` (default 0.8). Groups are the connected components of near-duplicate pairs (union-find).
- Normalised texts shorter than 15 characters only count as exact duplicates; near-duplicate matching on very short texts is meaningless.
- Computed within one cluster at a time, never across the whole dataset.
- LSH is an approximate candidate search, so some qualifying pairs may be missed.
  Connected components do not imply that every pair within a group meets the threshold.

**Burstiness**

- Sort timestamps, two-pointer sliding window of `BURST_WINDOW_MIN` minutes, take the maximum count, divide by `size`.
- `burstiness_reliable` (dataset-level flag): false when the whole dataset spans less than `6 * BURST_WINDOW_MIN` minutes. Then every cluster looks bursty, so burstiness is shown but ignored for both flags, and the report and dashboard say why.

Round every ratio to 4 decimals before storing.

## 6. Baseline and flags

**Baseline**

Small clusters have naturally higher ratios (one author in a 25-post cluster is already 4 percent), so compare clusters of similar size.

- Size buckets: 25 to 49, 50 to 199, 200 and up (`BASELINE_BUCKETS`).
- For each bucket with at least `MIN_CLUSTERS_PER_BUCKET` (default 10) clusters, store p50, p75 and p90 of: `dup_ratio`, `top_author_share`, `top5_author_share`, `burstiness`, `top_domain_share`, `top_hashtag_share`, `authors_per_post`.
- Buckets with too few clusters fall back to the global percentiles over all clusters. `baseline_for(size)` returns the right one and says which.

**`organic_looking`** is true when `dup_ratio`, `top_author_share` and (if reliable) `burstiness` are all at or below their p75. The analyzer uses it to cap concern.

**`behavior_anomaly`** is true when the cluster has at least `MIN_ANALYSIS_SIZE` posts and any of `dup_ratio`, `top_author_share` or (if reliable) `burstiness` is above the percentile `ANOMALY_PERCENTILE` (default p90). It returns `{flag, triggered_by: [stat names]}`. The analyzer and API use it for the behaviour floor.

`top_domain_share` and `top_hashtag_share` are shown and passed to the LLM but do not trigger the flag: news outlets and event hashtags concentrate naturally.

**Signal lines** (`describe_signals`): one short line for each flag-relevant stat above p90, plus `dup_authors` when near-duplicates come from 5 or more authors. Examples: "41% near-duplicate posts (similar clusters: p90 9%)", "near-identical text from 23 different accounts", "62% of posts in one 10-minute window (p90 18%)". Plain numbers, no judgement words.

## 7. Output

`data/clusters.json` gains these fields. Existing fields stay unchanged.

```
{
  "run": {..., "stats_version": "1"},
  "dataset": {"n_posts": 100000, "time_span_min": 812.5, "burstiness_reliable": true},
  "baseline": {
    "buckets": [
      {"min_size": 25, "max_size": 49, "n_clusters": 140, "source": "bucket",
       "pct": {"dup_ratio": {"p50": 0.0, "p75": 0.04, "p90": 0.12}, ...}}
    ],
    "global": {"n_clusters": 310, "pct": {...}}
  },
  "clusters": [
    {"cluster_id": 7, "size": 64, ...,
     "stats": {<all fields from section 5>},
     "baseline_bucket": "50-199",
     "flags": {"organic_looking": false,
               "behavior_anomaly": {"flag": true, "triggered_by": ["dup_ratio"]}},
     "signals": ["41% near-duplicate posts (similar clusters: p90 9%)"]}
  ]
}
```

`data/stats_report.json`: number of clusters, how many are anomalous and organic-looking, counts per trigger, the dataset time span, `burstiness_reliable`, buckets that fell back to global, and seconds per step.

## 8. Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `BURST_WINDOW_MIN` | `10` | Sliding window for burstiness |
| `NEAR_DUP_JACCARD` | `0.8` | Similarity for near-duplicates |
| `MINHASH_PERM` | `64` | MinHash permutations |
| `BASELINE_BUCKETS` | `25,50,200` | Lower bounds of size buckets |
| `MIN_CLUSTERS_PER_BUCKET` | `10` | Below this a bucket uses the global baseline |
| `ANOMALY_PERCENTILE` | `90` | Shared with the analyzer spec |
| `MIN_ANALYSIS_SIZE` | `20` | Shared with the project spec |

Add `datasketch` to `pyproject.toml` with a pinned version.

**Compatibility:** the implementation retains the existing `MIN_ANALYSIS_SIZE=1`
default to preserve standalone labeling behaviour. Set it to 20 for the minimum
anomaly size specified above. `all` continues labeling every cloud regardless.
Legacy flat baselines and exact-duplicate artifacts remain readable until
`stats` is rerun; upgraded artifacts use the versioned nested baseline.

## 9. Integration with the rest of the system

- **Analyzer:** read `stats`, `flags`, `signals` and `baseline_for(size)` from `clusters.json` instead of computing them. The user prompt shows each flag-relevant stat with its bucket's p50 and p90, then the signal lines. `rules.py` imports `behavior_anomaly` from `stats.py`. The rounded stats are part of `input_hash`, so changing stats settings triggers re-analysis of the affected clusters.
- **API:** `/api/clusters` adds `dup_ratio`, `dup_authors`, `top_author_share`, `burstiness`, `flags` and `signals`. `/api/clusters/{id}` adds the full `stats`, the bucket baseline and the bucket name. `/api/meta` adds the `dataset` block.
- **Dashboard:** the detail panel's stats table shows value, p50 and p90 with a mark above p90, and lists the signal lines at the top. The tooltip shows the first signal line if there is one. If `burstiness_reliable` is false, the burstiness row says so.
- **Project spec:** this replaces the stats part of section 7 there; the stats columns in section 4's config table move to section 8 here.
- **Cache freshness:** each group has a deterministic `stats_hash` covering stats,
  selected baseline, flags and reliability. Analysis records retain that hash.
  The API marks an older model assessment stale and keeps that group visible
  until re-analysis; no LLM calls happen in the stats stage itself.

## 10. Tests and checks

**Unit tests (`tests/test_stats.py`)**

- `normalize_text`: URLs, bare domains, mentions, hashtags, emoji, mixed case and Unicode.
- `extract_domains`: full URLs, bare shortened links, `www.`, and the file-name denylist.
- Near-duplicates: identical texts, texts differing in one word, short texts, unrelated texts.
- `densest_window_share`: one post, all posts at the same second, posts spread evenly, a window boundary exactly at 10 minutes.
- Baseline: bucket fallback with too few clusters; `baseline_for` at bucket edges.
- Flags: organic case, spam case (one author), coordination case (many authors, same text), small cluster below `MIN_ANALYSIS_SIZE`, unreliable burstiness.

**Synthetic check (`scripts/inject_synthetic.py`, test only)**

Adds clearly marked fake clusters to a copy of the data: 50 near-identical posts from 20 invented authors within 5 minutes, and one spam cluster from a single author. Run the stats stage on the copy and assert both are flagged. Never mixed into the data the dashboard serves.

**Sanity report (`python -m detector stats --inspect`)**

Prints the top 10 clusters by `dup_ratio`, `top_author_share` and `burstiness`, each with 3 example posts, so a person can check that high numbers match what the posts look like.

## 11. Definition of done

- [x] `python -m detector stats` runs on the existing artifacts and adds the fields from section 7 without changing clusters.
- [x] `cluster` and `all` run the stats stage automatically.
- [x] Unit tests and the isolated synthetic check pass.
- [x] The analyzer consumes versioned stats/baselines and shared flag/signal helpers; no duplicate implementation.
- [x] The dashboard shows the stats table with p50 and p90 and the signal lines.
- [x] `stats_report.json` exists and states whether burstiness is reliable for this dataset.
