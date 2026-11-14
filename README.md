# Online narrative mapping

Randomly sample **10,000** posts from the configured Bluesky snapshot, embed them
with multilingual E5 (384 dimensions), group them with UMAP (5D) + HDBSCAN, then
ask **Nemotron Lightning on Verda** to label and assess every detected cloud from
a sample of its posts and its spread statistics. Ungrouped posts stay as noise.

The default dataset is
[`itsmebatuhan/bluesky-10m-posts-15-languages`](https://huggingface.co/datasets/itsmebatuhan/bluesky-10m-posts-15-languages):
10.1 million posts, 15 languages, collected August 9-12, 2026. Its published
**CC BY-NC-SA 4.0** licence requires attribution, non-commercial use and
share-alike terms; obtain appropriate permission before commercial hosting.
The loader maps `did` to author and combines `did` + `rkey` into a post URI.
Language and reply metadata are retained; image metadata is **unknown** because
this dataset does not supply it. The older `author`/`uri` dataset remains supported.

The website has three pages:

- **`/`** explains the demo and links to the two modes.
- **`/explore`** shows groups from the preloaded 10,000-post run as balls, with a
  rounded chat box for **Nemotron 3.5 Super VL**. Send a post, a file or an
  article: Super assesses the original content, and a separate judge reviews its
  relationship to the closest group. Select a ball for divided detail sections
  and up to five example posts.
- **`/live`** streams the full configured dataset in batches, embeds the posts,
  forms new groups and calls the configured LLM to assess them. This simulates
  incoming traffic; the dataset is a finite historical snapshot, not a live feed.

Ball **area represents post count**; **colour represents group concern**. Ball
positions are only a non-overlapping layout, not semantic distance. Groups that
score **0/100** and ungrouped noise are not drawn. The complete dataset is still
available for matching; failed and pending groups remain visible.

## Open the processed 10,000-post run

```bash
cd ~/Junction_26
source .venv/bin/activate
python -m detector serve --data-dir data/clouds-10000 --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. No Qdrant is needed to view it. The chat needs the
Super VL settings below; the first message takes a few extra seconds while the
embedding model loads.

## Settings (`.env`, never committed)

| Setting | Purpose |
| --- | --- |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Nemotron Lightning: labels groups and judges contextual spread |
| `SUPER_LLM_API_KEY`, `SUPER_LLM_BASE_URL`, `SUPER_LLM_MODEL` | Nemotron 3.5 Super VL: assesses original content including images, reads attachments, writes answers |
| `CHAT_JUDGE` | `lightning` (default) or `super`: model for the separate contextual spread verdict |
| `SAMPLE_SIZE` | Posts sampled per group for analysis (default 50, or all posts in a smaller group); an explicit `.env` value overrides the default |
| `HF_DATASET` | Defaults to `itsmebatuhan/bluesky-10m-posts-15-languages`; `--dataset` overrides an older `.env` value |
| `MIN_REVIEW_PRIORITY` | Current completed groups below this score are dropped (default 1; stale assessments remain visible) |
| `LIVE_DATASET` | Dataset streamed in the live test; defaults to the 15-language Bluesky dataset |
| `LIVE_DATASET_REVISION` | Dataset revision to resolve on the first run (default `main`); its commit is pinned in the checkpoint |
| `LIVE_BATCH_SIZE`, `LIVE_WINDOW_POSTS` | Live batch size (500) and maximum rolling window (25,000); batch size must not exceed the window |
| `CHAT_JUDGE_TIMEOUT_S` | Time budget for the contextual judge, default 60 seconds; an existing `.env` value overrides it |
| `CHAT_ASSESSMENT_MAX_TOKENS` | Maximum output-token budget for structured content and spread assessments (default 32,768; set lower if the endpoint requires it) |

`.env.example` lists every setting with its default. Keep a warm Verda replica for
both models in production: if the Lightning judge does not answer within
`CHAT_JUDGE_TIMEOUT_S` (60 s by default), the chat still places the item and answers, but says
that the **spread review** is unavailable. The independent Super content assessment
is retained. It never switches judge on its own.

## Live processing and storage

Opening `/live` starts one shared processor **per server process**; it continues
when viewers disconnect, until the finite source is exhausted or the server
stops. Reloading reconnects to that run. Use one Uvicorn worker, protect the site
with authentication and rate limits, and expect LLM and CPU/GPU usage: opening
the page can start a long processing job. On error, fix the configuration and
restart the server; processing resumes after the last completed batch.

The small `live_progress.json` checkpoint and its lock file live in `DATA_DIR`,
which must be persistent across container restarts. The checkpoint pins the
dataset commit, even if `main` changes later. A different `LIVE_DATASET` or
`LIVE_DATASET_REVISION` requires a separate data folder. A batch interrupted
before its checkpoint can be processed again. Existing installations have no
saved cursor, so the first run after this update starts at the beginning once;
the previous in-memory position cannot be recovered. The source is a finite
historical dataset: after exhaustion it stays complete and does not restart.
The rolling window is not persisted, so its groups disappear on server restart,
including when restarting after the source has been exhausted.

Each batch is embedded and grouped independently using HDBSCAN on normalized
embeddings. Lightning assesses each group with the same prompt and rules as the
offline analyzer. Groups from different batches are **not merged**; events split
across batches can be missed. Group scores and baselines are batch-local; burst
timing is excluded because these batches do not establish a reliable dataset-wide
time baseline. Duplicate identifiers are removed within each batch, not across
the entire stream. The UI shows at most 60 highest-priority groups; it discloses
failed analyses. It does not claim that detected patterns prove coordination.
The existing `/api/live/stream` remains a legacy offline replay API; the new
page uses `/api/live/processed`.

The live processor retains only results and samples belonging to at most 25,000
recent eligible posts, evicting whole old batches. It does not write live post
vectors, samples or analyses to disk or alter the offline dataset/Qdrant.
Dataset streaming and model libraries can still maintain their own Hugging Face
metadata/model caches and logs: monitor those separately, and do not delete
existing files without checking what uses them. Restart loses the live window.

## Run the whole pipeline

Python 3.11+ and Qdrant are required.

```bash
docker compose up -d qdrant
LLM_WARMUP_TIMEOUT_S=120 LLM_TIMEOUT_S=60 make llm-check

# New dataset, separate artifacts/collection; labels every cloud.
MIN_ANALYSIS_SIZE=20 python -m detector all --posts 10000 \
  --dataset itsmebatuhan/bluesky-10m-posts-15-languages \
  --data-dir data/bluesky-15lang --collection bluesky_15lang &&
python -m detector serve --data-dir data/bluesky-15lang
```

Sampling is reproducible with `SEED=42`; change `SEED` for a new sample. The
normal loader downloads/scans the full multi-GB source, cleans it in Arrow
batches, deduplicates URIs, and uniformly samples eligible unique records **before**
converting the selected text to pandas. This initial preparation can be slow.
The dataset is a snapshot, not a live feed. Keep a data folder and its Qdrant
collection together. Existing `data/clouds-10000` artifacts are not replaced.

For a bounded connectivity/schema test only (not a representative global sample):

```bash
python -m detector load --dataset itsmebatuhan/bluesky-10m-posts-15-languages \
  --posts 50 --scan-limit 150 --data-dir data/bluesky-15lang-smoke
```

`load_report.json` records the sampling method, scanned-prefix limit, counts,
languages and metadata availability. Changed post text or metadata invalidates
derived embeddings, projections, stats and analyses as well as changed IDs.

Useful single steps:

```bash
# Upgrade spread statistics without Qdrant, embeddings, reclustering or LLM calls.
python -m detector stats --data-dir data/clouds-10000 --inspect

# Retry failed or missing cloud analyses; finished ones are cached. Needs Verda, not Qdrant.
python -m detector analyze --data-dir data/clouds-10000 --all-clusters

# Rebuild the map and chat-placement files (projection.parquet, projection_model.npz,
# embeddings.npy) without changing clouds or analyses. Needs Qdrant.
python -m detector project --data-dir data/clouds-10000 --collection clouds_10000
```

Restart the server after any of these.

Existing group labels do not change automatically. To refresh them using up to
50 sampled posts per group, set `SAMPLE_SIZE=50` in the `.env` used by the
analyzer, then run:

```bash
python -m detector analyze --data-dir data/clouds-10000 --all-clusters
```

Use the actual served data directory and restart the server afterward. This
sends requests to the configured LLM and may take time and incur cost. Older
assessments remain visible but are marked stale until refreshed.

## Behaviour and swarm indicators

The [behaviour-statistics spec](./Behaviour%20statistics%20spec.md) is implemented
as a separate, idempotent `stats` stage, also called automatically by `cluster`
and therefore `all`. It never changes group membership.

- Exact duplicates and MinHash/LSH near-duplicate groups (64 permutations,
  estimated Jaccard >= 0.8) are counted separately. Multi-account repetition
  (`dup_authors`, largest duplicate group) distinguishes a potential swarm signal
  from repetition dominated by one account. LSH is approximate and can miss pairs;
  connected groups may contain pairs that are not directly similar.
- Author diversity, top-five author share, links, domains, hashtags, images
  (when known) and first/last timestamps are retained as aggregate context.
- Compare against size buckets 25-49, 50-199 and 200+; buckets with fewer than
  10 groups fall back to the global baseline, explicitly disclosed.
- Burstiness uses a configurable sliding window (default 10 minutes). When the
  **whole sample** spans less than six windows, timing is displayed but excluded
  from anomaly flags, organic caps and review-priority points.
- The details card shows signal lines, value/p50/p90 comparisons, above-p90
  marks and examples. Missing image metadata is shown as unknown.
- `stats_report.json` records flags, triggers, fallback buckets, reliability and
  runtime. New stats hashes mark saved model assessments stale until re-analysis;
  stale groups remain visible. Analyzer prompt v5 and cache inputs include the
  selected baseline, aggregate stats, reliability and signals.

Configure `BURST_WINDOW_MIN`, `NEAR_DUP_JACCARD`, `MINHASH_PERM`,
`BASELINE_BUCKETS`, `MIN_CLUSTERS_PER_BUCKET` and `ANOMALY_PERCENTILE` in `.env`.
`MIN_ANALYSIS_SIZE` remains **1** by default for backward-compatible standalone
analysis; set it to **20**, as in the full-run example, for the spec's minimum
anomaly size. The `all` pipeline still labels every group.

These are **indicators, not proof of bots, automation, common control or malicious
intent**. The snapshot has no client/device provenance or account-history graph.
Random sampling can miss coordinated events; synthetic positives are regression
checks, not a measured real-world detection rate. Languages outside the model's
documented coverage need particular human review.

Tests: `python -m pytest -q`. To inject explicitly marked synthetic swarm/spam
groups into a separate test copy:

```bash
python -m scripts.inject_synthetic --source data/clouds-10000 \
  --output data/synthetic-stats-check
```

The output must be empty and separate from the served dataset. If injected
signals do not exceed that source's baseline, the checker fails explicitly;
it never weakens the baseline to force a positive.

## How the chat works

1. Attachments are read in memory: text, Markdown, CSV, JSON, HTML, PDF and images
   (up to 4 files of 4 MB). Nothing is stored.
2. A short message is placed as written. For longer text or attachments, Super
   writes a short digest of the material, and that digest is placed.
3. The text is embedded with the same model as the posts. Its closest group is
   the one holding most of its 25 nearest posts. Display positions play no role.
4. Super VL assesses the **original message, attachment text and images**, not
   just the digest: content category, content concern, techniques, evidence and
   confidence. Categories include propaganda-style persuasion, ordinary political
   persuasion, commercial persuasion, informational, personal expression and unclear.
   Fear plus blame and electoral mobilisation can be classified as propaganda
   without deciding factual truth or hidden intent. Viewpoint alone is not evidence;
   quoting propaganda critically is not endorsing it.
5. The configured judge separately reviews **contextual spread**, including
   whether the item fits the group. One item cannot show coordination, so only
   this spread verdict is capped without group evidence. It never lowers the
   content assessment.
6. Super answers from both assessments, leading with content and concrete evidence.
   Structured assessments request thinking disabled and share one time budget per
   assessment. Truncation can increase the output budget from 2,048 up to
   `CHAT_ASSESSMENT_MAX_TOKENS` (32,768 by default) over at most five attempts;
   other invalid responses allow at most three attempts. Truncated JSON is never
   accepted. A timeout, unsupported output budget, or provider ignoring the
   reasoning setting can still make an assessment unavailable; it is explicitly
   reported, never invented.

The public endpoint is guarded: 20 chat requests per minute in total, 2 at a time,
and a body size limit (also set in the `Caddyfile`).

## What the scores mean

- **Review priority (0-100)** is a triage heuristic, not a probability of
  propaganda. Final concern adds 0/15/39/60 points; unusual spread (repeated text,
  dominant authors, bursts) adds up to 40.
- **Concern** describes framing and spread. It is not a verdict on truth or on
  people. Human calibration is still needed before treating rankings as validated.
- A failed or unanalysed cloud is shown as such and is never dropped as insignificant.

Full contracts: [project spec](./Coordinated-behavior%20detector%20project%20spec.md),
[analyzer spec](./Analyzer%20spec%20narrative%20analysis%20stage.md),
[UI spec](./UI%20spec%20online%20narrative%20mapping%20and%20analysis.md).
