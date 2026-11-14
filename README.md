# Online narrative mapping

This demo helps people explore patterns in public posts and ask AI to assess
their own text, images and documents. It **simulates** incoming posts using a
historical dataset; it is not connected to a live social-media feed.

## Find your way

- [Use the website](#use-the-website)
- [Understand the results](#understand-the-results)
- [Browse or switch datasets](#browse-or-switch-datasets)
- [Start it on your computer](#start-it-on-your-computer)
- [Keep the VPS disk from filling up](#keeping-the-vps-disk-from-filling-up)
- [Troubleshooting](#troubleshooting)
- [Technical details and settings](#technical-details-and-settings)

## Use the website

1. Open the website and select **Open live analysis**. The first batch can take
   time to download, process and assess.
2. Select a ball to see the group's concern level, explanation and example
   posts. Close the card or focus the chat box to return to the conversation.
3. Type or paste a message in the box on the right. To include a document or
   image, use the **+** control. Press **Enter** to send; Shift+Enter adds a line.
4. Read the separate content and spread assessments. A group match is context,
   not a verdict on your submission. The AI can make mistakes; check the cited
   evidence and review important results yourself.

The top shows posts in the current window and the last batch's speed. Speed
colour ranges from red at 0 to green at 20 posts per second; it is a visual
scale, not a performance target. The bottom shows processed posts and batch
size. Open **Details** for more information.
**Source records read** includes records that were later filtered out; it is
not the same as the number of processed posts.

## Understand the results

- **A ball is a group of similar posts.** Its size shows the number of posts;
  its colour shows the group's concern level. Its position on screen has no
  meaning. Some groups are hidden when their review score is below the display
  threshold, while failed assessments stay visible.
- **The rolling window** normally holds up to 50,000 eligible posts. New batches
  replace the oldest batches. The chat compares your submission with posts in
  this same window, including posts that have no group.
- **Two assessments serve different purposes.** Nemotron Super evaluates the
  submitted material; a separate judge reviews its relationship to nearby
  posts. A low-concern group does not cancel a concerning content assessment.
- **A signal is not proof.** Repeated text can be harmless, political discussion
  is not automatically propaganda, and neither a match nor an AI assessment
  proves coordination, a bot, falsehood or someone's intent.

Behind the scenes, the app reads posts in batches, turns their text into
numerical representations, groups similar posts and asks Nemotron Lightning to
review samples. It also checks for exact repeated text across batches and
multiple accounts. These overlapping candidates can be missed or be benign;
they require human review. For chat, Super reads the submitted material and
its content assessment is kept separate from the spread review.

## Browse or switch datasets

Select **Browse datasets** on the menu to read the raw posts, 25 at a time.
Use **next**, **previous** or **random page**. The dataset used by live
analysis is marked **live**. Rows come straight from Hugging Face, so browsing
needs internet access and shows the dataset's latest version. Two datasets are
available:

- **Bluesky posts in 15 languages** (`itsmebatuhan/bluesky-10m-posts-15-languages`),
  the default for live analysis.
- **US Trump 2020 social media** (`falbanese/US_Trump_2020_social_media`):
  about 2.3 million Twitter posts from May 2020, MIT licence. It has only
  numeric author and target IDs, a date and the text: no post links, language
  or image information, and the timezone is not stated (times are read as UTC).
  Posts are identified by their position in the dataset.

To run live analysis on the Trump 2020 dataset, an administrator sets these
values in `.env` and restarts the app. Use a **new, empty data folder** so the
Bluesky progress is kept:

```bash
LIVE_DATASET=falbanese/US_Trump_2020_social_media
LIVE_DATASET_REVISION=main
DATA_DIR=/app/data/trump-2020   # or data/trump-2020 when running without Docker
```

To return to Bluesky, restore the previous three values. In Docker, the
Compose file's `DATA_DIR` setting overrides `.env`, so change it there. The
app refuses to start live processing if the dataset does not match the folder's
saved progress.

## Start it on your computer

If someone has already set up the app, open a terminal in the project folder
and run:

```bash
source .venv/bin/activate
python -m detector serve --data-dir data --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser and choose **Open live
analysis**. Leave the terminal running while you use the website. You do not
need to run the old 10,000-post pipeline or Qdrant for live analysis.

For a first setup, an administrator needs Python 3.11, internet access and
working Verda model endpoints. In the project folder, run `make setup`, then
fill in the Lightning and Super endpoint addresses, model names and keys in
the `.env` file. The example settings are in [.env.example](./.env.example).
Keep `.env` private and never share or commit API keys. Start the server with
the commands above. Processing sends requests to Verda and may incur costs.

### If the app is already running on a VPS

Ask the server administrator to update the project files and rebuild the app
container from the project directory:

```bash
docker compose up -d --build app
docker compose logs --tail=100 app
```

Rebuilding restarts the app and clears its in-memory window, but keeps the
saved cursor when the existing data volume is preserved. Keep the VPS's
existing Traefik routing and network configuration. Put the Traefik labels in
`docker-compose.override.yml` (copy
[docker-compose.override.example.yml](./docker-compose.override.example.yml)),
not in `docker-compose.yml`: the override file is not tracked by git, so
`git pull` cannot remove it. Without these labels Traefik answers
**404 page not found**. Protect public
access with authentication and appropriate rate limits before exposing it.

### Keeping the VPS disk from filling up

The app cleans up after itself: when it starts, and then every 6 hours, it
deletes everything the live analysis does not need. It keeps only:

- the small saved position (`live_progress.json`) in the data folder;
- the downloaded embedding model and the current live dataset's small
  description files.

It deletes old 10,000-post results and other files the app created in the data
folder, older dataset downloads in the Hugging Face cache, and download logs
older than a day. Files it did not create, such as `.txt` notes, are left alone.
The server log lists what was deleted. Set `DATA_DIR=/app/data` (the whole data
volume) so old run folders inside it are included.

Docker keeps old images and its own logs. Once, after updating, run this in the
project folder on the VPS:

```bash
make vps-clean          # removes the unused Qdrant container, old images and build cache
sudo rm -rf qdrant_storage   # old vector database; live analysis does not use it
df -h /                 # check free space
```

`make deploy` prunes old images automatically, and the app's Docker logs are
capped at 30 MB. To preview the app's cleanup without deleting anything, run
`docker compose exec app python -m detector cleanup --dry-run`. To turn
automatic cleanup off, set `STORAGE_CLEANUP=false`.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| The page is waiting for the first batch | Allow time for the dataset and embedding model to load. Check the server terminal if processing stops. |
| Chat says no posts are available | Wait for a batch to finish. Chat needs posts in the current in-memory window. |
| A model or spread review is unavailable | Check that the corresponding Verda endpoint and key are configured and the deployment is ready. A successful content assessment remains separate. |
| The stream stopped or the connection was interrupted | Read the displayed reason, correct the configuration, and restart the server; reload to reconnect. |
| The site shows only `404 page not found` | Traefik has no route to the app: check that `docker-compose.override.yml` with the Traefik labels exists, then run `docker compose up -d app`. |
| The dataset is exhausted after a restart | The saved cursor reached the end and the old window was not saved. Consult an administrator before starting a separate run; do not delete stored data blindly. |

The processor continues when viewers leave. Restarting clears the current
window but resumes from the saved source cursor; an interrupted batch can be
processed again. The dataset is finite, and an exhausted run does not
automatically restart. Do not send sensitive material to the hosted models
unless you are authorized to do so.

## Technical details and settings

<details>
<summary>For administrators and developers</summary>

The default source is
[`itsmebatuhan/bluesky-10m-posts-15-languages`](https://huggingface.co/datasets/itsmebatuhan/bluesky-10m-posts-15-languages),
a historical snapshot of 10.1 million posts in 15 languages. Its **CC BY-NC-SA
4.0** licence requires attribution, non-commercial use and share-alike terms;
obtain appropriate permission before commercial hosting. The loader maps `did`
to author and combines `did` and `rkey` into a post URI; image metadata is
unknown for this dataset. The older `author`/`uri` dataset remains supported.

## Settings (`.env`, never committed)

| Setting | Purpose |
| --- | --- |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Nemotron Lightning: labels groups and judges contextual spread |
| `SUPER_LLM_API_KEY`, `SUPER_LLM_BASE_URL`, `SUPER_LLM_MODEL` | Nemotron 3.5 Super VL: assesses original content including images, reads attachments, writes answers |
| `CHAT_JUDGE` | `lightning` (default) or `super`: model for the separate contextual spread verdict |
| `SAMPLE_SIZE` | Posts sampled per group for analysis (default 50, or all posts in a smaller group); an explicit `.env` value overrides the default |
| `HF_DATASET` | Defaults to `itsmebatuhan/bluesky-10m-posts-15-languages`; `--dataset` overrides an older `.env` value |
| `MIN_REVIEW_PRIORITY` | Current completed groups below this score are dropped (default 1; stale assessments remain visible) |
| `LIVE_DATASET` | Dataset streamed in the live test; defaults to the 15-language Bluesky dataset. `falbanese/US_Trump_2020_social_media` is also supported (use a separate `DATA_DIR`) |
| `LIVE_DATASET_REVISION` | Dataset revision to resolve on the first run (default `main`); its commit is pinned in the checkpoint |
| `LIVE_BATCH_SIZE`, `LIVE_WINDOW_POSTS` | Live batch size (500) and maximum rolling window (50,000); batch size must not exceed the window |
| `LIVE_PIPELINE_BATCHES` | Up to 4 batches in flight (1–8); upcoming batches are prepared while earlier groups await the LLM. Set 1 for serial processing |
| `LLM_CONCURRENCY` | Shared Live request limit across all batch and swarm reviews (default 8); use 32 for a warm, larger inference deployment |
| `STORAGE_CLEANUP`, `STORAGE_CLEANUP_HOURS` | Delete data the live view does not need at startup and every 6 hours (default on); see [storage](#keeping-the-vps-disk-from-filling-up) |
| `LIVE_MIN_CLUSTER_SIZE`, `LIVE_MIN_SAMPLES` | HDBSCAN group size (10) and density requirement (3) for Live and offline topic refinement; offline initial topics use `HDBSCAN_*` settings |
| `CHAT_JUDGE_TIMEOUT_S` | Time budget for the contextual judge, default 60 seconds; an existing `.env` value overrides it |
| `CHAT_ASSESSMENT_MAX_TOKENS` | Maximum output-token budget for structured content and spread assessments (default 32,768; set lower if the endpoint requires it) |

`.env.example` lists every setting with its default. Keep a warm Verda replica for
both models in production: if the Lightning judge does not answer within
`CHAT_JUDGE_TIMEOUT_S` (60 s by default), the chat still places the item and answers, but says
that the **spread review** is unavailable. The independent Super content assessment
is retained. It never switches judge on its own.

## Live processing and storage

Opening `/live` shows streaming groups and the Nemotron Super chat in one window,
and starts one shared processor **per server process**; it continues
when viewers disconnect, until the finite source is exhausted or the server
stops. Reloading reconnects to that run. Use one Uvicorn worker, protect the site
with authentication and rate limits, and expect LLM and CPU/GPU usage: opening
the page can start a long processing job. On error, fix the configuration and
restart the server; processing resumes after the last completed batch.
Chat placement and its related group use the retained live batches, not offline
artifacts. A request takes a consistent snapshot of the current window; it may
refer to a group that has since been evicted. Chat needs the Super endpoint and
at least one completed live batch. A restart clears the in-memory window, so
chat matching is unavailable until another batch finishes; if the dataset is
already exhausted, the window cannot be restored from the saved cursor.
Related-group matching uses batch-local groups; posts that also form a
cross-batch repeated-text candidate remain searchable, but that overlapping
candidate is not itself returned as the related group.
Source records read include rows later filtered out; processed posts count
eligible rows after cleaning and per-batch deduplication.

### Faster Live processing on a small VPS

The app prepares upcoming batches while the remote model reviews earlier ones.
For the dual-H100 inference deployment, start with:

```dotenv
LLM_CONCURRENCY=32
LIVE_PIPELINE_BATCHES=4
LIVE_BATCH_SIZE=500
LIVE_WINDOW_POSTS=50000
```

Restart/rebuild the app after changing settings. The startup log confirms the
pipeline depth, batch size and request cap. Keep one Uvicorn worker: adding
workers creates competing processors, not a faster shared window. Do not change
your Traefik override or enable Caddy for this update.

The H100s run the remote LLM; text embedding and grouping still run on the VPS.
Live clustering can use all available CPU cores. Concurrency is a maximum, not
a target: the app cannot fill 32 requests when too few groups need review.
Four batches add up to 2,000 source rows in flight, in memory only, alongside the
50,000-post completed window. Results and checkpoints are published in source
order, even when newer requests finish first. Each checkpoint saves that batch's
cursor, not the latest prefetched cursor, so interrupted work is replayed rather
than skipped.

Leave batch size and sampling unchanged initially; increasing either changes
grouping or review costs. If the VPS still has headroom and enough groups exist,
try `LIVE_PIPELINE_BATCHES=8`; revert to 4 if chat latency or memory pressure
increases. More concurrency alone cannot guarantee full GPU utilisation.
The displayed speed measures posts committed per processing interval, including
waiting for the next ordered batch and its swarm reviews; it is not the
end-to-end latency of an individual prefetched batch.

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

Each batch is embedded and clustered into smaller density-based leaves. Separately,
the rolling window checks for exact repeated text across at least two batches and
three distinct accounts, and sends those candidates for their own Lightning
assessment. These candidates overlap batch groups: do not add their post counts
together. Matching is case- and whitespace-normalized, but retains URLs and
mentions; paraphrases and near-duplicates across batches are not matched. A
repeated phrase can be benign, and neither it nor a model assessment proves
coordination or intent. Candidates, samples and the index are lost on restart.

The offline pipeline first creates broad topics and then attempts smaller
density-based subgroups within each topic. An unsplit topic or mixed remainder
is explicitly marked in its details. Existing offline artifacts must be reclustered
and reanalyzed to use this: clustering changes assignments and invalidates old
analyses, so use a separate data folder and Qdrant collection if the existing
run must be preserved. Sampling and clustering can still miss a minority
narrative, individual harmful posts or paraphrased campaigns; this is not
validated detection. Political discussion by itself is not evidence of propaganda.

Live baselines are batch-local; burst timing is excluded because these batches do
not establish a reliable dataset-wide time baseline. Duplicate identifiers are
removed within each batch, not across the entire stream. The UI shows at most 60
groups, hides completed groups below the review threshold and discloses failed
analyses. Smaller groups and repeatedly updated candidates can increase model calls.
The existing `/api/live/stream` remains a legacy offline replay API; the new
page uses `/api/live/processed`.

The live processor retains embeddings, text and results for up to 50,000
eligible posts in memory, evicting whole old batches. Embeddings alone require
about 77 MB at 50,000 posts (384 float32 values each); chat snapshots and post
data need additional memory. It does not write live post
vectors, samples or analyses to disk. Storage cleanup (`detector/storage.py`,
`STORAGE_CLEANUP`, `STORAGE_CLEANUP_HOURS`) removes app-created files
(`.parquet`, `.json`, `.jsonl`, `.npy`, `.npz`, `.tmp`) in `DATA_DIR` except the
Live cursor, Hugging Face hub repositories other than `EMBED_MODEL` and
`LIVE_DATASET` plus unreferenced old revisions, the prepared `datasets` cache
(Live only streams), and Xet logs older than a day. It refuses a root or home
`DATA_DIR`. Restart loses the live window.

## Legacy offline pipeline (optional)

Python 3.11+ and Qdrant are required. Storage cleanup deletes the pipeline's
results, so set `STORAGE_CLEANUP=false` while using it.

```bash
docker compose --profile offline up -d qdrant
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
  from anomaly flags and review-priority points.
- The details card shows signal lines, value/p50/p90 comparisons, above-p90
  marks and examples. Missing image metadata is shown as unknown.
- `stats_report.json` records flags, triggers, fallback buckets, reliability and
  runtime. New stats hashes mark saved model assessments stale until re-analysis;
  stale groups remain visible. Analyzer prompt v6 and cache inputs include the
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
3. The text is embedded with the same model as the posts. A related group must
   contain at least two of the nearest posts and at least 20% of them. The
   default neighbourhood is 25 posts. Display positions play no role.
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
- **Possible swarms raise concern by one level.** When far more posts than usual
  are near-identical, at least 5 accounts posted them and no single account
  dominates, concern goes up one level (at most to medium), even without a clear
  claim, blame or call to action. The card says when this happened. It is a
  reason to review, not proof of bots.
- A failed or unanalysed cloud is shown as such and is never dropped as insignificant.

Full contracts: [project spec](./Coordinated-behavior%20detector%20project%20spec.md),
[analyzer spec](./Analyzer%20spec%20narrative%20analysis%20stage.md),
[UI spec](./UI%20spec%20online%20narrative%20mapping%20and%20analysis.md).

</details>
