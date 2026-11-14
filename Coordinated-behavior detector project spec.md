# Coordinated-behavior detector: project spec

Single source of truth for the team and for coding agents. **MVP version:** one fixed dataset flows through one pipeline, from data to vectors to clusters to LLM analysis to a debug dashboard. Terse on purpose. **Decision** = fixed. **Default** = change only if the team agrees.

## 1. Goal and scope

Detect indicators of coordinated amplification in public social posts. The default run samples 10,000 random posts, creates 384D E5 embeddings, groups them using 5D UMAP + HDBSCAN, then asks Nemotron Lightning on Verda to describe every detected cloud from representative posts and spread statistics. The website ranks groups for human review using concern-coloured balls sized by post count, not a projected spatial view. Aggregate patterns are not proof of bot identity or common control.

- **In the MVP:** download the dataset once, embed, store vectors, cluster, compute simple behavior stats, run one LLM analysis per cluster, serve a FastAPI backend and a one-page debug dashboard on the Hostinger VPS.
- **Not in the MVP:** live ingestion, rolling windows, account or source scoring, a labeled evaluation set (section 12 lists these as next steps).
- **Never:** declare an idea true or false, or an account guilty. Output is descriptive, with evidence and a confidence level.
- **Challenge requirement:** the tool must run live on Hostinger with a public URL. Judges weigh honest detection logic and false-positive handling above polish.

## 2. Fixed decisions

| Part | Decision | Details |
| --- | --- | --- |
| Dataset | `itsmebatuhan/bluesky-10m-posts-15-languages` (Hugging Face) | 10.1M posts, 15 languages, August 9-12 2026 snapshot. Maps `did` to author, `did`+`rkey` to URI; preserves `lang`/`is_reply`; missing image presence stays null. CC BY-NC-SA 4.0: non-commercial use, attribution and share-alike; obtain permission for commercial use. Older `author`/`uri` sources still work via `--dataset` or `HF_DATASET`. |
| Embedding model | `intfloat/multilingual-e5-small` | Transformer encoder (BERT/XLM-R family): 12 layers, hidden size 384, 12 attention heads, about 118M parameters. Mean pooling over tokens, L2-normalised, 384-dim output. Max 128 tokens. Every input gets the prefix ` query:  ` (E5 models need a prefix; ` query:  ` is the symmetric-similarity choice). |
| Vector DB | Qdrant (Docker) | One collection `posts`, 384 dims, cosine distance, point id = `post_id`. |
| Clustering | UMAP, then HDBSCAN | UMAP to 5 dims (cosine), HDBSCAN from scikit-learn. Posts that fit no cluster stay as noise (`-1`). |
| LLM | Nemotron 3.5 Lightning, checkpoint `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, self-hosted on Verda | Hybrid Mamba-2 + MoE + attention model, 30B total / 3B active parameters. Served by vLLM on a Verda Serverless Container (OpenAI-compatible API, Verda inference API key), not on the VPS. Only the analyze stage calls it; the dashboard serves stored results (section 10). |
| Backend | Python 3.11+, FastAPI | Reads pipeline artifacts, no database server besides Qdrant. |
| Website | Three static pages, vanilla JS ES modules, circular group elements | Menu, Explore (group balls + chat) and Live test. No build step, no third-party scripts. |
| Chat model | Nemotron 3.5 Super VL (`nemotron-3.5-super-vl`) on Verda | OpenAI-compatible endpoint. Reads the user's request, files, articles and images, and writes the answer. Called only by `POST /api/chat`. |
| Deployment | Docker Compose on the Hostinger KVM 2 VPS | 2 vCPU, 8 GB RAM. Public URL on the free domain. |

- The embedding model runs on CPU, so the number of posts is capped by `N_POSTS` (section 4). Benchmark 1,000 posts first and size `N_POSTS` so embedding finishes in under 30 minutes.
- Nemotron officially supports English, Spanish, French, German, Italian and Japanese. Analyses of other languages may be weaker, and the dashboard says so.

## 3. Pipeline and repository layout

Stages exchange data through the artifacts below and the Qdrant collection. Each can be rerun alone; `cluster` also invokes the shared statistics stage.

1. **load:** Hugging Face dataset to `data/posts.parquet`
2. **embed:** posts to vectors in Qdrant
3. **cluster:** vectors from Qdrant to clusters, followed automatically by **stats**
4. **analyze:** one Nemotron assessment per detected cluster (representative sample + statistics, not all posts in one prompt)
5. **serve:** FastAPI backend plus the debug dashboard

CLI: `python -m detector <load|embed|cluster|project|stats|analyze|serve>`. `python -m detector all --posts 10000 --dataset itsmebatuhan/bluesky-10m-posts-15-languages` runs load/embed/cluster/stats/analyze and labels every cloud. `stats --inspect` recomputes spread numbers without reclustering, embeddings, Qdrant or LLM calls. `SEED` controls reproducible sampling.

| Artifact | Written by | Contents |
| --- | --- | --- |
| `data/posts.parquet` | load | `post_id` (int, 0..N-1), `uri`, `author`, `created_at`, `text`, nullable `has_images`, `lang`, nullable `is_reply` |
| Qdrant collection `posts` | embed | 384-dim vectors, point id = `post_id` |
| `data/assignments.parquet` | cluster | `post_id`, `cluster_id` (`-1` = noise) |
| `data/projection.parquet` | cluster, project | One real embedding projection per post: `post_id`, `cluster_id`, `x`, `y`, `z` |
| `data/projection_model.npz` | cluster, project | The fitted PCA (mean, 3 components, scale), so new text lands in the same display space |
| `data/embeddings.npy` | cluster, project | The 384D vectors in `projection.parquet` order, so the website finds nearest posts without Qdrant |
| `data/clusters.json` | cluster, stats | `run`, `dataset` reliability, size-bucket/global `baseline`; per cluster: flat compatibility stats plus `stats`, `flags`, `signals`, `baseline_bucket`, `stats_hash`, representatives |
| `data/stats_report.json` | stats | Flags/triggers, reliability, baseline fallbacks and timings |
| `data/analyses.jsonl` | analyze | One checkpointed AnalysisRecord per cluster; sample mapping, relevance, rules, prompts, raw/parsed output and timings (section 8) |
| `data/analyze_report.json` | analyze | Selected/analysed/cached/failed counts, relevance kinds, default tiers and new-request latencies |

```
repo/
  README.md  Makefile  Caddyfile  pyproject.toml  docker-compose.yml  .env.example  .gitignore
  docker/Dockerfile
  src/detector/
    __main__.py     # CLI entry point
    config.py       # all settings from env (pydantic-settings)
    io.py           # artifact paths and read/write helpers
    load.py         # stage 1
    embed.py        # stage 2
    cluster.py      # stage 3: UMAP + HDBSCAN, writes assignments and clusters.json
    projection.py   # deterministic 3-component PCA of original 384D embeddings for visualization only
    stats.py        # pure functions: behavior stats, baseline, organic-looking check
    analyze.py      # stage 4: sampling, prompt, LLM call, validation
    prompts.py      # versioned system prompt and user template
    rules.py        # pure enforced rules, behavior anomaly and display tier
    llm.py          # shared API client and bounded, model-aware readiness checks
    text.py         # shared pure helpers: JSON extraction, post-text cleaning
    embedding.py    # shared embedding helpers (device, cleaning, encode), used by embed and chat
    chat.py         # serve-time chat: attachments, placement, judge, Super answer
    schema.py       # pydantic models: analysis output, item verdict, records
    api.py          # FastAPI app: pages, /api/*, chat and live stream
  web/              # index.html (menu), explore.html, live.html, map.js, explore.js, live.js, menu.js, styles.css, fonts/
  tests/            # stats, rules, pipeline, LLM, API, chat, live stream, projection
  data/             # git-ignored artifacts
```

## 4. Configuration

All settings come from environment variables through `config.py`. Commit `.env.example` with these names; the real `.env` stays out of git. Defaults below are the **Default** values.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_DIR` | `./data` | Where artifacts are written |
| `HF_DATASET` | `itsmebatuhan/bluesky-10m-posts-15-languages` | Dataset id; explicit `--dataset` overrides older `.env` values |
| `N_POSTS` | `10000` | Random sample size after cleaning. `0` = all posts. Smoke test: `2000` |
| `SEED` | `42` | Seed for sampling, UMAP and sample selection |
| `MIN_POST_CHARS` | `20` | Shorter posts are dropped |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model |
| `EMBED_MAX_TOKENS` | `128` | Truncation length |
| `EMBED_BATCH` | `64` | Batch size for encoding and for Qdrant upserts |
| `EMBED_DEVICE` | `auto` | `cuda`, `mps` or `cpu`; `auto` picks the best available |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | empty | Set if Qdrant is reachable from outside the VPS |
| `QDRANT_COLLECTION` | `posts` | Collection name |
| `UMAP_N_NEIGHBORS` | `15` | UMAP neighbours |
| `UMAP_N_COMPONENTS` | `5` | UMAP output dimensions |
| `UMAP_MIN_DIST` | `0.0` | UMAP min distance |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `25` | Smallest cluster kept |
| `HDBSCAN_MIN_SAMPLES` | `5` | HDBSCAN core-point strictness |
| `LLM_API_KEY` | none (required for analyze) | Verda inference API key (console: Credentials, Inference API Keys) |
| `LLM_BASE_URL` | `https://containers.verda.com/<deployment>/v1` | OpenAI-compatible base URL of the Verda deployment |
| `LLM_MODEL` | `nemotron-lightning` | Model name the endpoint serves (vLLM --served-model-name) |
| `LLM_TEMPERATURE` / `LLM_TOP_P` | `1.0` / `0.95` | Sampling values recommended on the model card |
| `LLM_MAX_TOKENS` | `4096` | Output cap, leaves room if the model emits reasoning text |
| `LLM_MAX_TOKENS_CEILING` | `16384` | Upper budget for two compact-output retries after token truncation; must be at least LLM_MAX_TOKENS and fit the server's limits |
| `LLM_CONCURRENCY` | `8` | Parallel requests to our own endpoint |
| `LLM_TIMEOUT_S` | `120` | Per-request timeout |
| `LLM_WARMUP_TIMEOUT_S` | `900` | How long analyze waits for a scaled-to-zero endpoint to load the model |
| `MIN_ANALYSIS_SIZE` | `1` | Size cutoff for standalone analysis; `--all-clusters` and `all` bypass it |
| `MAX_ANALYZED_CLUSTERS` | `0` | Standalone analysis limit; 0 = unlimited; `--all-clusters` and `all` bypass it |
| `SAMPLE_SIZE` | `50` | Posts sent per cluster: up to 10 representative, 5 earliest and random posts to fill |
| `SUPER_LLM_API_KEY` / `SUPER_LLM_BASE_URL` | none | Verda inference key and OpenAI-compatible base URL (ending in `/v1`) of the Nemotron 3.5 Super VL deployment. Chat is off without them |
| `SUPER_LLM_MODEL` | `nemotron-3.5-super-vl` | Model name the chat endpoint serves |
| `SUPER_LLM_MAX_TOKENS` / `SUPER_LLM_TIMEOUT_S` | `4096` / `120` | Output cap and per-request timeout for the chat model |
| `CHAT_JUDGE` | `lightning` | Which model judges a submitted item: `lightning` (the `LLM_*` endpoint) or `super`. No automatic switch |
| `CHAT_JUDGE_TIMEOUT_S` | `60` | How long a chat waits for the judge before answering without a verdict |
| `CHAT_NEIGHBORS` | `25` | Nearest stored posts used to place an item and pick its nearest cloud |
| `CHAT_MAX_ATTACHMENTS` / `CHAT_MAX_ATTACHMENT_MB` | `4` / `4` | Upload limits per message |
| `CHAT_MAX_PER_MINUTE` / `CHAT_CONCURRENCY` | `20` / `2` | Cost guards for the public demo: total chat requests per minute and in parallel |
| `MIN_REVIEW_PRIORITY` | `1` | Assessed clouds scoring below this are dropped from the map. `0` keeps everything |
| `LIVE_POSTS_PER_SECOND` | `120` | Legacy offline replay API speed |
| `LIVE_DATASET` / `LIVE_BATCH_SIZE` / `LIVE_WINDOW_POSTS` | 15-language dataset / `500` / `25000` | Stream source, grouping batch and bounded in-memory rolling window |
| `RELEVANCE_HIDE_BELOW` | `0.35` | Hide lower-relevance topics unless their spread is unusual |
| `RELEVANCE_FULL_FROM` | `0.65` | Full-opacity relevance threshold; must be at least HIDE_BELOW |
| `ANOMALY_PERCENTILE` | `90` | Behavior baseline percentile: 50, 75 or 90 |
| `MIN_SAMPLE_FOR_CONFIDENCE` | `8` | Smaller samples cap confidence at 0.4 |

## 5. Stage 1: load data

Command: `python -m detector load`. Input: the Hugging Face dataset, downloaded once and cached. Output: `data/posts.parquet`.

1. Load the Hugging Face `train` split with disk-backed Arrow. Do not materialise the full 10M-post text column in pandas.
2. Normalise in batches of 10,000 rows. Require `text`, `created_at` and either `author`/`uri` or `did`/`rkey`; construct a canonical Bluesky URI including both DID and record key. Keep language and nullable reply/image metadata.
3. Strip and remove null/empty or too-short text, invalid timestamps and missing/empty identifiers. Deduplicate canonical URIs using an Arrow index before sampling.
4. **Do not** remove duplicate texts. Repeated text is a signal used in stage 3.
5. Seeded-shuffle eligible unique rows and select `N_POSTS` without replacement, then convert only the selected rows to pandas. With `--scan-limit`, stream only that many prefix rows and sample within that prefix; explicitly record that it is not a global sample.
6. Sort by `created_at`, then `uri`, assign `post_id`, and atomically replace `posts.parquet`. The fingerprint covers source, sampling method, ids, content and metadata. A change invalidates assignments, projection/model/embedding files, cluster stats and analyses; retain the embedding report so its fingerprint triggers vector rebuilding.
7. Save cleaning counts, language distribution, sampling method and image metadata availability in `load_report.json`. An insufficient eligible sample gets an explicit warning; no eligible records is an error.

Notes:

- `author` is the source identifier (DID in the new dataset). The tool reports aggregate patterns, never a per-account bot score.
- The new dataset card specifies CC BY-NC-SA 4.0; observe attribution/non-commercial/share-alike requirements and obtain permission before commercial hosting.
- The dataset is a snapshot, so all time-based stats are relative to its own time span.

## 6. Stage 2: embed and store

Command: `python -m detector embed`. Input: `data/posts.parquet`. Output: Qdrant collection `posts`.

**Model**

- `intfloat/multilingual-e5-small`, loaded with `SentenceTransformer(EMBED_MODEL, device=...)` and `model.max_seq_length = EMBED_MAX_TOKENS`.
- Architecture: Transformer encoder with 12 layers, hidden size 384 and 12 attention heads, then mean pooling over tokens, then L2 normalisation. Output: one 384-dim float32 vector per post. Encode with `normalize_embeddings=True` so cosine similarity equals the dot product.
- Input string: `"query: " + clean_text`. `clean_text` is the post text with URLs replaced by `<url>` and whitespace collapsed. Hashtags and mentions stay. The stored `text` in `posts.parquet` is never modified.

**Qdrant**

- Collection `posts`: vector size 384, distance Cosine, point id = `post_id`, payload `{"post_id": int}`.
- Create the collection if missing. `--recreate` explicitly rebuilds it.
- **Resumable:** read existing point ids and encode only missing posts when dataset fingerprint and embedding parameters match. Changed or unverified fingerprints/model/token/preprocessing settings rebuild the collection to avoid stale vectors. A long CPU run can be interrupted and restarted.
- Encode with `batch_size=EMBED_BATCH`, upsert in batches of 512, show a progress bar.

**Checks**

- Point count must equal the number of posts; otherwise fail before clustering.
- Pick 3 random posts and print their 5 nearest neighbours. Neighbours should read as the same topic. If they do not, stop and fix before clustering.
- Write throughput (posts per second) to `data/embed_report.json`.

**If CPU is too slow:** run the pipeline on a laptop with a GPU and copy the artifacts to the VPS with `make push-data` (section 10). `make bench` shows the speed before a long run.

## 7. Stage 3: cluster and stats

Command: `python -m detector cluster`. Input: vectors in Qdrant and `data/posts.parquet`. Output: `data/assignments.parquet` and `data/clusters.json`.

1. Read all vectors with Qdrant `scroll(with_vectors=True)` into a float32 array `X` of shape (N, 384), ordered by `post_id`. Assert N equals the number of posts.
2. Reduce with UMAP: `umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS, min_dist=UMAP_MIN_DIST, metric="cosine", random_state=SEED)`. A fixed `random_state` makes UMAP single-threaded: slower but reproducible. If it is too slow, drop `random_state` and set `n_jobs=2`.
3. Cluster with `sklearn.cluster.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, metric="euclidean", cluster_selection_method="eom")` on the reduced vectors. Label `-1` = noise.
4. Write `assignments.parquet` (`post_id`, `cluster_id`). Invalidate analyses if cloud membership changed.
5. For each non-noise cluster compute the centroid (mean of its rows of `X`, re-normalised), the `representative_post_ids` (the 10 posts with the highest cosine similarity to the centroid) and the behavior stats below.
6. Compute `baseline` over all non-noise clusters, then `organic_looking` per cluster.
7. Write `clusters.json`: a `run` block (parameters, `n_posts`, `n_clusters`, `noise_ratio`, seconds per step), the `baseline`, and the cluster list sorted by `size` descending.
8. Project the original 384D embeddings with deterministic 3-component PCA, scaled isotropically into the unit sphere; write `projection.parquet` for all posts, including noise, plus `projection_model.npz` and `embeddings.npy` for chat placement. `python -m detector project` rebuilds these three files from the stored vectors without changing cloud membership or analyses. Store source dimensions, retained variance for 2D/3D, and the lossy-projection warning under `run.projection`. The 2D view uses PC1/PC2; 3D adds PC3. These display coordinates do not affect UMAP/HDBSCAN grouping or Nemotron scoring.

Expect minutes, not seconds, for 100k posts. Log progress per step. Use `N_POSTS=2000` for quick runs.

**Behavior stats** (pure functions in `stats.py`, unit tested, no LLM involved).
The [behaviour-statistics spec](./Behaviour%20statistics%20spec.md) supersedes the
original exact-only/global-baseline design:

| Stat | Definition |
| --- | --- |
| `size` | Number of posts in the cluster |
| `n_authors` | Distinct authors |
| `top_author_share` | Share of posts written by the most active author |
| `dup_ratio` | Share with a near-duplicate in the group: MinHash/LSH estimated Jaccard >= 0.8, 64 permutations, connected components; NFKC, URL and mention normalisation |
| `exact_dup_ratio`, `dup_authors`, `largest_dup_group` | Exact repetition, distinct authors of near-duplicates and largest repeated-text group |
| `authors_per_post`, `top5_author_share` | Aggregate author diversity and concentration |
| `top_domain_share` | Share of posts containing the most common link domain (0 if no links) |
| `top_hashtag_share` | Share of posts containing the most common hashtag (0 if none) |
| `burstiness` | Share in the densest configurable sliding window (10 minutes by default) |
| `time_span_min` | Minutes between the first and last post |

**Baseline:** p50/p75/p90 in size buckets 25-49, 50-199 and 200+, with global fallback below 10 groups per bucket. Includes near-duplicates, author concentration/diversity, burstiness, domain/hashtag concentration. Links and hashtags never independently trigger anomalies.

**Flags:** organic-looking requires near-duplicates, top-author share and reliable burstiness <= bucket p75. Anomaly requires at least `MIN_ANALYSIS_SIZE` posts and any active metric > chosen percentile. If the dataset spans less than six burst windows, timing is shown but ignored in flags, caps and priority. Image share is unknown without complete image metadata. Statistics are rounded to four decimals and written atomically; membership never changes. Retained default `MIN_ANALYSIS_SIZE=1` preserves standalone analysis; use 20 for the spec's minimum anomaly size.

## 8. Stage 4: LLM analysis

Detailed source of truth: [Analyzer spec: narrative analysis stage](./Analyzer%20spec%20narrative%20analysis%20stage.md).

Command: `python -m detector analyze` (`--force` ignores the cache; `--cluster <id>` analyses one cluster; `--all-clusters` bypasses legacy selection limits; `--dry-run` prints prompts without network calls or writes). Input: `clusters.json`, `assignments.parquet`, `posts.parquet`. Output: `data/analyses.jsonl`, checkpointed after each cluster, and `data/analyze_report.json` at completion. The LLM describes content, framing and narrative relevance. It does not decide whether anything is coordinated; that comes from the behavior stats (section 7).

**Selection and sampling**

- By default label every non-noise cloud; standalone analysis may be capped explicitly with `MIN_ANALYSIS_SIZE`/`MAX_ANALYZED_CLUSTERS`. `--all-clusters` and the full pipeline always process every cloud. Other clouds stay visible as `not_analysed`.
- Sample `SAMPLE_SIZE` (default 50) posts per cluster: up to 10 `representative_post_ids`, then up to 5 earliest remaining posts, then random posts to fill (seeded with `SEED + cluster_id`), no duplicates. If the cluster is too small, take what exists.
- Shuffle with the same seed, give ids `p1..pK`; replace URLs with `<url:domain>`, collapse whitespace, cut to 400 characters, then escape `<`, `>` and `&`.
- `input_hash` covers model id, prompt version, sample ids/text, rounded stats, organic flag and enforced-rule settings. Display thresholds are excluded. Skip unchanged successful results without warming the endpoint.

**Call**

- Async OpenAI SDK: `AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT_S, max_retries=0)`, `chat.completions.create` with `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_MAX_TOKENS`. A semaphore limits parallel calls to `LLM_CONCURRENCY`.
- Warm-up: before the first call, poll GET {LLM\_BASE\_URL}/models every 10 s until it answers 200 or LLM\_WARMUP\_TIMEOUT\_S passes, printing progress. A scaled-to-zero Verda endpoint needs minutes to load the model. Structured output: the endpoint runs vLLM, so pass the Analysis JSON schema as response\_format of type json\_schema to force valid JSON. If the model card documents a chat-template switch that turns reasoning off, pass it in extra\_body; reasoning on only costs time.
- Parse defensively whatever the settings: remove any reasoning block (for example `<think>...</think>`) and code fences, take the text from the first `{` to the last `}`, `json.loads`, validate with pydantic.
- Retries: up to 3 on timeouts, 429 or 5xx with backoff (2 s, 4 s, 8 s); 1 retry on invalid JSON/schema with the validation error appended. After that `status = "failed"`, raw response stored. A refusal or empty answer is also `failed`.
- Truncation recovery: two additional bounded retries with compact JSON and doubled output budget up to `LLM_MAX_TOKENS_CEILING`. Never accept truncated JSON; keep all sampled posts and reasoning disabled. Exhaustion stores the final raw completion as a visible failure.

The client sends `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`,
as supported by the model's published `chat_template.jinja`, to leave the output
budget for the final answer. `llm-check` uses the same switch and `LLM_MAX_TOKENS`.
Token-limit finishes are failures, even if partial content is present. For missing
or truncated answers, `raw_response` preserves the full completion JSON (including
finish reason, usage and any reasoning) for debugging.

**System prompt:** version `v5` in `prompts.py`, with readable topic labels, the complete relevance rubric plus near-duplicate/multi-account context, size-bucket percentiles and timing reliability. The version and stats hashes invalidate older cached assessments. Political symmetry, dominant-cluster scoring and untrusted-data handling remain.

The user template gives each statistic beside its median/p90, followed by escaped posts. The JSON schema is supplied through structured output.

**Output schema (pydantic model `Analysis` in `schema.py`)**

```
label: str                 # max 6 words, neutral name of the idea
summary: str               # max 40 words, neutral
relevance_kind: enum        # narrative, issue_discussion, news_relay, plain_topic,
                            # personal_chatter, commercial_or_spam, unclear
relevance: float            # 0..1, content relevance independent of concern
relevance_reason: str       # max 25 words
claims: [{text, kind: "checkable"|"opinion", evidence: [post ids]}]   # max 5
techniques: [{name, evidence: [post ids]}]
templating: {detected: bool, evidence: [post ids]}
framing_diversity: "low"|"medium"|"high"
legit_lookalike: str|null  # e.g. "looks like syndicated news headlines"
concern_level: "none"|"low"|"medium"|"high"|"insufficient_evidence"
confidence: float          # 0..1
explanation: str           # max 60 words, plain language, cites the stats it used
```

Technique names: `fear_appeal`, `scapegoating`, `false_urgency`, `loaded_language`, `false_dichotomy`, `conspiracy_framing`, `unverified_authority`, `call_to_mass_action`, `whataboutism`.

**Rules enforced in code after validation, in this order**

1. Drop unknown evidence ids and empty claims/techniques; reset templating if no evidence remains.
2. Clamp relevance/confidence to 0..1. Nonfinite scores are invalid.
3. Cap medium/high concern to low for organic statistics (`organic_stats`).
4. Cap medium/high concern to low for legitimate lookalikes (`lookalike`).
5. Cap concern to none if relevance is below 0.2 and spread is not anomalous (`plain_topic`).
6. Cap confidence at 0.4 below `MIN_SAMPLE_FOR_CONFIDENCE`.
7. Preserve `concern_level_model` and `relevance_model`; log every enforced change in `rule_log`.

Overlong label, summary, relevance reason and explanation are truncated with an ellipsis, not rejected.

**Each line of `analyses.jsonl`:** `cluster_id`, `input_hash`, `prompt_version`, `status` (`ok` or `failed`), `sample` (list of `{pid, post_id}`), `behavior_anomaly` (`flag`, `triggered_by`), `system_prompt`, `user_prompt`, `raw_response`, `parsed` (after the rules), `concern_level_model`, `relevance_model`, `capped_by`, `rule_log`, `attempts`, `latency_s`, `model`, `created_at`. Display tier is not stored.

Network-free tests cover all rules/order, thresholds, parsing/truncation, retries, dry-run, cache invalidation, CLI and checkpoint durability. Manual calibration, mirror and injection checks remain pre-demo work described in the analyzer spec.

## 9. Stage 5: API and website

Command: `python -m detector serve` (uvicorn, port 8000; `--data-dir` picks the artifact folder). It loads `clusters.json`, `analyses.jsonl`, `assignments.parquet`, `posts.parquet`, `projection.parquet`, `projection_model.npz` and `embeddings.npy` into memory at startup, so restart it after rerunning a stage. No auth, same-origin, no CORS. Everything is read-only except `POST /api/chat`, which stores nothing.

| Endpoint | Returns |
| --- | --- |
| `GET /`, `/explore`, `/live` | The menu, the map with chat, and the live test. Other files are served from `web/` under `/assets` |
| `GET /api/health` | `{"status": "ok"}` |
| `GET /api/meta` | `run`, `baseline`, models, counts (`n_posts`, `n_clusters`, `n_analysed`, `n_failed`, `n_pending`, `n_hidden`, `n_discarded`), thresholds, `min_review_priority`, projection metadata, `chat` (`available`, `error`, `model`, `judge`, `limits`) and `live` (`posts_per_second`) |
| `GET /api/clusters` | Cluster stats/status, label, concern/confidence, relevance/kind/reason, current `behavior_anomaly`, `display_tier`, `review_priority`, `review_score` and `discarded`. Excludes hidden tiers by default; `?include_hidden=true` returns all |
| `GET /api/clusters/{id}` | Full cluster record plus its analysis line: `parsed`, prompts, `raw_response`, and `sample` with each post's text and author |
| `GET /api/clusters/{id}/posts?offset=0&limit=50` | Paginated posts of the cluster |
| `GET /api/map` | `columns: [post_id, cluster_id, x, y, z]`, compact `points` arrays and projection metadata; HTTP 503 with a rebuild instruction when the projection is missing |
| `GET /api/live/stream` | Server-sent events: `start`, then batches `{points, sent, total, clock}` every 250 ms in post-time order, then `done` |
| `GET /api/live/processed`, `/api/live/clusters/{id}` | Shared streamed-dataset processing snapshots and rolling group details. Groups are batch-local; memory is bounded by the rolling window. |
| `POST /api/chat` | JSON `{message, attachments: [{name, type, data(base64)}], history}`. Returns `reply`, `placed_text`, `placed_from`, `placement` (`coordinates`, `nearest_cloud`, `neighbors`), independent `content_assessment` and `content_review` (`model`, `status`, `reason`), contextual `verdict` and `judge` (`model`, `status`, `reason`, `rule_log`), `attachments`, `models` |

**Review priority** (API-time, 0 to 100): a triage heuristic, not a probability or validated detection quality. Final concern contributes 0/15/39/60 points for none/low/medium/high; insufficient evidence adds zero. For each of duplicate ratio, top author share and burstiness, normalized excess is `max(0, value - baseline) / (1 - baseline)` at the chosen anomaly percentile (zero when the baseline is one). Spread adds `20 * (max(excesses) + mean(excesses))`, up to 40. The API exposes `review_priority` and the full `review_score` breakdown; failed or unanalysed clouds have an explicit `spread_only` partial status. Confidence and relevance stay separate.

**Discarding insignificant clouds.** A cloud is `discarded` when its current analysis succeeded and review priority is below `MIN_REVIEW_PRIORITY` (default 1). Discarded clouds and noise are not drawn but remain stored for matching. Failed, unanalysed and stale-assessment groups are never discarded. `analysis_stale` compares stats hashes, prompt version and sample count; `/api/meta.n_stale` counts them. Relevance tiers remain available.

**Chat pipeline** (`chat.py`, one request, nothing stored):

1. **Read.** Attachments are decoded in memory: text, Markdown, CSV, JSON, XML, HTML (tags, scripts and styles removed), PDF (`pypdf`, up to 60 pages) and PNG/JPEG/WebP/GIF images. Other types, empty files, binary files and oversized files are rejected with a reason. Text is capped at 40,000 characters per file and 80,000 in total; truncation is reported.
2. **Digest.** A request of at most 600 characters without attachments is placed as written. Otherwise Nemotron Super writes a digest of the material as one short post (at most 60 words, the material's own framing), and the digest is what gets placed. The response says which was used and shows the digest.
3. **Place.** The text is embedded with the same E5 model and preprocessing as the posts, projected with the stored PCA model, and compared to all stored embeddings. The nearest cloud is the one holding the most of the `CHAT_NEIGHBORS` nearest posts, if it holds at least 20 percent of them; otherwise the item is reported as landing among ungrouped posts.
4. **Assess original content.** Super VL receives the original message, extracted attachment text and original images, not the placement digest. It returns `ContentAssessment`: category (`propaganda`, `political_persuasion`, `commercial_persuasion`, `informational`, `personal_expression`, `unclear`), content concern, techniques, up to five quotes or visible details, confidence and explanation. Propaganda-style persuasion is an observable rhetorical category, not proof of falsehood, intent or coordination. Fear plus blame and political mobilisation qualifies independently of neighbouring groups. Ordinary advocacy or criticism alone does not. Opposite viewpoints use identical criteria; critical quotation is not endorsement. Propaganda or medium/high content concern requires nonempty evidence.
5. **Contextual spread judge.** The configured model (Lightning by default) gets the placement text, nearest cloud's label, summary, concern/spread statistics and eight closest posts. It returns the existing `Verdict`. Both structured assessments disable thinking, allow three attempts within one time budget, and grow output from 2048 to 4096 to 8192 on truncation. Invalid or truncated JSON is never accepted.
6. **Spread rules** (`rules.apply_verdict_rules`, in order): clamp scores; drop duplicate techniques; no nearest cloud means `fits_cloud = false`; medium/high contextual concern is lowered to low unless the item fits a cloud with unusual spread or medium/high concern (`no_spread_evidence`); relevance below 0.2 without anomalous spread gives concern none (`plain_topic`). These rules apply only to the contextual spread verdict, never to content assessment. Changes are returned in `rule_log`.
7. **Answer.** Super answers in plain text, leading with the content category and evidence. It distinguishes content from contextual spread concern; low group concern or a missing spread judge cannot downgrade the content result.

Failure handling: an unavailable content assessment has `content_assessment: null` and an explicit reason in `content_review`; it is never fabricated. If the spread judge exceeds `CHAT_JUDGE_TIMEOUT_S` or returns no valid verdict, `verdict` is null with the reason, but a successful content assessment stands. There is no silent judge switch. A failed Super digest or final answer returns HTTP 502 with a reason. Bad input is HTTP 400, an oversized body 413, rate/concurrency guard 429, and missing placement artifacts or credentials 503 with the fix. Material, digest and posts are untrusted in every prompt; models have no tools.

**Website** (`web/`, see the UI spec): menu, **Explore** with group balls from the preloaded 10,000-post run and rounded Super chat, and **Live test** with new batch-local groups from a streamed historical dataset and a rolling ranking. Ball area is proportional to post count; colour shows group concern. Positions are a deterministic non-overlapping layout, never semantic coordinates. Both modes have divided group cards with example posts. PCA artifacts remain supported by the backend but are not used to position balls; Explore renders counts even without a projection.

## 10. Run locally and deploy

Every machine runs the same code and the same commands; only `.env` differs. Test locally with a small sample, deploy to the VPS when it works. A `Makefile` wraps every command below, and its recipes are the plain commands, so anyone can copy them (on Windows, use WSL).

**Compose: one file for both**

- `qdrant`: pinned `qdrant/qdrant` tag, volume `./qdrant_storage:/qdrant/storage`, port `127.0.0.1:6333:6333` (reachable from the host only, never from the internet).
- `app`: built from `docker/Dockerfile` (`python:3.11-slim`, CPU-only torch wheels, `src/` and `web/` copied in). `env_file: .env`, plus `environment:` `QDRANT_URL=http://qdrant:6333` and `DATA_DIR=/app/data` so the container always finds Qdrant. Volumes `./data:/app/data` and `./hf_cache:/root/.cache/huggingface`. Port `127.0.0.1:8000:8000`. Command `python -m detector serve --host 0.0.0.0`.
- `caddy`: image `caddy:2`, **only in the `prod` profile**, ports 80 and 443. `Caddyfile`: `{$SITE_ADDRESS} { reverse_proxy app:8000 }`. `SITE_ADDRESS=:80` serves plain HTTP on the VPS IP for testing; set it to the domain to get HTTPS.
- Locally, `docker compose up -d` starts Qdrant and the app. On the VPS, `docker compose --profile prod up -d` adds Caddy.

**`.env.example`**

```
LLM_BASE_URL=https://containers.verda.com/<deployment>/v1
LLM_API_KEY=
N_POSTS=2000
EMBED_DEVICE=auto
SITE_ADDRESS=:80
```

Everything else uses the section 4 defaults. `QDRANT_URL` defaults to `http://localhost:6333`, which is right for native runs; the container overrides it. `serve` binds `127.0.0.1` unless `--host` is given.

**Make targets**

| Target | What it does |
| --- | --- |
| `make setup` | Creates `.env` from `.env.example` if missing, creates `.venv`, runs `pip install -e ".[dev]"` |
| `make qdrant` | `docker compose up -d qdrant` |
| `make bench` | `python -m detector embed --benchmark 1000`: encodes 1,000 posts without storing them, prints device, posts per second and projected time for `N_POSTS` |
| `make smoke` | Pipeline natively with `N_POSTS=2000`, then serves at http://localhost:8000 |
| `make run` | Pipeline natively with exactly 10,000 sampled posts and every cloud assessed, then serves |
| `make serve` | Serves existing artifacts natively |
| `make docker-smoke` | Same as `smoke`, all inside Docker (no Python needed on the host) |
| `make test` | `pytest` |
| `make reset` | Asks to confirm, stops Qdrant, deletes `data/` and `qdrant_storage/` |
| `make push-data VPS=root@<ip>` | Copies local artifacts to the VPS (see below) |
| `make deploy` | On the VPS: `git pull`, then `docker compose --profile prod up -d --build` |
| `make llm-check` | Waits for the LLM endpoint to load (up to LLM\_WARMUP\_TIMEOUT\_S), lists its models, sends one tiny chat completion, prints latency. Fails clearly if the URL, key or model name is wrong |

**Local first run**

Needs Docker, Python 3.11 and git.

```
git clone <repo> && cd <repo>
make setup          # then put LLM_BASE_URL and LLM_API_KEY in .env
make qdrant
make smoke          # dashboard at http://localhost:8000
```

- Native Python uses a GPU when there is one: Apple Silicon (`mps`) or NVIDIA (`cuda`). Docker on a Mac cannot use the GPU, so `make docker-smoke` embeds on CPU.
- Without `LLM_API_KEY`, `analyze` prints a clear warning and skips; `make smoke` still runs the other stages and serves, with clusters shown as `not_analysed`.
- Analysis and `llm-check` use the same Verda OpenAI-compatible API client and authenticated `/models` readiness check. They verify `LLM_MODEL`, fail immediately on configuration/authentication/model-list errors, and report elapsed/remaining warm-up time. SDK retries are disabled to avoid hidden retry multiplication.
- **Production Verda:** keep at least one warm/ready replica using the deployment's scaling settings; confirm a healthy deployment and completed engine startup before directing traffic to it. A scaled-to-zero inference container is a cold-start tradeoff, not an always-ready API. For a bounded readiness test, run `LLM_WARMUP_TIMEOUT_S=120 LLM_TIMEOUT_S=60 make llm-check`; this does not change `.env`. Run analysis only after it succeeds.
- A `routed_experts.py` gate/up-projection warning does not by itself prove startup failure. If readiness times out, inspect the complete Verda startup logs and deployment health, including subsequent errors, GPU memory and model/vLLM image compatibility. App-side retries cannot repair the remote engine. No automatic provider, endpoint, credential or model switching is performed.
- **Changing `N_POSTS` is safe.** `load` writes a dataset fingerprint (`N_POSTS`, `SEED`, hash of the sorted `uri` list) to `data/load_report.json`. `embed` saves the fingerprint its collection was built from in `data/embed_report.json` and recreates the collection when they differ. No `make reset` needed.

**VPS first deploy**

Hostinger KVM 2 with the Docker template. The repo must be public or have a read-only deploy key.

```
ssh root@<vps-ip>
git clone <repo> app && cd app
cp .env.example .env   # set LLM_BASE_URL and LLM_API_KEY; keep SITE_ADDRESS=:80 for now
make deploy            # dashboard at http://<vps-ip>
```

Then get data onto it, either way:

- **Run the pipeline on the VPS:** `docker compose run --rm app python -m detector all`, then `docker compose restart app`. Fine for smoke runs and up to about 100k posts.
- **Copy from a laptop** (faster for big runs): `make push-data VPS=root@<vps-ip>` stops local Qdrant, packs `data/` and `qdrant_storage/`, copies them with `scp`, then on the VPS stops the stack, unpacks into `app/` and starts it again. Both machines use the same pinned Qdrant tag.

When the domain points at the VPS, set `SITE_ADDRESS=<domain>` in `.env` and run `make deploy` again; Caddy fetches the HTTPS certificate. **Updates:** push from a laptop, then `make deploy` on the VPS. Rerun the pipeline only when a stage or its settings changed.

**LLM endpoint on Verda**

The analysis model runs as our own deployment on Verda Serverless Containers, paid from the team's Verda credits. Laptops and the VPS only call it over HTTPS. Only `analyze` uses it: the dashboard serves stored results, so the live demo never depends on Verda. The challenge allows hosted models as long as the product runs on Hostinger.

Deployment settings (Verda console, Containers, New deployment):

- **Image:** `docker.io/vllm/vllm-openai`, pinned to a release that supports Nemotron 3.5 Lightning. Check vLLM's recipe for the model and Verda's driver note (R580) when choosing the tag.
- **GPU:** one H100 80 GB or newer. The NVFP4 checkpoint needs Hopper or Blackwell.
- **Start command:** `vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --served-model-name nemotron-lightning --max-model-len 32768`, plus the reasoning-parser flag from vLLM's recipe. A cluster prompt is about 3k to 5k tokens, so 32k context leaves memory for parallel requests.
- **Env:** `HF_TOKEN` (read-only Hugging Face token) so the weights download without rate limits.
- **Port and health:** vLLM's port 8000, health check `/health`.
- **Scaling:** min 0, max 1 replica. Verda bills per minute while a replica runs, so it scales back to zero after the run. The first request after idle waits for the model to load; `analyze` handles that with its warm-up wait (section 8).
- **Credentials:** create an inference API key under Credentials, Inference API Keys. The base URL is shown under Containers API as `https://containers.verda.com/<deployment>/`; `LLM_BASE_URL` is that plus `v1`.

Run `make llm-check` before a full run. With 40 clusters and `LLM_CONCURRENCY=8`, analysis takes a few minutes once the model is loaded.

**Fallback:** only `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` point at Verda, so any OpenAI-compatible Nemotron endpoint works without code changes. NVIDIA's API (`https://integrate.api.nvidia.com/v1`, model `nvidia/nemotron-3.5-lightning-30b-a3b`) is the backup if the Verda deployment fails during the hackathon.

**Expected embedding time**

Rough estimates for short posts with `multilingual-e5-small`; `make bench` gives the real number on each machine.

| Machine | Posts per second | 100k posts | 2M posts |
| --- | --- | --- | --- |
| VPS, 2 vCPU | about 50 to 150 | 10 to 35 min | 4 to 11 h |
| Laptop CPU, or Docker on a Mac | about 150 to 400 | 4 to 11 min | 1.5 to 4 h |
| Apple Silicon (`mps`) or NVIDIA GPU | about 800 to 3,000 | 1 to 2 min | 10 to 45 min |

UMAP on 100k posts takes several more minutes on the VPS because a fixed seed makes it single-threaded (section 7).

**Rules**

- Open only SSH, 80 and 443 in the VPS firewall.
- Secrets live only in `.env` on the machine that needs them.
- Memory: 100k posts is about 150 MB of vectors and comfortable on 8 GB. Do not go above about 300k posts without a test run.
- After a good run, back up with Qdrant stopped: `tar czf data_backup.tgz data/ qdrant_storage/`. If something breaks before judging, restore and restart instead of rerunning.
- Before submitting, open the public URL from a device outside the team network and click through a bubble, the detail panel and the table view.
- Optional: the Hostinger Connector, API and CLI can deploy and manage the VPS from the editor.

## 11. Build order and definition of done

Stages only share artifacts, so tracks can run in parallel once M0 is done. M0 includes small fixtures (`tests/fixtures/`: 300 synthetic posts, a `clusters.json` with 8 narratives, an `analyses.jsonl` with 6 successful analyses and 1 failure, and 1 narrative left unanalysed) so analyze, the API and the dashboard can be built before the upstream stages exist. The fixtures cover all concern levels, low confidence, insufficient evidence, failed and unanalysed states. Their labels and concern values are synthetic UI examples, not detector findings.

1. **M0, skeleton:** repo layout, `config.py`, `.env.example`, CLI with all stage names, Dockerfile, Compose with Qdrant, Makefile with the section 10 targets, fixtures. The smoke target works on a fresh clone.
2. **M1, load and embed:** both stages work at `N_POSTS=2000`; the nearest-neighbour check reads sensibly.
3. **M2, cluster and deploy:** cluster and stats work at 2,000 posts; API and table view run; the app is **live on the VPS with a public URL** using smoke data.
4. **M3, analyze:** Verda endpoint deployed and the llm-check target passing, then one cluster end to end, then all selected clusters; enforced rules and their tests.
5. **M4, dashboard:** bubble view, detail panel, debug sections, filters.
6. **M5, full run:** `N_POSTS=100000` on the VPS, backup of `data/` and `qdrant_storage/`, public URL checked from outside.

**Definition of done**

- [ ] `python -m detector all` runs from an empty `data/` without errors.
- [ ] The public URL serves the dashboard with at least 10 analysed clusters from the full run.
- [ ] Each analysed cluster shows label, summary, explanation, concern level, confidence, and evidence ids that exist in its sample.
- [ ] A failed analysis shows as `failed` with its raw response, not as a server error.
- [ ] Rerunning `analyze` with unchanged inputs makes zero LLM calls.
- [ ] Unit tests for `stats.py` and the enforced rules pass.
- [ ] A backup of `data/` and `qdrant_storage/` exists.

## 12. Rules for coding agents and what is out of the MVP

**Rules for coding agents**

- Read this doc first and stay inside the contracts in sections 3 to 9. A change to an artifact schema, endpoint or environment variable updates this doc in the same PR.
- Stages communicate only through artifacts and Qdrant. No cross-stage imports. Shared code lives only in `config.py`, `io.py`, `schema.py`, `stats.py`, `rules.py`, `prompts.py`, `llm.py`, `text.py`, `embedding.py` and `projection.py`. `chat.py` belongs to the serve stage.
- Every stage has a `run()` function and a CLI entry, is idempotent, logs counts and timings, and fails with a clear message when an input artifact is missing.
- Put testable logic in pure functions: stats, sampling, rule enforcement, JSON extraction. Tests use no network; LLM tests use recorded responses from `tests/fixtures/`.
- Settings come only from environment variables through `config.py`. No hard-coded keys, URLs or paths. Never log or commit `LLM_API_KEY`.
- Use `SEED` for every random choice.
- Pin exact dependency versions in `pyproject.toml` after the first successful install, and pin the Docker image tags.
- Prompts live in `prompts.py` with a version string. Changing them bumps the version, which changes `input_hash`. Pure rules in `rules.py` are shared with the API.
- Do not add services, frameworks or features that this doc does not list without team agreement.

**Known limits (say them in the demo)**

- The dataset is a snapshot of ordinary posts, so it may contain little real coordination. Clusters flagged here show the method working, not a finding about the platform.
- The behavior stats are simple heuristics with fixed thresholds, and the LLM's descriptions have not been validated against labels yet.
- Nemotron's official language support is limited to six languages.

**Out of the MVP, in priority order**

1. **Evaluation:** a hand-labeled set of clusters (coordinated, organic, legitimate lookalike), precision and recall, a panel of legitimate lookalikes, a mirror test for political symmetry and a prompt-injection test. This is what the judges' false-positive criterion asks for.
2. **Account and source features:** account age, posting rate, duplicate ratio and link-domain reputation, feeding a source score.
3. **Live ingestion** from the Bluesky Jetstream and rolling windows.
4. **Better naming and stability** of clusters across reruns.
