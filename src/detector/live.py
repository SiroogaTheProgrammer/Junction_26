"""Bounded in-memory processing of a streamed dataset, shared by live viewers."""
import asyncio
import fcntl
import logging
import threading
from collections import deque
from itertools import islice
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.cluster import HDBSCAN

from detector.analyze import _analyze_one
from detector.chat import Placer
from detector.config import Settings
from detector.embedding import choose_device, clean_for_embedding, encode, load_model
from detector.llm import create_client
from detector.load import prepare_frame
from detector.io import read_json, write_json
from detector.patterns import RollingPatterns
from detector.rules import behavior_anomaly, review_priority
from detector.stats import baseline_for, build_baseline, cluster_stats, describe_signals

logger = logging.getLogger(__name__)


class LiveCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1, le=1)
    dataset: str
    requested_revision: str
    resolved_revision: str = Field(min_length=1)
    source_state: dict[str, Any] | None = None
    scanned: int = Field(default=0, ge=0)
    processed: int = Field(default=0, ge=0)
    group_id: int = Field(default=0, ge=0)
    complete: bool = False

    @model_validator(mode="after")
    def validate_progress(self) -> "LiveCheckpoint":
        if self.processed > self.scanned:
            raise ValueError("processed posts cannot exceed scanned source rows")
        if self.scanned and self.source_state is None:
            raise ValueError("source state is required after scanning rows")
        return self


def _resolve_revision(settings: Settings) -> str:
    info = HfApi().dataset_info(settings.live_dataset, revision=settings.live_dataset_revision)
    if not info.sha:
        raise ValueError("Dataset revision could not be resolved to a commit")
    return info.sha


def _source(settings: Settings, revision: str) -> Any:
    dataset = load_dataset(settings.live_dataset, revision=revision, streaming=True)
    if "train" not in dataset:
        raise ValueError(f"Dataset {settings.live_dataset} does not contain a train split")
    return dataset["train"]


def _prepare(raw: list[dict], settings: Settings, first_id: int, row_offset: int = 0) -> pd.DataFrame:
    frame, _ = prepare_frame(pd.DataFrame(raw), settings.min_post_chars, row_offset)
    frame = frame.drop_duplicates("uri").reset_index(drop=True)
    frame.insert(0, "post_id", range(first_id, first_id + len(frame)))
    return frame


def _group(
    frame: pd.DataFrame, model: Any, settings: Settings, next_id: int, vectors: np.ndarray | None = None,
) -> list[tuple[dict, pd.DataFrame]]:
    if len(frame) < settings.live_min_cluster_size:
        return []
    if vectors is None:
        texts = ["query: " + clean_for_embedding(text) for text in frame["text"]]
        vectors = encode(model, texts, settings.embed_batch, progress=False)
    if not np.isfinite(vectors).all():
        raise ValueError("The live embedding model returned non-finite vectors")
    labels = HDBSCAN(
        min_cluster_size=settings.live_min_cluster_size,
        min_samples=min(settings.live_min_samples, len(frame) - 1),
        metric="euclidean",
        cluster_selection_method="leaf",
    ).fit_predict(vectors)
    groups = []
    for label in sorted(set(labels) - {-1}):
        indexes = np.flatnonzero(labels == label)
        posts = frame.iloc[indexes].copy()
        centroid = vectors[indexes].mean(axis=0)
        representatives = indexes[np.argsort(-(vectors[indexes] @ centroid), kind="stable")[:10]]
        stats = cluster_stats(posts, settings)
        groups.append(({
            **stats, "stats": stats, "cluster_id": next_id + len(groups),
            "representative_post_ids": frame.iloc[representatives]["post_id"].astype(int).tolist(),
        }, posts))
    return groups


def _result(
    group: dict, posts: pd.DataFrame, record: dict, baseline: dict, settings: Settings, scope: str,
) -> dict[str, Any]:
    parsed = record.get("parsed")
    anomaly = behavior_anomaly(group, baseline, settings.anomaly_percentile)
    priority = review_priority(parsed, group, baseline, settings.anomaly_percentile)
    sample_ids = {item["post_id"] for item in record.get("sample", [])}
    sampled = posts.loc[posts["post_id"].isin(sample_ids), ["post_id", "text"]]
    lookup = dict(zip(sampled["post_id"], sampled["text"], strict=True))
    sample = [{
        **item, "text": lookup[item["post_id"]],
    } for item in record.get("sample", []) if item["post_id"] in lookup]
    row = {
        **group, "status": record["status"],
        "label": parsed.get("label") if parsed else None,
        "concern_level": parsed.get("concern_level") if parsed else None,
        "review_priority": priority["score"], "behavior_anomaly": anomaly.flag,
        "anomaly_triggered_by": anomaly.triggered_by,
        "signals": describe_signals(group, baseline), "burstiness_reliable": False,
        "burst_window_min": settings.burst_window_min,
        "burstiness_note": "Burst timing is excluded because this window cannot establish a reliable dataset-wide time baseline.",
        "bucket_baseline": baseline_for(group["size"], baseline),
        "source_scope": scope,
        "discarded": record["status"] == "ok" and priority["score"] < settings.min_review_priority,
    }
    return {"cluster": row, "analysis": {**record, "sample": sample}}


class LiveProcessor:
    """Processes bounded batches with an on-disk cursor and an in-memory window."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.task: asyncio.Task | None = None
        self.batches: deque[tuple[int, list[dict[str, Any]]]] = deque()
        self.retained = 0
        self.processed = 0
        self.scanned = 0
        self.group_id = 0
        self.revision = 0
        self.status = "idle"
        self.error: str | None = None
        self.checkpoint_path = settings.data_path / "live_progress.json"
        self.resumed_from = 0
        self.last_batch: dict[str, int | float | None] | None = None
        self.patterns = RollingPatterns()
        self.pattern_ids: dict[str, int] = {}
        self.pattern_results: dict[str, dict[str, Any]] = {}
        self.index: deque[tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str], dict[int, dict]]] = deque()
        self.encoder_model: Any | None = None
        self.encoder_lock = threading.Lock()

    def _encode(self, texts: list[str]) -> np.ndarray:
        with self.encoder_lock:
            if self.encoder_model is None:
                raise RuntimeError("The live embedding model is not ready")
            return encode(
                self.encoder_model, ["query: " + clean_for_embedding(text) for text in texts],
                self.settings.embed_batch, progress=False,
            )

    def chat_context(self) -> tuple[Placer, dict[int, dict], dict[int, str]]:
        batches = list(self.index)
        if not batches:
            raise RuntimeError("No processed posts in the live window yet. Wait for a batch to finish.")
        vectors = np.concatenate([batch[0] for batch in batches])
        post_ids = np.concatenate([batch[1] for batch in batches])
        cluster_ids = np.concatenate([batch[2] for batch in batches])
        texts = {key: value for batch in batches for key, value in batch[3].items()}
        clouds = {key: value for batch in batches for key, value in batch[4].items()}
        placer = Placer(self.settings, vectors, None, post_ids, cluster_ids,
                        encoder=lambda text: self._encode([text])[0])
        return placer, clouds, texts

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self.run())

    def snapshot(self) -> dict[str, Any]:
        groups = [group for _, batch in self.batches for group in batch] + list(self.pattern_results.values())
        visible = sorted(
            (group["cluster"] for group in groups if not group["cluster"]["discarded"]),
            key=lambda row: (-row["review_priority"], -row["size"], row["cluster_id"]),
        )[:60]
        return {
            "status": self.status, "error": self.error, "revision": self.revision,
            "scanned": self.scanned, "processed": self.processed, "retained": self.retained,
            "window_limit": self.settings.live_window_posts,
            "dataset": self.settings.live_dataset, "resumed_from": self.resumed_from,
            "batch_size": self.settings.live_batch_size, "last_batch": self.last_batch,
            "groups": visible, "total_groups": len(groups),
            "hidden_groups": sum(group["cluster"]["discarded"] for group in groups),
            "failed_groups": sum(group["cluster"]["status"] == "failed" for group in groups),
        }

    def detail(self, group_id: int) -> dict[str, Any] | None:
        return next((group for group in (
                     [group for _, batch in self.batches for group in batch] + list(self.pattern_results.values()))
                     if group["cluster"]["cluster_id"] == group_id), None)

    def _record_batch(self, started: float, source_rows: int, posts: int, grouped_posts: int) -> None:
        elapsed = perf_counter() - started
        self.last_batch = {
            "source_rows": source_rows, "posts": posts, "seconds": elapsed,
            "grouped_posts": grouped_posts,
            "posts_per_second": posts / elapsed if elapsed > 0 else None,
        }

    async def run(self) -> None:
        self.status = "starting"
        self.revision += 1
        lock = None
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            lock = (self.checkpoint_path.parent / "live_progress.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another process is already running live processing for this data folder") from exc
            if not self.settings.llm_api_key or not self.settings.llm_api_key.get_secret_value().strip():
                raise ValueError("Configure LLM_API_KEY and LLM_BASE_URL for live group analysis")
            saved = read_json(self.checkpoint_path)
            if saved is not None:
                checkpoint = LiveCheckpoint.model_validate(saved)
                if (checkpoint.dataset, checkpoint.requested_revision) != (
                    self.settings.live_dataset, self.settings.live_dataset_revision,
                ):
                    raise ValueError("Live dataset or revision differs from the saved checkpoint; use a separate data folder")
            else:
                resolved = await asyncio.to_thread(_resolve_revision, self.settings)
                checkpoint = LiveCheckpoint(
                    dataset=self.settings.live_dataset, requested_revision=self.settings.live_dataset_revision,
                    resolved_revision=resolved,
                )
                write_json(self.checkpoint_path, checkpoint.model_dump(mode="json"))
            self.scanned, self.processed, self.group_id = checkpoint.scanned, checkpoint.processed, checkpoint.group_id
            self.resumed_from = checkpoint.scanned
            self.revision += 1
            if checkpoint.complete:
                self.status = "complete"
                self.revision += 1
                return
            dataset = await asyncio.to_thread(_source, self.settings, checkpoint.resolved_revision)
            if checkpoint.source_state is not None:
                dataset.load_state_dict(checkpoint.source_state)
            source = iter(dataset)
            device = await asyncio.to_thread(choose_device, self.settings.embed_device)
            model = await asyncio.to_thread(load_model, self.settings, device)
            self.encoder_model = model
            self.status = "running"
            self.revision += 1
            async with create_client(self.settings) as client:
                while True:
                    batch_started = perf_counter()
                    raw = await asyncio.to_thread(lambda: list(islice(source, self.settings.live_batch_size)))
                    if not raw:
                        checkpoint.complete = True
                        write_json(self.checkpoint_path, checkpoint.model_dump(mode="json"))
                        self.status = "complete"
                        self.revision += 1
                        break
                    frame = await asyncio.to_thread(_prepare, raw, self.settings, self.processed, self.scanned)
                    self.scanned += len(raw)
                    if frame.empty:
                        checkpoint.source_state = dataset.state_dict()
                        checkpoint.scanned = self.scanned
                        write_json(self.checkpoint_path, checkpoint.model_dump(mode="json"))
                        self._record_batch(batch_started, len(raw), 0, 0)
                        self.revision += 1
                        continue
                    vectors = await asyncio.to_thread(self._encode, frame["text"].astype(str).tolist())
                    if not np.isfinite(vectors).all():
                        raise ValueError("The live embedding model returned non-finite vectors")
                    groups = await asyncio.to_thread(_group, frame, model, self.settings, self.group_id, vectors)
                    self.group_id += len(groups)
                    # Individual batches do not provide a reliable dataset-wide time baseline.
                    baseline = build_baseline([group for group, _ in groups], self.settings, burstiness_reliable=False)
                    semaphore = asyncio.Semaphore(self.settings.llm_concurrency)
                    records = await asyncio.gather(*(
                        _analyze_one(semaphore, client, self.settings, group, posts, baseline, False, {})
                        for group, posts in groups
                    ))
                    results = [
                        _result(group, posts, record, baseline, self.settings, "live batch")
                        for (group, posts), record in zip(groups, records, strict=True)
                    ]
                    count = len(frame)
                    self.batches.append((count, results))
                    self.retained += count
                    self.processed += count
                    changed = self.patterns.add(frame)
                    while self.retained > self.settings.live_window_posts:
                        removed, _ = self.batches.popleft()
                        self.retained -= removed
                        changed.update(self.patterns.evict())
                    candidates = []
                    for key in sorted(changed):
                        posts = self.patterns.candidate(key)
                        if posts is None:
                            self.pattern_results.pop(key, None)
                            self.pattern_ids.pop(key, None)
                            continue
                        if key not in self.pattern_ids:
                            self.pattern_ids[key] = self.group_id
                            self.group_id += 1
                        stats = await asyncio.to_thread(cluster_stats, posts, self.settings)
                        group = {
                            **stats, "stats": stats, "cluster_id": self.pattern_ids[key],
                            "representative_post_ids": posts["post_id"].head(10).astype(int).tolist(),
                        }
                        candidates.append((key, group, posts))
                    pattern_records = await asyncio.gather(*(
                        _analyze_one(semaphore, client, self.settings, group, posts, baseline, False, {})
                        for _, group, posts in candidates
                    ))
                    for (key, group, posts), record in zip(candidates, pattern_records, strict=True):
                        self.pattern_results[key] = _result(
                            group, posts, record, baseline, self.settings, "repeated text across live batches",
                        )
                    checkpoint.source_state = dataset.state_dict()
                    checkpoint.scanned, checkpoint.processed, checkpoint.group_id = self.scanned, self.processed, self.group_id
                    write_json(self.checkpoint_path, checkpoint.model_dump(mode="json"))
                    ids = frame["post_id"].to_numpy(dtype=np.int64)
                    membership = np.full(len(frame), -1, dtype=np.int64)
                    positions = {post_id: index for index, post_id in enumerate(ids)}
                    for group, posts in groups:
                        for post_id in posts["post_id"]:
                            membership[positions[int(post_id)]] = group["cluster_id"]
                    texts = dict(zip(ids.tolist(), frame["text"].astype(str), strict=True))
                    clouds = {
                        item["cluster"]["cluster_id"]: {
                            **item["cluster"],
                            "summary": (item["analysis"].get("parsed") or {}).get("summary"),
                        } for item in results
                    }
                    self.index.append((vectors, ids, membership, texts, clouds))
                    while sum(len(batch[1]) for batch in self.index) > self.settings.live_window_posts:
                        self.index.popleft()
                    self._record_batch(batch_started, len(raw), count, sum(len(posts) for _, posts in groups))
                    self.revision += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Live dataset processing stopped")
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            self.revision += 1
        finally:
            if lock is not None:
                lock.close()

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
