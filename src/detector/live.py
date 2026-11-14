"""Bounded in-memory processing of a streamed dataset, shared by live viewers."""
import asyncio
import logging
from collections import deque
from itertools import islice
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import HDBSCAN

from detector.analyze import _analyze_one
from detector.config import Settings
from detector.embedding import choose_device, clean_for_embedding, encode, load_model
from detector.llm import create_client
from detector.load import prepare_frame
from detector.rules import behavior_anomaly, review_priority
from detector.stats import baseline_for, build_baseline, cluster_stats, describe_signals

logger = logging.getLogger(__name__)


def _source(settings: Settings) -> Any:
    dataset = load_dataset(settings.live_dataset, streaming=True)
    if "train" not in dataset:
        raise ValueError(f"Dataset {settings.live_dataset} does not contain a train split")
    return iter(dataset["train"])


def _prepare(raw: list[dict], settings: Settings, first_id: int) -> pd.DataFrame:
    frame, _ = prepare_frame(pd.DataFrame(raw), settings.min_post_chars)
    frame = frame.drop_duplicates("uri").reset_index(drop=True)
    frame.insert(0, "post_id", range(first_id, first_id + len(frame)))
    return frame


def _group(frame: pd.DataFrame, model: Any, settings: Settings, next_id: int) -> list[tuple[dict, pd.DataFrame]]:
    if len(frame) < settings.hdbscan_min_cluster_size:
        return []
    texts = ["query: " + clean_for_embedding(text) for text in frame["text"]]
    vectors = encode(model, texts, settings.embed_batch, progress=False)
    if not np.isfinite(vectors).all():
        raise ValueError("The live embedding model returned non-finite vectors")
    labels = HDBSCAN(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=min(settings.hdbscan_min_samples, len(frame) - 1),
        metric="euclidean",
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


class LiveProcessor:
    """Processes one bounded batch at a time; no live artifacts are written to disk."""

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

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self.run())

    def snapshot(self) -> dict[str, Any]:
        groups = [group for _, batch in self.batches for group in batch]
        visible = sorted(
            (group["cluster"] for group in groups if not group["cluster"]["discarded"]),
            key=lambda row: (-row["review_priority"], -row["size"], row["cluster_id"]),
        )[:60]
        return {
            "status": self.status, "error": self.error, "revision": self.revision,
            "scanned": self.scanned, "processed": self.processed, "retained": self.retained,
            "window_limit": self.settings.live_window_posts,
            "groups": visible, "total_groups": len(groups),
            "failed_groups": sum(group["cluster"]["status"] == "failed" for group in groups),
        }

    def detail(self, group_id: int) -> dict[str, Any] | None:
        return next((group for _, batch in self.batches for group in batch
                     if group["cluster"]["cluster_id"] == group_id), None)

    async def run(self) -> None:
        self.status = "starting"
        self.revision += 1
        try:
            if not self.settings.llm_api_key or not self.settings.llm_api_key.get_secret_value().strip():
                raise ValueError("Configure LLM_API_KEY and LLM_BASE_URL for live group analysis")
            source = await asyncio.to_thread(_source, self.settings)
            device = await asyncio.to_thread(choose_device, self.settings.embed_device)
            model = await asyncio.to_thread(load_model, self.settings, device)
            self.status = "running"
            self.revision += 1
            async with create_client(self.settings) as client:
                while True:
                    raw = await asyncio.to_thread(lambda: list(islice(source, self.settings.live_batch_size)))
                    if not raw:
                        self.status = "complete"
                        self.revision += 1
                        break
                    frame = await asyncio.to_thread(_prepare, raw, self.settings, self.processed)
                    self.scanned += len(raw)
                    if frame.empty:
                        self.revision += 1
                        continue
                    groups = await asyncio.to_thread(_group, frame, model, self.settings, self.group_id)
                    self.group_id += len(groups)
                    # Individual batches do not provide a reliable dataset-wide time baseline.
                    baseline = build_baseline([group for group, _ in groups], self.settings, burstiness_reliable=False)
                    semaphore = asyncio.Semaphore(self.settings.llm_concurrency)
                    records = await asyncio.gather(*(
                        _analyze_one(semaphore, client, self.settings, group, posts, baseline, False, {})
                        for group, posts in groups
                    ))
                    results = []
                    for (group, posts), record in zip(groups, records, strict=True):
                        parsed = record.get("parsed")
                        anomaly = behavior_anomaly(group, baseline, self.settings.anomaly_percentile)
                        priority = review_priority(parsed, group, baseline, self.settings.anomaly_percentile)
                        lookup = posts.set_index("post_id").to_dict(orient="index")
                        sample = [{
                            **item, "text": lookup[item["post_id"]]["text"],
                        } for item in record.get("sample", []) if item["post_id"] in lookup]
                        row = {
                            **group, "status": record["status"],
                            "label": parsed.get("label") if parsed else None,
                            "concern_level": parsed.get("concern_level") if parsed else None,
                            "review_priority": priority["score"], "behavior_anomaly": anomaly.flag,
                            "anomaly_triggered_by": anomaly.triggered_by,
                            "signals": describe_signals(group, baseline), "burstiness_reliable": False,
                            "burst_window_min": self.settings.burst_window_min,
                            "burstiness_note": "Burst timing is excluded because this batch cannot establish a reliable dataset-wide time baseline.",
                            "bucket_baseline": baseline_for(group["size"], baseline),
                            "source_scope": "live batch",
                            "discarded": record["status"] == "ok" and priority["score"] < self.settings.min_review_priority,
                        }
                        results.append({"cluster": row, "analysis": {**record, "sample": sample}})
                    count = len(frame)
                    self.batches.append((count, results))
                    self.retained += count
                    self.processed += count
                    while self.retained > self.settings.live_window_posts:
                        removed, _ = self.batches.popleft()
                        self.retained -= removed
                    self.revision += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Live dataset processing stopped")
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            self.revision += 1

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
