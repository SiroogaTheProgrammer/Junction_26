import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
from datasets import Dataset

from detector import cluster, live
from detector.config import Settings
from detector.patterns import RollingPatterns
from detector.stats import cluster_stats
from tests.test_analyze_rules import _analysis


def _frame(start, texts, authors=None):
    return pd.DataFrame({
        "post_id": range(start, start + len(texts)),
        "text": texts,
        "author": authors or [f"author-{i}" for i in range(start, start + len(texts))],
        "created_at": pd.date_range("2026-01-01", periods=len(texts), freq="min", tz="UTC"),
    })


def test_rolling_candidate_requires_multiple_batches_and_accounts_and_expires():
    patterns = RollingPatterns()
    text = "An identical statement repeated by several separate accounts"
    key = next(iter(patterns.add(_frame(0, [text, text], ["a", "b"]))))
    assert patterns.candidate(key) is None
    patterns.add(_frame(2, [text], ["c"]))
    assert len(patterns.candidate(key)) == 3
    patterns.evict()
    assert patterns.candidate(key) is None
    patterns.evict()
    assert not patterns.posts
    patterns.add(_frame(3, [text, text], ["a", "a"]))
    patterns.add(_frame(5, [text], ["b"]))
    assert patterns.candidate(key) is None


def test_subdivision_retains_parent_topic_and_all_assigned_posts(monkeypatch):
    class Estimator:
        def __init__(self, **kwargs):
            assert kwargs["cluster_selection_method"] == "leaf"

        def fit_predict(self, values):
            return np.array([0] * 10 + [1] * 10 + [-1] * 5)

    monkeypatch.setattr(cluster, "HDBSCAN", Estimator)
    labels = np.array([7] * 25 + [-1] * 2)
    result, parents = cluster._subdivide_topics(np.zeros((27, 4)), labels, Settings(_env_file=None))
    assert list(result) == [0] * 10 + [1] * 10 + [2] * 5 + [-1] * 2
    assert parents == {0: (7, "focused subgroup"), 1: (7, "focused subgroup"), 2: (7, "mixed remainder")}


def test_cross_batch_repetition_is_reviewed_separately_from_broad_topic(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        live_batch_size=25, live_window_posts=50,
    )
    repeated = "A repeated call to action in the middle of otherwise varied public discussion"
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": repeated if i in (0, 1, 25, 26) else f"Ordinary public discussion post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(50)
    ]
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "fixed")
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(source).to_iterable_dataset())
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())

    monkeypatch.setattr(live, "encode", lambda _model, texts, _batch, progress=False:
                        np.ones((len(texts), 384), dtype=np.float32))

    def broad_topic(frame, model, config, next_id, vectors):
        stats = cluster_stats(frame, config)
        return [({**stats, "stats": stats, "cluster_id": next_id, "representative_post_ids": []}, frame)]

    async def assess(semaphore, client, config, group, posts, baseline, force, existing):
        parsed = _analysis("high" if len(posts) == 4 else "none").model_dump(mode="json")
        return {"cluster_id": group["cluster_id"], "status": "ok", "parsed": parsed,
                "sample": [{"pid": f"p{i + 1}", "post_id": int(post_id)}
                           for i, post_id in enumerate(posts["post_id"])]}

    monkeypatch.setattr(live, "_group", broad_topic)
    monkeypatch.setattr(live, "_analyze_one", assess)
    processor = live.LiveProcessor(settings)
    asyncio.run(processor.run())
    assert processor.status == "complete"
    candidates = [row for row in processor.snapshot()["groups"]
                  if row["source_scope"] == "repeated text across live batches"]
    assert len(candidates) == 1
    assert candidates[0]["size"] == 4
    assert candidates[0]["concern_level"] == "high"
    assert len(processor.detail(candidates[0]["cluster_id"])["analysis"]["sample"]) == 4
