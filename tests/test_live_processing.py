import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from detector import analyze, live
from detector import api
from detector.config import Settings
from detector.stats import cluster_stats
from fastapi.testclient import TestClient
from tests.test_analyze_rules import _analysis


def test_stream_processes_beyond_first_window_without_writing_live_files(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        llm_base_url="https://test.invalid/v1", live_batch_size=25, live_window_posts=50,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i), "text": f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(80)
    ]
    monkeypatch.setattr(live, "_source", lambda _: iter(source))
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())

    def group(frame, model, config, next_id):
        stats = cluster_stats(frame, config)
        return [({**stats, "stats": stats, "cluster_id": next_id,
                  "representative_post_ids": frame["post_id"].head(10).tolist()}, frame)]

    monkeypatch.setattr(live, "_group", group)
    client = AsyncMock()
    monkeypatch.setattr(live, "create_client", lambda _: client)

    async def analyze(semaphore, client, config, cluster, posts, baseline, force, existing):
        return {"cluster_id": cluster["cluster_id"], "status": "ok",
                "parsed": _analysis().model_dump(mode="json"),
                "sample": [{"pid": f"p{i + 1}", "post_id": int(post_id)}
                           for i, post_id in enumerate(posts["post_id"])]}

    monkeypatch.setattr(live, "_analyze_one", analyze)
    processor = live.LiveProcessor(settings)
    asyncio.run(processor.run())
    snapshot = processor.snapshot()
    assert snapshot["status"] == "complete"
    assert (snapshot["scanned"], snapshot["processed"], snapshot["retained"]) == (80, 80, 30)
    assert snapshot["window_limit"] == 50
    assert processor.detail(0) is None and processor.detail(1) is None
    assert processor.detail(2)["analysis"]["sample"][0]["post_id"] == 50
    assert processor.detail(3)["cluster"]["size"] == 5
    assert list(tmp_path.iterdir()) == []


def test_live_embedding_and_hdbscan_group_new_posts(monkeypatch):
    frame = pd.DataFrame({
        "post_id": range(30), "text": [f"Example content number {i}" for i in range(30)],
        "author": [f"author-{i}" for i in range(30)],
        "created_at": pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC"),
    })
    vectors = np.zeros((30, 384), dtype=np.float32)
    vectors[:, 0] = 1
    monkeypatch.setattr(live, "encode", lambda *_args, **_kwargs: vectors)

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, values):
            assert values.shape == (30, 384)
            return np.array([0] * 25 + [-1] * 5)

    monkeypatch.setattr(live, "HDBSCAN", Estimator)
    groups = live._group(frame, object(), Settings(_env_file=None), 123)
    assert len(groups) == 1
    group, posts = groups[0]
    assert (group["cluster_id"], group["size"], len(posts)) == (123, 25, 25)
    assert len(group["representative_post_ids"]) == 10


def test_stream_passes_10000_posts_and_never_retains_more_than_25000(monkeypatch):
    settings = Settings(
        _env_file=None, llm_api_key="test-key", llm_base_url="https://test.invalid/v1",
        live_batch_size=500, live_window_posts=25000,
    )
    source = (
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(25300)
    )
    monkeypatch.setattr(live, "_source", lambda _: iter(source))
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "_group", lambda *_: [])
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())
    processor = live.LiveProcessor(settings)

    async def verify():
        task = asyncio.create_task(processor.run())
        while not task.done():
            assert processor.retained <= 25000
            await asyncio.sleep(0)
        await task

    asyncio.run(verify())
    assert processor.processed == processor.scanned == 25300
    assert processor.retained == 24800
    assert processor.status == "complete"


def test_live_errors_are_visible_and_batch_must_fit_window(monkeypatch):
    with pytest.raises(ValueError, match="LIVE_BATCH_SIZE cannot exceed"):
        Settings(_env_file=None, live_batch_size=50, live_window_posts=25)
    processor = live.LiveProcessor(Settings(_env_file=None))
    asyncio.run(processor.run())
    assert processor.snapshot()["status"] == "error"
    assert "LLM_API_KEY" in processor.snapshot()["error"]


def test_real_hdbscan_groups_separate_synthetic_batches(monkeypatch):
    rng = np.random.default_rng(42)
    vectors = rng.normal(0, 0.002, (60, 384)).astype(np.float32)
    vectors[:30, 0] += 1
    vectors[30:, 1] += 1
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    frame = pd.DataFrame({
        "post_id": range(60), "text": [f"Different test message number {i}" for i in range(60)],
        "author": [f"author-{i}" for i in range(60)],
        "created_at": pd.date_range("2026-01-01", periods=60, freq="min", tz="UTC"),
    })
    monkeypatch.setattr(live, "encode", lambda *_args, **_kwargs: vectors)
    groups = live._group(frame, object(), Settings(_env_file=None, hdbscan_min_cluster_size=10), 0)
    assert sorted(group["size"] for group, _ in groups) == [30, 30]


def test_live_snapshot_and_details_api_without_offline_artifacts(tmp_path, monkeypatch):
    group = {"cluster": {"cluster_id": 5}, "analysis": {"sample": [{"text": "example"}]}}

    class FakeProcessor:
        def __init__(self, settings):
            self.started = False

        def start(self):
            self.started = True

        def snapshot(self):
            return {"revision": 1, "status": "complete", "processed": 25001}

        def detail(self, group_id):
            return group if group_id == 5 else None

        async def close(self):
            pass

    monkeypatch.setattr(live, "LiveProcessor", FakeProcessor)
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path))
    with TestClient(api.app) as client:
        response = client.get("/api/live/processed")
        assert '"processed": 25001' in response.text
        assert api.app.state.live.started
        assert client.get("/api/live/clusters/5").json() == group
        assert client.get("/api/live/clusters/5/posts").json()["posts"][0]["text"] == "example"
        assert client.get("/api/live/clusters/6").status_code == 404


def test_real_live_analyzer_produces_viewable_sample_and_group(monkeypatch):
    settings = Settings(
        _env_file=None, llm_api_key="test-key", llm_base_url="https://test.invalid/v1",
        live_batch_size=25, live_window_posts=50,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(25)
    ]
    monkeypatch.setattr(live, "_source", lambda _: iter(source))
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())

    def group(frame, model, config, next_id):
        stats = cluster_stats(frame, config)
        return [({**stats, "stats": stats, "cluster_id": next_id,
                  "representative_post_ids": frame["post_id"].head(10).tolist()}, frame)]

    monkeypatch.setattr(live, "_group", group)
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())
    parsed = _analysis()
    request = AsyncMock(return_value=(parsed.model_dump_json(), parsed, 1))
    monkeypatch.setattr(analyze, "_request_analysis", request)
    processor = live.LiveProcessor(settings)
    asyncio.run(processor.run())
    assert processor.status == "complete"
    group = processor.detail(0)
    assert group["analysis"]["status"] == "ok"
    assert len(group["analysis"]["sample"]) == 25
    assert group["cluster"]["label"] == parsed.label
    assert group["cluster"]["burstiness_reliable"] is False
    assert "<post id=" in request.await_args.args[2]
