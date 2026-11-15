import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from detector import analyze, live
from detector import api
from detector.config import Settings
from detector.stats import build_baseline, cluster_stats
from fastapi.testclient import TestClient
from tests.test_analyze_rules import _analysis


@pytest.fixture(autouse=True)
def mock_stream_embeddings(monkeypatch):
    def fake_encode(_model, texts, _batch_size, progress=False):
        return np.ones((len(texts), 384), dtype=np.float32)
    monkeypatch.setattr(live, "encode", fake_encode)


def test_stream_processes_beyond_first_window_and_only_writes_cursor(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        llm_base_url="https://test.invalid/v1", live_batch_size=25, live_window_posts=50,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i), "text": f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(80)
    ]
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "test-revision")
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(source).to_iterable_dataset())
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())

    def group(frame, model, config, next_id, vectors):
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
    placer, clouds, texts = processor.chat_context()
    assert set(placer.post_ids) == set(range(50, 80))
    assert set(clouds) == {2, 3}
    assert set(texts) == set(range(50, 80))
    assert len(placer.embeddings) == processor.retained
    assert {path.name for path in tmp_path.iterdir()} == {"live_progress.json", "live_progress.lock"}
    checkpoint = live.LiveCheckpoint.model_validate(live.read_json(tmp_path / "live_progress.json"))
    assert checkpoint.complete and checkpoint.processed == 80


@pytest.mark.parametrize("eligible", [0, 20, 25])
def test_last_completed_batch_speed_measures_eligible_posts(tmp_path, monkeypatch, eligible):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        live_batch_size=25, live_window_posts=50,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": f"A sufficiently long example post number {i}" if i < eligible else "short",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(25)
    ]
    times = iter([10.0, 14.0, 20.0])
    monkeypatch.setattr(live, "perf_counter", lambda: next(times))
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "test-revision")
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(source).to_iterable_dataset())
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "_group", lambda *_: [])
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())

    processor = live.LiveProcessor(settings)
    assert processor.snapshot()["last_batch"] is None
    asyncio.run(processor.run())
    snapshot = processor.snapshot()
    assert snapshot["status"] == "complete"
    assert snapshot["batch_size"] == 25
    assert snapshot["last_batch"] == {
        "source_rows": 25, "posts": eligible, "seconds": 4.0,
        "grouped_posts": 0, "posts_per_second": eligible / 4,
    }


def test_batch_speed_unavailable_for_zero_elapsed_time(tmp_path, monkeypatch):
    processor = live.LiveProcessor(Settings(_env_file=None, data_dir=tmp_path))
    monkeypatch.setattr(live, "perf_counter", lambda: 10.0)
    processor._record_batch(10.0, 25, 20, 0)
    assert processor.snapshot()["last_batch"]["posts_per_second"] is None


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


def test_live_clustering_finds_smaller_groups_without_forcing_noise(monkeypatch):
    rng = np.random.default_rng(42)
    vectors = rng.normal(0, .08, (100, 64))
    vectors[:15, 0] += 3
    vectors[15:30, 1] += 3
    vectors[30:] = rng.normal(0, 1, (70, 64))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    frame = pd.DataFrame({
        "post_id": range(100), "text": [f"Example content number {i}" for i in range(100)],
        "author": [f"author-{i}" for i in range(100)],
        "created_at": pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC"),
    })
    monkeypatch.setattr(live, "encode", lambda *_args, **_kwargs: vectors)
    old = Settings(_env_file=None, live_min_cluster_size=25, live_min_samples=5)
    assert live._group(frame, object(), old, 0) == []
    groups = live._group(frame, object(), Settings(_env_file=None), 0)
    assert len(groups) == 2
    assert all(group["size"] >= 10 for group, _ in groups)
    assert sum(group["size"] for group, _ in groups) < len(frame)


def test_snapshot_distinguishes_hidden_groups_from_no_groups(tmp_path):
    processor = live.LiveProcessor(Settings(_env_file=None, data_dir=tmp_path))
    assert processor.snapshot()["total_groups"] == 0
    assert processor.snapshot()["hidden_groups"] == 0
    processor.batches.append((10, [{"cluster": {
        "cluster_id": 0, "discarded": True, "status": "ok",
    }}]))
    snapshot = processor.snapshot()
    assert snapshot["total_groups"] == snapshot["hidden_groups"] == 1
    assert snapshot["groups"] == []


def test_result_preserves_order_and_metadata_for_sampled_posts():
    settings = Settings(_env_file=None)
    posts = pd.DataFrame({
        "post_id": [1, 2, 3], "text": ["first", "second", "third"],
        "author": ["a", "b", "c"],
        "created_at": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
    })
    stats = cluster_stats(posts, settings)
    group = {**stats, "stats": stats, "cluster_id": 0}
    baseline = build_baseline([group], settings, burstiness_reliable=False)
    record = {
        "status": "ok", "parsed": _analysis().model_dump(mode="json"),
        "sample": [{"post_id": 3, "pid": "p1"}, {"post_id": 99, "pid": "missing"},
                   {"post_id": 1, "pid": "p2"}],
    }
    result = live._result(group, posts, record, baseline, settings, "live batch")
    assert result["analysis"]["sample"] == [
        {"post_id": 3, "pid": "p1", "text": "third"},
        {"post_id": 1, "pid": "p2", "text": "first"},
    ]


def test_live_chat_uses_window_without_offline_artifacts(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, super_llm_api_key="test-key",
        super_llm_base_url="https://test.invalid/v1", chat_neighbors=4,
    )
    monkeypatch.setattr(api, "settings", settings)
    monkeypatch.setattr(api, "create_super_client", lambda _: AsyncMock())
    processor = live.LiveProcessor(settings)
    processor.encoder_model = object()
    vectors = np.zeros((4, 384), dtype=np.float32)
    vectors[:, 0] = 1
    monkeypatch.setattr(live, "encode", lambda *_args, **_kwargs: vectors[:1])

    async def fake_chat(**kwargs):
        vector = kwargs["placer"].embed("query")
        placement = kwargs["placer"].place(vector)
        return {
            "placement": placement,
            "texts": [kwargs["post_text"](row["post_id"]) for row in placement["neighbors"]],
            "labels": [row["label"] for row in kwargs["clouds"].values()],
        }

    monkeypatch.setattr(api, "run_chat", fake_chat)
    with TestClient(api.app) as client:
        api.app.state.live = processor
        assert client.get("/api/live/chat/status").json()["chat"]["available"]
        response = client.post("/api/live/chat", json={"message": "Test"})
        assert response.status_code == 503
        assert "No processed posts" in response.json()["detail"]
        processor.index.append((
            vectors, np.array([4, 5, 6, 7]), np.array([11, 11, -1, -1]),
            {4: "live four", 5: "live five", 6: "live six", 7: "live seven"},
            {11: {"label": "Live group"}},
        ))
        response = client.post("/api/live/chat", json={"message": "Test"})
        assert response.status_code == 200
        body = response.json()
        assert body["placement"]["nearest"]["cluster_id"] == 11
        assert body["placement"]["coordinates"] is None
        assert set(body["texts"]) == {"live four", "live five", "live six", "live seven"}
        assert body["labels"] == ["Live group"]
        assert client.get("/api/live/chat/status").json()["window_posts"] == 4
        assert api.app.state.dashboard is None


def test_stream_passes_50000_posts_and_never_retains_more_than_50000(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, llm_api_key="test-key", llm_base_url="https://test.invalid/v1",
        live_batch_size=500, data_dir=tmp_path,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(50300)
    ]
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "test-revision")
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(source).to_iterable_dataset())
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "_group", lambda *_: [])
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())
    processor = live.LiveProcessor(settings)

    async def verify():
        task = asyncio.create_task(processor.run())
        while not task.done():
            assert processor.retained <= 50000
            await asyncio.sleep(0)
        await task

    asyncio.run(verify())
    assert processor.processed == processor.scanned == 50300
    assert processor.retained == 49800
    assert processor.status == "complete"


def test_restart_resumes_after_completed_batch_not_dataset_start(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        llm_base_url="https://test.invalid/v1", live_batch_size=25, live_window_posts=50,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(75)
    ]
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "fixed-revision")
    revisions = []

    def open_source(_settings, revision):
        revisions.append(revision)
        return Dataset.from_list(source).to_iterable_dataset()

    monkeypatch.setattr(live, "_source", open_source)
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())
    seen = []

    def group(frame, *_):
        first = int(frame["post_id"].iloc[0])
        seen.append((first, frame["uri"].iloc[0]))
        if len(seen) == 2:
            raise RuntimeError("simulated interruption")
        return []

    monkeypatch.setattr(live, "_group", group)
    first = live.LiveProcessor(settings)
    asyncio.run(first.run())
    assert first.status == "error"
    saved = live.read_json(tmp_path / "live_progress.json")
    assert (saved["scanned"], saved["processed"], saved["complete"]) == (25, 25, False)
    second = live.LiveProcessor(settings)
    asyncio.run(second.run())
    assert second.status == "complete"
    assert seen == [
        (0, "at://did:example:0/app.bsky.feed.post/0"),
        (25, "at://did:example:25/app.bsky.feed.post/25"),
        (25, "at://did:example:25/app.bsky.feed.post/25"),
        (50, "at://did:example:50/app.bsky.feed.post/50"),
    ]
    assert (second.scanned, second.processed) == (75, 75)
    assert revisions == ["fixed-revision", "fixed-revision"]
    third = live.LiveProcessor(settings)
    asyncio.run(third.run())
    assert third.status == "complete" and third.processed == 75
    assert len(revisions) == 2


def test_changed_live_dataset_does_not_replace_checkpoint(tmp_path, monkeypatch):
    checkpoint = live.LiveCheckpoint(
        dataset="first", requested_revision="main", resolved_revision="abc", scanned=25, source_state={},
    )
    path = tmp_path / "live_progress.json"
    live.write_json(path, checkpoint.model_dump())
    before = path.read_bytes()
    settings = Settings(_env_file=None, data_dir=tmp_path, llm_api_key="key", live_dataset="second")
    processor = live.LiveProcessor(settings)
    asyncio.run(processor.run())
    assert processor.status == "error"
    assert "differs from the saved checkpoint" in processor.error
    assert path.read_bytes() == before


def test_empty_batch_advances_checkpoint_and_resume(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="key",
        live_batch_size=25, live_window_posts=50,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": "short" if i < 25 else f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(50)
    ]
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "fixed-revision")
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(source).to_iterable_dataset())
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())
    seen = []

    def group(frame, *_):
        seen.append(frame["uri"].iloc[0])
        if len(seen) == 1:
            raise RuntimeError("simulated interruption")
        return []

    monkeypatch.setattr(live, "_group", group)
    first = live.LiveProcessor(settings)
    asyncio.run(first.run())
    saved = live.read_json(tmp_path / "live_progress.json")
    assert (saved["scanned"], saved["processed"]) == (25, 0)
    second = live.LiveProcessor(settings)
    asyncio.run(second.run())
    assert second.status == "complete"
    assert (second.resumed_from, second.scanned, second.processed) == (25, 50, 25)
    assert seen == ["at://did:example:25/app.bsky.feed.post/25"] * 2


@pytest.mark.parametrize("progress", [
    {"scanned": 5, "processed": 6, "source_state": {}},
    {"scanned": 5, "processed": 0, "source_state": None},
])
def test_invalid_checkpoint_is_reported_without_overwrite(tmp_path, progress):
    path = tmp_path / "live_progress.json"
    saved = {"dataset": "example", "requested_revision": "main", "resolved_revision": "abc", **progress}
    live.write_json(path, saved)
    before = path.read_bytes()
    processor = live.LiveProcessor(Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="key", live_dataset="example",
    ))
    asyncio.run(processor.run())
    assert processor.status == "error"
    assert "ValidationError" in processor.error
    assert path.read_bytes() == before


def test_live_checkpoint_lock_prevents_competing_processors(tmp_path):
    path = tmp_path / "live_progress.lock"
    with path.open("a") as lock:
        live.fcntl.flock(lock, live.fcntl.LOCK_EX | live.fcntl.LOCK_NB)
        processor = live.LiveProcessor(Settings(_env_file=None, data_dir=tmp_path, llm_api_key="key"))
        asyncio.run(processor.run())
    assert processor.status == "error"
    assert "Another process is already running" in processor.error
    assert not (tmp_path / "live_progress.json").exists()


def test_live_errors_are_visible_and_batch_must_fit_window(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="LIVE_BATCH_SIZE cannot exceed"):
        Settings(_env_file=None, live_batch_size=50, live_window_posts=25)
    processor = live.LiveProcessor(Settings(_env_file=None, data_dir=tmp_path))
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
    groups = live._group(frame, object(), Settings(_env_file=None, live_min_cluster_size=10), 0)
    assert sorted(group["size"] for group, _ in groups) == [30, 30]


def test_live_snapshot_and_details_api_without_offline_artifacts(tmp_path, monkeypatch):
    group = {"cluster": {"cluster_id": 5}, "analysis": {"sample": [{"text": "example"}]}}

    class FakeProcessor:
        def __init__(self, settings):
            self.started = False
            self.revision = 1

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


def test_live_stream_builds_snapshots_only_when_revision_changes(monkeypatch):
    class Processor:
        revision = 0
        calls = 0

        def start(self):
            pass

        def snapshot(self):
            self.calls += 1
            return {"revision": self.revision, "status": "complete" if self.revision else "running"}

    processor = Processor()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(live=processor)),
        is_disconnected=AsyncMock(return_value=False),
    )
    sleeps = 0

    async def tick(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            processor.revision = 1

    monkeypatch.setattr(api.asyncio, "sleep", tick)

    async def collect():
        response = await api.live_processed(request)
        return [event async for event in response.body_iterator]

    events = asyncio.run(collect())
    assert len(events) == processor.calls == 2
    assert '"status": "running"' in events[0]
    assert '"status": "complete"' in events[1]


def test_real_live_analyzer_produces_viewable_sample_and_group(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, llm_api_key="test-key", llm_base_url="https://test.invalid/v1",
        live_batch_size=25, live_window_posts=50, data_dir=tmp_path,
    )
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": f"A sufficiently long example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(25)
    ]
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "test-revision")
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(source).to_iterable_dataset())
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())

    def group(frame, model, config, next_id, vectors):
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


@pytest.fixture
def pipeline_source(tmp_path, monkeypatch):
    source = [
        {"did": f"did:example:{i}", "rkey": str(i),
         "text": f"A sufficiently long unique example post number {i}",
         "created_at": "2026-01-01T00:00:00Z"}
        for i in range(125)
    ]
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "fixed-revision")
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(source).to_iterable_dataset())
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())

    def group(frame, _model, config, next_id, _vectors):
        stats = cluster_stats(frame, config)
        return [({**stats, "stats": stats, "cluster_id": next_id}, frame)]

    monkeypatch.setattr(live, "_group", group)
    return Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        live_batch_size=25, live_window_posts=50, live_pipeline_batches=4, llm_concurrency=2,
    )


def test_pipeline_overlaps_batches_but_bounds_preparation_and_requests(pipeline_source, monkeypatch):
    prepared = []
    peak = active = 0
    original_group = live._group
    checkpoints = []
    original_write = live.write_json

    def write(path, data):
        original_write(path, data)
        if data["scanned"]:
            checkpoints.append((data["scanned"], data["processed"]))

    monkeypatch.setattr(live, "write_json", write)

    async def verify():
        nonlocal peak, active
        loop = asyncio.get_running_loop()
        four_prepared, two_requests, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        def group(frame, *args):
            result = original_group(frame, *args)
            prepared.append(int(frame["post_id"].iloc[0]))
            if len(prepared) == 4:
                loop.call_soon_threadsafe(four_prepared.set)
            return result

        async def assess(semaphore, _client, _settings, cluster, *_args):
            nonlocal peak, active
            async with semaphore:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    two_requests.set()
                try:
                    await release.wait()
                    return {"status": "ok", "parsed": _analysis("low").model_dump(), "sample": []}
                finally:
                    active -= 1

        monkeypatch.setattr(live, "_group", group)
        monkeypatch.setattr(live, "_analyze_one", assess)
        processor = live.LiveProcessor(pipeline_source)
        task = asyncio.create_task(processor.run())
        try:
            await asyncio.wait_for(four_prepared.wait(), 5)
            await asyncio.wait_for(two_requests.wait(), 5)
            assert prepared == [0, 25, 50, 75]
            assert processor.scanned == processor.processed == 0
            assert peak == 2
        finally:
            release.set()
            await asyncio.wait_for(task, 5)
        assert processor.status == "complete"
        assert processor.processed == processor.scanned == 125
        assert processor.retained == 50
        assert peak == 2 and active == 0

    asyncio.run(verify())
    assert checkpoints == [(25, 25), (50, 50), (75, 75), (100, 100), (125, 125), (125, 125)]


def test_pipeline_commits_in_source_order_when_later_reviews_finish_first(pipeline_source, monkeypatch):
    finished = []
    checkpoints = []
    original_write = live.write_json

    def write(path, data):
        original_write(path, data)
        if data["scanned"]:
            checkpoints.append(data["scanned"])

    monkeypatch.setattr(live, "write_json", write)

    async def verify():
        later_done = asyncio.Event()

        async def assess(semaphore, _client, _settings, cluster, posts, *_args):
            first = int(posts["post_id"].iloc[0])
            async with semaphore:
                if first == 0:
                    await asyncio.wait_for(later_done.wait(), 5)
                else:
                    later_done.set()
                finished.append(first)
                return {"status": "ok", "parsed": _analysis("low").model_dump(), "sample": []}

        monkeypatch.setattr(live, "_analyze_one", assess)
        processor = live.LiveProcessor(pipeline_source)
        await asyncio.wait_for(processor.run(), 5)
        assert processor.status == "complete"
        assert finished[0] != 0
        assert sorted(finished) == [0, 25, 50, 75, 100]
        assert checkpoints == [25, 50, 75, 100, 125, 125]

    asyncio.run(verify())


def test_pipeline_cancellation_keeps_only_committed_cursor_and_resumes(pipeline_source, monkeypatch):
    seen = []
    active = 0
    original_write = live.write_json

    async def verify():
        nonlocal active
        committed, blocked = asyncio.Event(), asyncio.Event()

        def write(path, data):
            original_write(path, data)
            if data["scanned"] == 25:
                committed.set()

        async def assess(semaphore, _client, _settings, cluster, posts, *_args):
            nonlocal active
            first = int(posts["post_id"].iloc[0])
            seen.append(first)
            async with semaphore:
                active += 1
                try:
                    if first:
                        blocked.set()
                        await asyncio.Event().wait()
                    return {"status": "ok", "parsed": _analysis("low").model_dump(), "sample": []}
                finally:
                    active -= 1

        monkeypatch.setattr(live, "write_json", write)
        monkeypatch.setattr(live, "_analyze_one", assess)
        processor = live.LiveProcessor(pipeline_source)
        processor.start()
        await asyncio.wait_for(committed.wait(), 5)
        await asyncio.wait_for(blocked.wait(), 5)
        await processor.close()
        assert active == 0
        saved = live.read_json(processor.checkpoint_path)
        assert saved["scanned"] == saved["processed"] == 25
        assert not saved["complete"]

        replay = []

        async def finish(_semaphore, _client, _settings, cluster, posts, *_args):
            replay.extend(posts["post_id"].tolist())
            return {"status": "ok", "parsed": _analysis("low").model_dump(), "sample": []}

        monkeypatch.setattr(live, "_analyze_one", finish)
        resumed = live.LiveProcessor(pipeline_source)
        await asyncio.wait_for(resumed.run(), 5)
        assert resumed.status == "complete"
        assert resumed.resumed_from == 25
        assert sorted(replay) == list(range(25, 125))
        assert resumed.processed == resumed.scanned == 125

    asyncio.run(verify())


def test_cpu_cancellation_waits_for_source_thread_to_finish():
    import threading

    async def verify():
        loop = asyncio.get_running_loop()
        started, finished = asyncio.Event(), asyncio.Event()
        release = threading.Event()

        def work():
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=5)
            loop.call_soon_threadsafe(finished.set)

        task = asyncio.create_task(live._cpu(work))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(verify())


def test_batch_and_pattern_reviews_share_the_real_analyzers_32_request_limit(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, data_dir=tmp_path, llm_concurrency=32)
    posts = pd.DataFrame({
        "post_id": range(10), "text": [f"A sufficiently long post number {i}" for i in range(10)],
        "author": [f"author-{i}" for i in range(10)],
        "created_at": pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC"),
    })
    stats = cluster_stats(posts, settings)
    groups = [({**stats, "stats": stats, "cluster_id": i}, posts) for i in range(40)]
    baseline = build_baseline([group for group, _ in groups], settings, burstiness_reliable=False)

    async def verify():
        active = peak = 0
        full, release = asyncio.Event(), asyncio.Event()

        async def request(*_args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 32:
                full.set()
            try:
                await release.wait()
                parsed = _analysis("low")
                return parsed.model_dump_json(), parsed, 1
            finally:
                active -= 1

        monkeypatch.setattr(analyze, "_request_analysis", request)
        processor = live.LiveProcessor(settings)
        semaphore = asyncio.Semaphore(32)
        # The two assessment sets represent regular groups and rolling-pattern candidates.
        batches = [asyncio.create_task(processor._assess(part, baseline, object(), semaphore))
                   for part in (groups[:20], groups[20:])]
        try:
            await asyncio.wait_for(full.wait(), 5)
            assert active == peak == 32
        finally:
            release.set()
            results = await asyncio.gather(*batches)
        assert sum(map(len, results)) == 40
        assert all(record["status"] == "ok" for batch in results for record in batch)
        assert active == 0 and peak == 32

    asyncio.run(verify())


@pytest.mark.parametrize("configured_trump", [True, False])
def test_locked_trump_processing_preserves_cursor_and_chat_across_restart(tmp_path, monkeypatch, configured_trump):
    trump = api.DATASETS["trump-2020"]["repo"]
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        live_batch_size=25, live_window_posts=50, storage_cleanup=False,
        live_dataset=trump if configured_trump else api.DATASETS["bluesky"]["repo"],
    )
    monkeypatch.setattr(api, "settings", settings)
    monkeypatch.setattr(live, "_resolve_revision", lambda _: "fixed-revision")
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "_group", lambda *_: [])
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())
    opened = []

    def source(config, _revision):
        opened.append(config.live_dataset)
        if config.live_dataset == api.DATASETS["trump-2020"]["repo"]:
            rows = [{"source_id": 500 + i, "date": "2020-05-10 07:00:00",
                     "texto": f"A sufficiently long Trump dataset example number {i}"} for i in range(25)]
        else:
            rows = [{"did": f"did:example:{i}", "rkey": str(i),
                     "text": f"A sufficiently long Bluesky example number {i}",
                     "created_at": "2026-01-01T00:00:00Z"} for i in range(25)]
        return Dataset.from_list(rows).to_iterable_dataset()

    monkeypatch.setattr(live, "_source", source)
    live.write_json(tmp_path / "live_source.json", {"id": "trump-2020"})
    checkpoint = (tmp_path if configured_trump else tmp_path / "live-sources" / "trump-2020") / "live_progress.json"
    with TestClient(api.app) as client:
        assert client.get("/api/live/processed").status_code == 200
        original = checkpoint.read_bytes()
        active = api.app.state.live
        assert active.status == "complete"
        assert len(active.chat_context()[2]) == 25
        assert all("Trump dataset" in text for text in active.chat_context()[2].values())
        assert live.read_json(checkpoint)["processed"] == 25
        assert live.read_json(tmp_path / "live_source.json") == {"id": "trump-2020"}
        assert {item["id"] for item in client.get("/api/datasets").json() if item["live"]} == {"trump-2020"}
        for dataset in ("trump-2020", "bluesky"):
            assert client.post(f"/api/datasets/{dataset}/run").status_code == 403
        assert api.app.state.live is active
        assert checkpoint.read_bytes() == original

    # Even a changed deployment default must not move or reset the Trump cursor.
    monkeypatch.setattr(api, "settings", settings.model_copy(update={
        "live_dataset": api.DATASETS["bluesky"]["repo"] if configured_trump else trump,
    }))
    with TestClient(api.app) as client:
        assert api.app.state.live.settings.live_dataset == trump
        client.get("/api/live/processed")
        assert opened == [trump]  # Finished cursors are not silently replayed.
        assert api.app.state.live.checkpoint_path == checkpoint
        assert api.app.state.live.processed == 25
        assert checkpoint.read_bytes() == original
        assert live.read_json(tmp_path / "live_source.json") == {"id": "trump-2020"}


def test_locked_trump_resumes_unfinished_checkpoint_without_replaying_rows(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        live_batch_size=25, live_window_posts=50, storage_cleanup=False,
    )
    monkeypatch.setattr(api, "settings", settings)
    monkeypatch.setattr(live, "choose_device", lambda _: "cpu")
    monkeypatch.setattr(live, "load_model", lambda *_: object())
    monkeypatch.setattr(live, "_group", lambda *_: [])
    monkeypatch.setattr(live, "create_client", lambda _: AsyncMock())
    rows = [{"source_id": 500 + i, "date": "2020-05-10 07:00:00",
             "texto": f"A sufficiently long Trump dataset example number {i}"} for i in range(50)]
    source = Dataset.from_list(rows).to_iterable_dataset()
    iterator = iter(source)
    for _ in range(25):
        next(iterator)
    checkpoint = live.LiveCheckpoint(
        dataset=api.DATASETS["trump-2020"]["repo"],
        requested_revision=settings.live_dataset_revision,
        resolved_revision="fixed-revision", source_state=source.state_dict(),
        scanned=25, processed=25, group_id=7,
    )
    path = tmp_path / "live-sources" / "trump-2020" / "live_progress.json"
    live.write_json(path, checkpoint.model_dump(mode="json"))
    live.write_json(tmp_path / "live_source.json", {"id": "bluesky"})
    original = path.read_bytes()
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(rows).to_iterable_dataset())

    with TestClient(api.app) as client:
        assert path.read_bytes() == original
        assert client.post("/api/datasets/bluesky/run").status_code == 403
        client.get("/api/live/processed")
        processor = api.app.state.live
        assert processor.status == "complete"
        assert processor.resumed_from == 25 and processor.processed == 50
        assert set(processor.chat_context()[2].values()) == {row["texto"] for row in rows[25:]}
        assert live.read_json(tmp_path / "live_source.json") == {"id": "bluesky"}


def test_existing_live_stream_reports_dataset_switch():
    class Processor:
        revision = 0

        def start(self):
            pass

        def snapshot(self):
            return {"revision": 0, "status": "running"}

    original = Processor()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(live=original)),
        is_disconnected=AsyncMock(return_value=False),
    )

    async def verify():
        response = await api.live_processed(request)
        stream = response.body_iterator
        first = await anext(stream)
        assert '"status": "running"' in first
        request.app.state.live = Processor()
        switched = await anext(stream)
        assert "event: dataset_changed" in switched
        await stream.aclose()

    asyncio.run(verify())


def test_three_account_repeated_framing_uses_real_rules_despite_small_group_floor(
    pipeline_source, monkeypatch,
):
    settings = pipeline_source.model_copy(update={"live_window_posts": 100, "min_analysis_size": 10})
    repeated = (
        "No president has tried to fire a sitting governor for disagreeing with his agenda. "
        "Sending firing notices to central-bank officials is a power grab. Read: https://apnews.com/article/example"
    )
    rows = [{
        "did": f"did:example:{i}", "rkey": str(i),
        "text": repeated if i in {0, 25, 50} else f"A sufficiently long unique message number {i}",
        "created_at": "2026-01-01T00:00:00Z",
    } for i in range(75)]
    monkeypatch.setattr(live, "_source", lambda *_: Dataset.from_list(rows).to_iterable_dataset())
    monkeypatch.setattr(live, "_group", lambda *_: [])

    async def request(_client, _settings, prompt):
        assert '"source_scope": "repeated text across live batches"' in prompt
        parsed = _analysis("low")
        parsed.claims = parsed.claims[:1]
        parsed.claims[0].text = "The president is trying to overrule central-bank officials."
        parsed.claims[0].evidence = ["p1", "p2", "p3"]
        parsed.techniques = []
        parsed.templating.evidence = ["p1", "p2", "p3"]
        return parsed.model_dump_json(), parsed, 1

    monkeypatch.setattr(analyze, "_request_analysis", request)
    processor = live.LiveProcessor(settings)
    asyncio.run(processor.run())
    assert processor.status == "complete", processor.error
    candidate = next(iter(processor.pattern_results.values()))
    assert candidate["cluster"]["size"] == 3
    assert not candidate["cluster"]["behavior_anomaly"]
    assert candidate["cluster"]["concern_level"] == "medium"
    assert candidate["analysis"]["parsed"]["confidence"] == 0.4
    assert any(entry["rule"] == "repeated_cross_account_framing" for entry in candidate["analysis"]["rule_log"])
