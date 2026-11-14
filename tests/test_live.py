import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from detector import api
from detector.config import Settings
from tests.fixtures.generate import generate

FIXTURES = Path(__file__).parent / "fixtures"


def events(text):
    parsed = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((fields.get("event", "message"), json.loads(fields["data"])))
    return parsed


def test_live_stream_replays_every_post_once_in_arrival_order(tmp_path, monkeypatch):
    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json", "projection.parquet"):
        shutil.copy2(FIXTURES / name, tmp_path / name)
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path, live_posts_per_second=600))
    monkeypatch.setattr(api, "LIVE_TICK_S", 0.01)
    with TestClient(api.app) as client:
        response = client.get("/api/live/stream")
        assert response.headers["content-type"].startswith("text/event-stream")
        stream = events(response.text)
        assert client.get("/api/meta").json()["live"] == {"posts_per_second": 600}
        stored = {point[0]: point for point in client.get("/api/map").json()["points"]}
    assert stream[0] == ("start", {"total": 300, "posts_per_second": 600})
    assert stream[-1] == ("done", {"total": 300})
    batches = [data for name, data in stream if name == "message"]
    assert [len(batch["points"]) for batch in batches] == [6] * 50
    assert [batch["sent"] for batch in batches] == list(range(6, 301, 6))
    points = [point for batch in batches for point in batch["points"]]
    assert len({point[0] for point in points}) == 300
    assert all(point == stored[point[0]] for point in points)
    posts = pd.read_parquet(tmp_path / "posts.parquet").set_index("post_id")
    arrival = [posts.loc[point[0], "created_at"] for point in points]
    assert arrival == sorted(arrival)
    assert batches[-1]["clock"] == max(arrival).isoformat()


def test_live_stream_needs_the_projection(tmp_path, monkeypatch):
    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json"):
        shutil.copy2(FIXTURES / name, tmp_path / name)
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path))
    with TestClient(api.app) as client:
        response = client.get("/api/live/stream")
        assert response.status_code == 503
        assert "projection is missing" in response.json()["detail"]
