import time
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from detector import api
from detector.config import Settings
from tests.fixtures.generate import generate
import pandas as pd
import pytest


def test_read_only_dashboard_api_reads_fixture_artifacts(tmp_path, monkeypatch):
    source = Path(__file__).parent / "fixtures"
    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json", "analyses.jsonl", "projection.parquet"):
        shutil.copy2(source / name, tmp_path / name)
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path))

    with TestClient(api.app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        for page in ("/", "/explore", "/live"):
            response = client.get(page)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-cache"
        for asset in ("map.js", "group-card.js", "labels.js", "chat.js", "explore.js", "live.js", "menu.js", "styles.css"):
            response = client.get(f"/assets/{asset}")
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-cache"
        for weight in ("Regular", "SemiBold"):
            font = client.get(f"/assets/fonts/JetBrainsMono-{weight}.woff2")
            assert font.status_code == 200
            assert font.content.startswith(b"wOF2")
        meta = client.get("/api/meta").json()
        assert meta["n_posts"] == 300
        assert meta["n_clusters"] == 8
        assert meta["n_analysed"] == 6
        assert meta["n_failed"] == 1
        assert meta["map_available"] is True
        assert meta["n_pending"] == 1
        points = client.get("/api/map").json()
        assert points["columns"] == ["post_id", "cluster_id", "x", "y", "z"]
        assert len(points["points"]) == 300
        assert len({point[0] for point in points["points"]}) == 300
        assert points["projection"]["source_dimensions"] == 384
        cluster_rows = client.get("/api/clusters?include_hidden=true").json()
        assert len(cluster_rows) == 8
        visible = client.get("/api/clusters").json()
        assert all(row["display_tier"] != "hidden" for row in visible)
        assert meta["n_hidden"] == len(cluster_rows) - len(visible) == 1
        assert cluster_rows[1]["display_tier"] == "hidden"
        assert cluster_rows[5]["display_tier"] == "faded"
        assert cluster_rows[5]["behavior_anomaly"] is True
        assert meta["relevance_thresholds"] == {"hide_below": 0.35, "full_from": 0.65}
        assert meta["anomaly_percentile"] == 90
        assert {row["status"] for row in cluster_rows} == {"ok", "failed", "not_analysed"}
        assert {row["concern_level"] for row in cluster_rows if row["status"] == "ok"} == {
            "none", "low", "medium", "high", "insufficient_evidence",
        }
        assert any(row["status"] == "ok" and row["confidence"] < 0.5 for row in cluster_rows)
        assert cluster_rows[0]["status"] == "ok"
        assert cluster_rows[3]["review_priority"] > cluster_rows[0]["review_priority"]
        assert cluster_rows[3]["review_score"]["content_points"] == 60
        assert cluster_rows[7]["review_score"]["status"] == "spread_only"
        # A finished 0/100 assessment is dropped from the map; failed or pending clouds never are.
        for row in cluster_rows:
            assert row["discarded"] == (row["status"] == "ok" and row["review_priority"] == 0)
        assert any(row["discarded"] for row in cluster_rows)
        assert not cluster_rows[6]["discarded"] and not cluster_rows[7]["discarded"]
        assert meta["n_discarded"] == sum(row["discarded"] for row in cluster_rows)
        assert meta["min_review_priority"] == 1
        detail = client.get("/api/clusters/0").json()
        assert detail["analysis"]["sample"][0]["pid"] == "p1"
        assert detail["analysis"]["sample"][0]["author"] == "user-0"
        page = client.get("/api/clusters/0/posts?offset=10&limit=5").json()
        assert page["total"] == 60
        assert len(page["posts"]) == 5
        api.app.state.dashboard["analyses"][5]["behavior_anomaly"]["flag"] = False
        assert client.get("/api/clusters/5").json()["cluster"]["behavior_anomaly"] is True
        # Display thresholds are applied to stored analysis without any LLM calls.
        monkeypatch.setattr(api, "settings", Settings(
            _env_file=None, data_dir=tmp_path, relevance_hide_below=0, relevance_full_from=0.1,
        ))
        rows = client.get("/api/clusters").json()
        assert len(rows) == 8
        assert rows[1]["display_tier"] == "full"
        assert client.get("/api/meta").json()["n_hidden"] == 0


def test_missing_projection_keeps_review_queue_but_returns_explicit_map_error(tmp_path, monkeypatch):
    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json", "analyses.jsonl"):
        shutil.copy2(Path(__file__).parent / "fixtures" / name, tmp_path / name)
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path))
    with TestClient(api.app) as client:
        meta = client.get("/api/meta").json()
        assert meta["map_available"] is False
        assert "Run python -m detector cluster" in meta["map_error"]
        response = client.get("/api/map")
        assert response.status_code == 503
        assert response.json()["detail"] == meta["map_error"]
        assert client.get("/api/clusters?include_hidden=true").status_code == 200


def test_stats_metadata_baselines_and_stale_assessments_are_exposed(tmp_path, monkeypatch):
    from detector.stats import run as stats

    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json", "analyses.jsonl"):
        shutil.copy2(Path(__file__).parent / "fixtures" / name, tmp_path / name)
    configured = Settings(_env_file=None, data_dir=tmp_path)
    monkeypatch.setattr(api, "settings", configured)
    stats(configured)
    with TestClient(api.app) as client:
        meta = client.get("/api/meta").json()
        assert meta["dataset"]["burstiness_reliable"] is True
        assert meta["run"]["stats_version"] == "1"
        assert meta["n_stale"] == 7
        row = client.get("/api/clusters/5").json()["cluster"]
        assert row["dup_authors"] == 1
        assert row["stats"]["largest_dup_group"] == 30
        assert row["signals"]
        assert row["bucket_baseline"]["name"] == "25-49"
        assert row["flags"]["behavior_anomaly"]["flag"] == row["behavior_anomaly"]
        assert row["analysis_stale"] and not row["discarded"]
        api.app.state.dashboard["analyses"][5]["stats_hash"] = row["stats_hash"]
        assert client.get("/api/clusters/5").json()["cluster"]["analysis_stale"] is False


def test_prompt_and_sample_changes_mark_assessments_stale(tmp_path, monkeypatch):
    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json", "analyses.jsonl"):
        shutil.copy2(Path(__file__).parent / "fixtures" / name, tmp_path / name)
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path))
    with TestClient(api.app) as client:
        assert client.get("/api/meta").status_code == 200
        record = api.app.state.dashboard["analyses"][0]
        assert not client.get("/api/clusters/0").json()["cluster"]["analysis_stale"]
        record["prompt_version"] = "v4"
        assert client.get("/api/clusters/0").json()["cluster"]["analysis_stale"]
        record["prompt_version"] = api.PROMPT_VERSION
        record["sample"] = record["sample"][:20]
        assert client.get("/api/clusters/0").json()["cluster"]["analysis_stale"]
        row = client.get("/api/clusters/0").json()["cluster"]
        assert not row["discarded"]


@pytest.mark.parametrize("corruption", ["membership", "nan", "duplicate", "columns"])
def test_projection_cannot_silently_display_mismatched_data(tmp_path, monkeypatch, corruption):
    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json", "projection.parquet"):
        shutil.copy2(Path(__file__).parent / "fixtures" / name, tmp_path / name)
    path = tmp_path / "projection.parquet"
    points = pd.read_parquet(path)
    if corruption == "membership":
        points.loc[0, "cluster_id"] = 999
    elif corruption == "nan":
        points.loc[0, "x"] = float("nan")
    elif corruption == "duplicate":
        points.loc[0, "post_id"] = points.loc[1, "post_id"]
    else:
        points = points.drop(columns=["z"])
    points.to_parquet(path, index=False)
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path))
    with pytest.raises(ValueError):
        api._load_dashboard_data()


def test_dataset_browser_lists_allowed_datasets_and_pages_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path))
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"features": [{"name": "texto"}, {"name": "source_id"}],
                    "rows": [{"row_idx": 25, "row": {"texto": "hello", "source_id": 509}}], "num_rows_total": 2_300_000}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, params):
            seen.update(params, url=url)
            return Response()

    monkeypatch.setattr(api.httpx, "AsyncClient", Client)
    with TestClient(api.app) as client:
        assert client.get("/data").status_code == 200
        listed = {item["id"]: item for item in client.get("/api/datasets").json()}
        assert listed["trump-2020"]["repo"] == "falbanese/US_Trump_2020_social_media"
        assert listed["bluesky"]["live"] is True and listed["trump-2020"]["live"] is False
        body = client.get("/api/datasets/trump-2020/rows?offset=25&length=1").json()
        assert body == {"id": "trump-2020", "repo": "falbanese/US_Trump_2020_social_media", "columns": ["texto", "source_id"],
                        "rows": [{"texto": "hello", "source_id": 509}], "offset": 25, "total": 2_300_000}
        assert seen["dataset"] == "falbanese/US_Trump_2020_social_media" and seen["offset"] == 25 and seen["length"] == 1
        assert client.get("/api/datasets/other/rows").status_code == 404
        assert client.get("/api/datasets/trump-2020/rows?length=500").status_code == 422

        async def fail(self, url, params):
            raise api.httpx.ConnectError("offline")
        monkeypatch.setattr(Client, "get", fail)
        failed = client.get("/api/datasets/trump-2020/rows")
        assert failed.status_code == 502 and "Hugging Face" in failed.json()["detail"]


@pytest.mark.parametrize("enabled", [True, False])
def test_server_runs_storage_cleanup_at_startup_only_when_enabled(tmp_path, monkeypatch, enabled):
    calls = []
    monkeypatch.setattr("detector.storage.cleanup", lambda settings: calls.append(settings.data_path))
    monkeypatch.setattr(api, "settings", Settings(_env_file=None, data_dir=tmp_path, storage_cleanup=enabled))
    with TestClient(api.app) as client:
        for _ in range(50):
            if calls or not enabled:
                break
            time.sleep(0.02)
        assert client.get("/api/health").status_code == 200
    assert calls == ([tmp_path] if enabled else [])
