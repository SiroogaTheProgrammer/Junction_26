import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from detector import analyze
from detector.__main__ import parser
from detector.config import Settings
from detector.io import read_json, read_jsonl, write_jsonl
from detector.schema import AnalysisRecord
from tests.fixtures.generate import generate
from tests.test_analyze_rules import _analysis


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    generate()
    for name in ("posts.parquet", "assignments.parquet", "clusters.json", "analyses.jsonl"):
        shutil.copy2(Path(__file__).parent / "fixtures" / name, tmp_path / name)
    settings = Settings(
        _env_file=None, data_dir=tmp_path, llm_api_key="test-key",
        llm_base_url="https://test.invalid/v1", max_analyzed_clusters=2,
    )
    warmup = AsyncMock()
    request = AsyncMock(return_value=(_analysis().model_dump_json(), _analysis(), 1))
    client = AsyncMock()
    monkeypatch.setattr(analyze, "_warm_up", warmup)
    monkeypatch.setattr(analyze, "_request_analysis", request)
    monkeypatch.setattr(analyze, "create_client", lambda settings: client)
    return settings, warmup, request


def test_pipeline_persists_records_report_and_uses_cache_without_endpoint(pipeline):
    settings, warmup, request = pipeline
    asyncio.run(analyze.run(settings))
    records = read_jsonl(settings.data_path / "analyses.jsonl")
    assert len(records) == 2
    for record in records:
        AnalysisRecord.model_validate(record)
        assert record["prompt_version"] == analyze.PROMPT_VERSION
        assert "display_tier" not in record
        assert record["concern_level_model"] == "high"
        assert record["relevance_model"] == 0.8
    report = read_json(settings.data_path / "analyze_report.json")
    assert (report["selected"], report["analysed"], report["cached"], report["failed"]) == (2, 2, 0, 0)
    assert sum(report["counts_per_relevance_kind"].values()) == 2
    assert sum(report["tier_counts"].values()) == 2
    request.reset_mock()
    warmup.reset_mock()
    asyncio.run(analyze.run(settings))
    request.assert_not_called()
    warmup.assert_not_called()
    report = read_json(settings.data_path / "analyze_report.json")
    assert report["cached"] == 2
    assert report["total_latency_s"] == 0
    asyncio.run(analyze.run(settings, force=True))
    assert request.await_count == 2
    warmup.assert_awaited_once()


def test_single_cluster_preserves_other_current_results_and_bypasses_size_limit(pipeline):
    settings, _, request = pipeline
    asyncio.run(analyze.run(settings))
    before = read_jsonl(settings.data_path / "analyses.jsonl")
    asyncio.run(analyze.run(settings, cluster_id=7))
    records = read_jsonl(settings.data_path / "analyses.jsonl")
    assert len(records) == 3
    assert records[:2] == before
    assert records[-1]["cluster_id"] == 7
    assert request.await_count == 3
    with pytest.raises(ValueError, match="was not found"):
        asyncio.run(analyze.run(settings, cluster_id=999))


def test_dry_run_ignores_endpoint_and_cache_and_does_not_write(pipeline, capsys):
    settings, warmup, request = pipeline
    settings.llm_api_key = None
    path = settings.data_path / "analyses.jsonl"
    before = path.read_bytes()
    asyncio.run(analyze.run(settings, cluster_id=0, dry_run=True))
    output = json.loads(capsys.readouterr().out)
    assert output["cluster_id"] == 0
    assert len(output["sample"]) == settings.sample_size == 50
    assert output["user_prompt"].count("<post id=") == 50
    assert "<posts>" in output["user_prompt"]
    assert "Relevance rubric:" in output["system_prompt"]
    assert output["input_hash"]
    assert path.read_bytes() == before
    assert not (settings.data_path / "analyze_report.json").exists()
    warmup.assert_not_called()
    request.assert_not_called()


@pytest.mark.parametrize("field,value", [("llm_api_key", None), ("llm_base_url", "")])
def test_missing_endpoint_warns_and_preserves_artifacts(pipeline, caplog, field, value):
    settings, warmup, request = pipeline
    setattr(settings, field, value)
    path = settings.data_path / "analyses.jsonl"
    before = path.read_bytes()
    asyncio.run(analyze.run(settings))
    assert path.read_bytes() == before
    assert "skipping analysis" in caplog.text
    request.assert_not_called()
    warmup.assert_not_called()


def test_failed_calls_are_recorded_and_retried_on_next_run(pipeline):
    settings, _, request = pipeline
    request.side_effect = analyze.AnalysisCallError("Empty", 4, '{"reason":"empty"}')
    asyncio.run(analyze.run(settings, cluster_id=0))
    record = read_jsonl(settings.data_path / "analyses.jsonl")[0]
    assert record["status"] == "failed"
    assert record["attempts"] == 4
    assert record["raw_response"] == '{"reason":"empty"}'
    assert read_json(settings.data_path / "analyze_report.json")["failed"] == 1
    request.side_effect = None
    asyncio.run(analyze.run(settings, cluster_id=0))
    assert request.await_count == 2
    assert read_jsonl(settings.data_path / "analyses.jsonl")[0]["status"] == "ok"


def test_finished_clusters_survive_interruption(pipeline, monkeypatch):
    settings, _, _ = pipeline

    async def interrupted_run():
        persisted = asyncio.Event()
        calls = 0

        async def request(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _analysis().model_dump_json(), _analysis(), 1
            await persisted.wait()
            raise asyncio.CancelledError()

        def persist(path, records):
            write_jsonl(path, records)
            if any(record["cluster_id"] == 0 for record in records):
                persisted.set()

        monkeypatch.setattr(analyze, "_request_analysis", request)
        monkeypatch.setattr(analyze, "write_jsonl", persist)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(analyze.run(settings), timeout=3)

    asyncio.run(interrupted_run())
    records = read_jsonl(settings.data_path / "analyses.jsonl")
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["cluster_id"] == 0


def test_hash_ignores_display_settings_but_covers_sample_text_and_rule_settings(pipeline):
    settings, _, _ = pipeline
    document = read_json(settings.data_path / "clusters.json")
    cluster = document["clusters"][0]
    posts = pd.read_parquet(settings.data_path / "posts.parquet").head(cluster["size"])

    def digest():
        return analyze._cluster_input_hash(settings, cluster, posts, document["baseline"])

    initial = digest()
    settings.sample_size = 20
    assert digest() != initial
    settings.sample_size = 50
    settings.relevance_hide_below = 0.1
    settings.relevance_full_from = 0.2
    assert digest() == initial
    posts.loc[:, "text"] = "Changed text with identical post ids"
    assert digest() != initial
    text_digest = digest()
    settings.min_sample_for_confidence = 21
    assert digest() != text_digest


def test_cli_accepts_analyzer_flags():
    args = parser().parse_args(["analyze", "--cluster", "7", "--dry-run", "--force"])
    assert (args.cluster, args.dry_run, args.force) == (7, True, True)


def test_all_clouds_bypasses_legacy_limits_and_labels_small_clouds(pipeline):
    settings, _, request = pipeline
    settings.max_analyzed_clusters = 1
    settings.min_analysis_size = 60
    asyncio.run(analyze.run(settings, all_clusters=True))
    records = read_jsonl(settings.data_path / "analyses.jsonl")
    assert len(records) == 8
    assert request.await_count == 8
    assert all(record["status"] == "ok" for record in records)


def test_zero_limit_means_every_eligible_cloud():
    settings = Settings(_env_file=None)
    clouds = [{"cluster_id": i, "size": 1 + i} for i in range(75)]
    assert len(analyze.select_clusters(clouds, settings)) == 75
    assert parser().parse_args(["all", "--posts", "10000"]).posts == 10000


def test_stats_stage_reaches_prompt_cache_and_saved_assessment(pipeline):
    from detector.stats import run as stats

    settings, _, request = pipeline
    stats(settings)
    document = read_json(settings.data_path / "clusters.json")
    group = document["clusters"][0]
    asyncio.run(analyze.run(settings, cluster_id=group["cluster_id"]))
    record = read_jsonl(settings.data_path / "analyses.jsonl")[0]
    assert record["stats_hash"] == group["stats_hash"]
    assert "baseline bucket:" in record["user_prompt"]
    assert "near-identical text" in record["user_prompt"]
    assert "dup_authors" in record["user_prompt"]
    calls = request.await_count
    asyncio.run(analyze.run(settings, cluster_id=group["cluster_id"]))
    assert request.await_count == calls
    settings.burst_window_min = 20
    stats(settings)
    asyncio.run(analyze.run(settings, cluster_id=group["cluster_id"]))
    assert request.await_count == calls + 1
    record = read_jsonl(settings.data_path / "analyses.jsonl")[0]
    assert "densest 20 min" in record["user_prompt"]
