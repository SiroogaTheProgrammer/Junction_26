import pandas as pd
import pytest

from detector.config import Settings
from detector.io import read_json, write_json
from detector.stats import (
    baseline_for, behavior_anomaly, build_baseline, cluster_stats, compute_baseline,
    compute_stats, densest_window_share, describe_signals, extract_domains,
    extract_hashtags, is_organic_looking, near_duplicate_groups, normalize_text, run,
)


def test_normalize_text_replaces_urls_removes_mentions_and_collapses_punctuation():
    assert normalize_text("HELLO, @person! Visit https://example.com/a?b=1") == "hello visit <url>"


def test_compute_stats_counts_repetition_and_behavior_signals():
    posts = pd.DataFrame(
        [
            {"post_id": 0, "author": "a", "created_at": "2025-01-01T00:00:00Z", "text": "Same post! #News https://a.example/x"},
            {"post_id": 1, "author": "a", "created_at": "2025-01-01T00:01:00Z", "text": "same post @someone #News https://a.example/y"},
            {"post_id": 2, "author": "a", "created_at": "2025-01-01T00:02:00Z", "text": "Another claim #News https://b.example"},
            {"post_id": 3, "author": "b", "created_at": "2025-01-01T00:20:00Z", "text": "An unrelated view"},
        ]
    )

    stats = compute_stats(posts)

    assert stats["size"] == 4
    assert stats["n_authors"] == 2
    assert stats["top_author_share"] == 0.75
    assert stats["dup_ratio"] == 0.5
    assert stats["top_domain_share"] == 0.5
    assert stats["top_hashtag_share"] == 0.75
    assert stats["burstiness"] == 0.75
    assert stats["time_span_min"] == 20


def test_baseline_and_organic_looking_use_p75():
    clusters = [
        {"dup_ratio": 0.1, "top_author_share": 0.2, "top_domain_share": 0.3, "top_hashtag_share": 0.4, "burstiness": 0.5},
        {"dup_ratio": 0.2, "top_author_share": 0.4, "top_domain_share": 0.5, "top_hashtag_share": 0.6, "burstiness": 0.7},
        {"dup_ratio": 0.3, "top_author_share": 0.6, "top_domain_share": 0.7, "top_hashtag_share": 0.8, "burstiness": 0.9},
        {"dup_ratio": 0.4, "top_author_share": 0.8, "top_domain_share": 0.9, "top_hashtag_share": 1.0, "burstiness": 1.0},
    ]
    baseline = compute_baseline(clusters)

    assert baseline["dup_ratio"]["p75"] == pytest.approx(0.325)
    assert is_organic_looking(
        {"dup_ratio": 0.3, "top_author_share": 0.6, "burstiness": 0.9},
        baseline,
    )
    assert not is_organic_looking(
        {"dup_ratio": 0.4, "top_author_share": 0.6, "burstiness": 0.9},
        baseline,
    )


def test_empty_baseline_is_zeroed():
    assert compute_baseline([])["burstiness"] == {"p50": 0.0, "p75": 0.0, "p90": 0.0}


def test_unicode_bare_domains_and_filename_denylist():
    assert normalize_text("ＨＥＬＬＯ @someone #Été_2026 😀 example.com/path") == "hello été 2026 <url>"
    assert extract_domains("https://WWW.Example.com/a, example.com/b www.site.org/c image.png notes.pdf") == ["example.com", "site.org"]
    assert extract_domains("mail@example.com; news.example.org/story") == ["news.example.org"]
    assert extract_hashtags("#Été #été #東京 #abc_12 #abc_12") == ["abc_12", "été", "東京"]
    assert extract_domains("https://image.png/a") == ["image.png"]


def test_near_duplicate_components_short_text_and_nonmatches():
    base = "the government is hiding the detailed public report about water quality in our local community please share this important evidence with every resident today"
    texts = [base, base.replace("today", "tomorrow"), base.upper(), "short", "SHORT!", "shorter", "", "", "A totally unrelated message about repairing a bicycle in the rain."]
    groups = near_duplicate_groups(texts)
    assert groups[0] == groups[1] == groups[2] >= 0
    assert groups[3] == groups[4] >= 0
    assert groups[5:] == [-1] * 4
    assert near_duplicate_groups(["", ""]) == [-1, -1]


@pytest.mark.parametrize("threshold", [0.01, 0.8, 0.99, 1.0])
def test_near_duplicate_threshold_boundaries(threshold):
    assert near_duplicate_groups(["Sufficiently long same content"] * 2, threshold) == [0, 0]


@pytest.mark.parametrize("times,expected", [
    (["2026-01-01T00:00:00Z"], 1),
    (["2026-01-01T00:00:00Z"] * 4, 1),
    (["2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z"], 1),
    (["2026-01-01T00:00:00Z", "2026-01-01T00:10:01Z"], 0.5),
    ([f"2026-01-01T0{index}:00:00Z" for index in range(4)], 0.25),
])
def test_densest_window_boundaries(times, expected):
    assert densest_window_share(pd.Series(times), 10) == expected


def metric_row(size, value):
    return {"size": size, **dict.fromkeys((
        "dup_ratio", "top_author_share", "top5_author_share", "burstiness",
        "top_domain_share", "top_hashtag_share", "authors_per_post",
    ), value)}


def test_size_baseline_edges_fallback_and_reliability():
    rows = [metric_row(30, 0.1), metric_row(40, 0.2), metric_row(50, 0.8), metric_row(200, 0.9)]
    settings = Settings(_env_file=None, min_clusters_per_bucket=2, min_analysis_size=20)
    baseline = build_baseline(rows, settings, burstiness_reliable=False)
    assert baseline_for(24, baseline)["name"] == "global"
    assert baseline_for(25, baseline)["name"] == baseline_for(49, baseline)["name"] == "25-49"
    assert baseline_for(49, baseline)["source"] == "bucket"
    assert baseline_for(50, baseline)["name"] == baseline_for(199, baseline)["name"] == "50-199"
    assert baseline_for(50, baseline)["source"] == "global"
    assert baseline_for(200, baseline)["name"] == "200+"
    stats = metric_row(30, 0.1)
    stats["burstiness"] = 1
    assert is_organic_looking(stats, baseline)
    assert not behavior_anomaly(stats, baseline).flag
    assert not any("window" in line for line in describe_signals(stats, baseline))
    stats["dup_ratio"] = 1
    assert behavior_anomaly(stats, baseline).triggered_by == ["dup_ratio"]
    stats["size"] = 19
    assert not behavior_anomaly(stats, baseline).flag


def test_full_statistics_distinguish_multi_author_repetition_from_single_author():
    posts = pd.DataFrame({
        "post_id": range(50), "author": [f"synthetic-{index % 20}" for index in range(50)],
        "text": ["Share this report about the local water supply right now #report example.com/news"] * 50,
        "created_at": pd.date_range("2026-01-01", periods=50, freq="5s", tz="UTC"),
        "has_images": pd.Series([pd.NA] * 50, dtype="boolean"),
    })
    swarm = cluster_stats(posts)
    assert (swarm["dup_ratio"], swarm["exact_dup_ratio"], swarm["dup_authors"], swarm["largest_dup_group"]) == (1, 1, 20, 50)
    assert swarm["image_share"] is None
    assert swarm["authors_per_post"] == 0.4
    assert swarm["top5_author_share"] == 0.3
    assert swarm["top_domain"] == "example.com" and swarm["link_share"] == 1
    assert swarm["top_hashtag"] == "report"
    posts["author"] = "synthetic-single"
    spam = cluster_stats(posts)
    assert spam["dup_authors"] == 1
    assert spam["top_author_share"] == 1


def test_stats_stage_is_idempotent_preserves_membership_and_reports_short_snapshot(tmp_path, capsys):
    posts = pd.DataFrame({
        "post_id": range(12), "author": [f"synthetic-{index}" for index in range(12)],
        "text": ["Repeated sufficiently long message"] * 6 + [f"Other post about topic {index}" for index in range(6)],
        "created_at": pd.date_range("2026-01-01", periods=12, freq="min", tz="UTC"),
    })
    assignments = pd.DataFrame({"post_id": range(12), "cluster_id": [0] * 6 + [1] * 6})
    posts.to_parquet(tmp_path / "posts.parquet")
    assignments.to_parquet(tmp_path / "assignments.parquet")
    write_json(tmp_path / "clusters.json", {
        "run": {"original": True}, "clusters": [
            {"cluster_id": index, "size": 6, "representative_post_ids": [index * 6]} for index in range(2)
        ],
    })
    settings = Settings(_env_file=None, data_dir=tmp_path)
    original = (tmp_path / "assignments.parquet").read_bytes()
    run(settings, inspect=True)
    first = (tmp_path / "clusters.json").read_bytes()
    doc = read_json(tmp_path / "clusters.json")
    assert doc["run"]["stats_version"] == "1"
    assert doc["dataset"]["burstiness_reliable"] is False
    assert doc["dataset"]["image_metadata_available"] is False
    assert doc["clusters"][0]["representative_post_ids"] == [0]
    assert "Top groups by dup_ratio" in capsys.readouterr().out
    report = read_json(tmp_path / "stats_report.json")
    assert report["burstiness_reliable"] is False
    assert len(report["global_fallback_buckets"]) == 3
    run(settings)
    assert first == (tmp_path / "clusters.json").read_bytes()
    assert original == (tmp_path / "assignments.parquet").read_bytes()


@pytest.mark.parametrize("overrides", [
    {"baseline_buckets": "50,25"}, {"baseline_buckets": "a,50"},
    {"baseline_buckets": "25,25"}, {"near_dup_jaccard": 1.1},
    {"minhash_perm": 0}, {"burst_window_min": 0},
])
def test_invalid_stats_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)
