import pytest

from detector.rules import review_priority


def cluster(value):
    return dict.fromkeys(("dup_ratio", "top_author_share", "burstiness"), value)


def baseline(value):
    return {metric: {"p90": value} for metric in cluster(value)}


@pytest.mark.parametrize("concern,points", [("none", 0), ("low", 15), ("medium", 39), ("high", 60), ("insufficient_evidence", 0)])
def test_review_score_content_mapping(concern, points):
    result = review_priority({"concern_level": concern}, cluster(0.1), baseline(0.2))
    assert result["score"] == points
    assert result["content_points"] == points
    assert result["spread_points"] == 0
    assert result["status"] == "complete"


def test_review_score_is_bounded_explainable_and_not_confidence_or_relevance():
    parsed = {"concern_level": "high", "confidence": 0.1, "relevance": 0.2}
    result = review_priority(parsed, cluster(1.0), baseline(0.5))
    assert result["score"] == 100
    assert result["spread_points"] == 40
    assert all(signal["normalized_excess"] == 1 for signal in result["signals"])
    assert "not a probability" in result["explanation"]
    parsed.update(confidence=0.9, relevance=0.9)
    assert review_priority(parsed, cluster(1.0), baseline(0.5)) == result


def test_review_score_spread_only_and_boundary():
    result = review_priority(None, cluster(1.0), baseline(0.5))
    assert result["score"] == 40
    assert result["status"] == "spread_only"
    assert review_priority(None, cluster(0.5), baseline(0.5))["score"] == 0
    assert review_priority(None, cluster(1.0), baseline(1.0))["score"] == 0


def test_single_anomaly_increases_priority_without_becoming_a_verdict():
    stats = cluster(0.1)
    stats["dup_ratio"] = 1.0
    result = review_priority({"concern_level": "none"}, stats, baseline(0.2))
    assert result["score"] == 27
    assert result["content_points"] == 0
    with pytest.raises(ValueError):
        review_priority(None, cluster(2.0), baseline(0.2))


def test_unreliable_timing_cannot_raise_priority_or_hide_size_baseline():
    from detector.config import Settings
    from detector.stats import build_baseline

    rows = [{"size": 30, **cluster(0.1)}, {"size": 40, **cluster(0.2)}, {"size": 1000, **cluster(0.9)}]
    bounds = build_baseline(rows, Settings(_env_file=None, min_clusters_per_bucket=2), burstiness_reliable=False)
    stats = {"size": 30, **cluster(0.1)}
    stats["burstiness"] = 1.0
    result = review_priority(None, stats, bounds)
    assert result["score"] == 0
    assert result["burstiness_reliable"] is False
    assert [signal["metric"] for signal in result["signals"]] == ["dup_ratio", "top_author_share"]
    stats["dup_ratio"] = 0.5
    result = review_priority(None, stats, bounds)
    assert result["score"] > 0
    assert result["signals"][0]["threshold"] == 0.19
