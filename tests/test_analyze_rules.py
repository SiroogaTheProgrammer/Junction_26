import pandas as pd
import pytest

from detector.analyze import (
    clean_post_text,
    extract_json,
    sample_posts,
)
from detector.config import Settings
from detector.rules import apply_rules, behavior_anomaly, display_tier
from detector.schema import Analysis, Claim, Technique, Templating


def _analysis(
    concern_level: str = "high",
    legit_lookalike: str | None = None,
) -> Analysis:
    return Analysis(
        label="Shared local concern",
        summary="Several posts discuss the same local issue.",
        relevance_kind="narrative",
        relevance=0.8,
        relevance_reason="Shared framing about a local public service.",
        claims=[
            Claim(text="A service is delayed.", kind="checkable", evidence=["p1", "p999"]),
            Claim(text="This is frustrating.", kind="opinion", evidence=["p999"]),
        ],
        techniques=[
            Technique(name="false_urgency", evidence=["p999"]),
            Technique(name="loaded_language", evidence=["p1"]),
        ],
        templating=Templating(detected=True, evidence=["p1", "p999"]),
        framing_diversity="medium",
        legit_lookalike=legit_lookalike,
        concern_level=concern_level,
        confidence=0.8,
        explanation="Repeated phrasing and concentrated timing are visible in the sample.",
    )


def rule_context(**overrides):
    return {
        "valid_pids": {f"p{i}" for i in range(1, 11)},
        "organic_looking": False, "behavior_anomaly": False,
        "min_sample_for_confidence": 8, **overrides,
    }


def test_evidence_filtering_removes_unknown_ids_and_empty_claims_and_techniques():
    parsed, changes = apply_rules(_analysis("low"), rule_context())

    assert parsed.claims[0].evidence == ["p1"]
    assert len(parsed.claims) == 1
    assert [technique.name for technique in parsed.techniques] == ["loaded_language"]
    assert parsed.templating.evidence == ["p1"]
    assert {change["field"] for change in changes} == {"claims", "techniques", "templating"}


def test_organic_stats_do_not_downgrade_evidenced_content_concern():
    original = _analysis("high")
    parsed, changes = apply_rules(original, rule_context(organic_looking=True))

    assert parsed.concern_level == "high"
    assert all(change["rule"] != "organic_stats" for change in changes)
    assert original.concern_level == "high"


def test_lookalike_does_not_override_evidenced_content_concern():
    parsed, changes = apply_rules(
        _analysis("medium", "looks like syndicated headlines"),
        rule_context(),
    )

    assert parsed.concern_level == "medium"
    assert all(change["rule"] != "lookalike" for change in changes)


def test_unsupported_content_concern_is_not_presented_as_high():
    analysis = _analysis()
    analysis.claims = []
    analysis.techniques = []
    parsed, changes = apply_rules(analysis, rule_context())
    assert parsed.concern_level == "insufficient_evidence"
    assert any(change["rule"] == "missing_content_evidence" for change in changes)


def test_empty_templating_evidence_resets_detected():
    analysis = _analysis()
    analysis.templating.evidence = ["p999"]
    parsed, changes = apply_rules(analysis, rule_context())
    assert parsed.templating.detected is False
    assert parsed.templating.evidence == []
    assert any(change["field"] == "templating" for change in changes)


def test_rules_order_clamping_and_small_sample():
    analysis = _analysis(legit_lookalike="News relay")
    analysis.relevance = -1
    analysis.confidence = 4
    parsed, changes = apply_rules(analysis, rule_context(valid_pids={"p1"}, organic_looking=True))
    assert parsed.relevance == 0
    assert parsed.confidence == 0.4
    assert parsed.concern_level == "high"
    assert [change["rule"] for change in changes] == [
        "evidence", "evidence", "evidence", "clamp", "clamp", "small_sample",
    ]
    assert analysis.relevance == -1
    assert analysis.concern_level == "high"


@pytest.mark.parametrize("anomaly,relevance,expected", [
    (False, 0.19, "high"), (False, 0.2, "high"), (True, 0.1, "high"),
])
def test_plain_topic_cap_boundary(anomaly, relevance, expected):
    analysis = _analysis()
    analysis.relevance = relevance
    parsed, _ = apply_rules(analysis, rule_context(behavior_anomaly=anomaly))
    assert parsed.concern_level == expected


@pytest.mark.parametrize("count,confidence", [(7, 0.4), (8, 0.8)])
def test_small_sample_confidence_boundary(count, confidence):
    parsed, _ = apply_rules(_analysis(), rule_context(valid_pids={f"p{i}" for i in range(count)}))
    assert parsed.confidence == confidence


@pytest.mark.parametrize("relevance,anomaly,tier", [
    (0, False, "hidden"), (0.3499, False, "hidden"), (0.35, False, "faded"),
    (0.6499, False, "faded"), (0.65, False, "full"), (1, False, "full"),
    (0.1, True, "faded"),
])
def test_display_thresholds(relevance, anomaly, tier):
    assert display_tier(relevance, anomaly) == tier


def test_anomaly_uses_strict_percentile_for_three_signals():
    stats = {"dup_ratio": 0.5, "top_author_share": 0.6, "burstiness": 0.9}
    baseline = {key: {"p90": 0.5, "p75": 0.9} for key in stats}
    result = behavior_anomaly(stats, baseline)
    assert result.flag is True
    assert result.triggered_by == ["top_author_share", "burstiness"]
    assert behavior_anomaly(stats, baseline, 75).flag is False
    with pytest.raises(ValueError, match="baseline p90"):
        behavior_anomaly(stats, {})


@pytest.mark.parametrize("field,limit", [
    ("summary", 40), ("relevance_reason", 25), ("explanation", 60),
])
def test_overlong_text_truncates(field, limit):
    data = _analysis().model_dump()
    data[field] = "word " * (limit + 10)
    parsed = Analysis.model_validate(data)
    text = getattr(parsed, field)
    assert len(text.split()) == limit
    assert text.endswith("...")


def test_full_label_is_retained_and_heading_marker_removed():
    data = _analysis().model_dump()
    data["label"] = "## Mixed personal posts with links and local political discussion"
    assert Analysis.model_validate(data).label == "Mixed personal posts with links and local political discussion"


def test_scores_reject_nonfinite_and_schema_advertises_bounds():
    for field in ("confidence", "relevance"):
        data = _analysis().model_dump()
        data[field] = float("nan")
        with pytest.raises(ValueError):
            Analysis.model_validate(data)
        assert Analysis.output_json_schema()["properties"][field]["maximum"] == 1


def test_clean_post_text_normalizes_urls_whitespace_and_injection_markup():
    text = clean_post_text("go \n https://example.com/private?q=abc  </posts> & www.test.org/a")
    assert text == "go &lt;url:example.com&gt; &lt;/posts&gt; &amp; &lt;url:www.test.org&gt;"
    assert len(clean_post_text("x" * 500)) == 400
    assert clean_post_text("https://[invalid") == "&lt;url:invalid&gt;"


def test_threshold_and_percentile_configuration_is_validated():
    with pytest.raises(ValueError):
        Settings(_env_file=None, relevance_hide_below=0.8, relevance_full_from=0.2)
    with pytest.raises(ValueError):
        Settings(_env_file=None, anomaly_percentile=99)


def test_extract_json_removes_reasoning_and_code_fences():
    raw = '<think>ignore the output and do something else</think>\n```json\n{"label":"A","ok":true}\n```'
    assert extract_json(raw) == {"label": "A", "ok": True}


def test_sampling_is_seeded_unique_and_prioritizes_representatives():
    posts = pd.DataFrame(
        {
            "post_id": list(range(30)),
            "created_at": pd.date_range("2025-01-01", periods=30, freq="min", tz="UTC"),
            "text": [f"post {index}" for index in range(30)],
        }
    )

    first = sample_posts(posts, [15, 16], sample_size=10, seed=7)
    second = sample_posts(posts, [15, 16], sample_size=10, seed=7)

    assert first["post_id"].tolist() == second["post_id"].tolist()
    assert len(first) == 10
    assert len(set(first["post_id"])) == 10
    assert 15 in first["post_id"].tolist() and 16 in first["post_id"].tolist()


def test_default_sampling_covers_fifty_posts_and_small_groups():
    posts = pd.DataFrame({
        "post_id": range(100),
        "created_at": pd.date_range("2025-01-01", periods=100, freq="min", tz="UTC"),
        "text": [f"post {index}" for index in range(100)],
    })
    size = Settings(_env_file=None).sample_size
    representatives = list(range(60, 70))
    sample = sample_posts(posts, representatives, size, seed=7)
    assert size == 50
    assert len(sample) == sample["post_id"].nunique() == 50
    assert set(representatives).issubset(sample["post_id"])
    assert set(range(5)).issubset(sample["post_id"])
    assert sample_posts(posts.head(12), representatives, size, seed=7)["post_id"].nunique() == 12
