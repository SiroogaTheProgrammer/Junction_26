from typing import Any, Literal

from detector.schema import Analysis, Verdict
from detector.stats import active_anomaly_stats, baseline_for, behavior_anomaly

Tier = Literal["full", "faded", "hidden"]


def display_tier(
    relevance: float, anomaly: bool, thresholds: dict[str, float] | None = None,
) -> Tier:
    thresholds = thresholds if thresholds is not None else {"hide_below": 0.35, "full_from": 0.65}
    hide_below, full_from = thresholds["hide_below"], thresholds["full_from"]
    if not 0 <= hide_below <= full_from <= 1:
        raise ValueError("Relevance thresholds must satisfy 0 <= hide_below <= full_from <= 1")
    if relevance >= full_from:
        return "full"
    if relevance >= hide_below or anomaly:
        return "faded"
    return "hidden"


def review_priority(
    parsed: dict[str, Any] | None, cluster: dict[str, Any], baseline: dict[str, Any],
    percentile: int = 90,
) -> dict[str, Any]:
    concern_points = {"none": 0, "low": 15, "medium": 39, "high": 60, "insufficient_evidence": 0}
    content = concern_points[parsed["concern_level"]] if parsed else 0
    cluster = cluster.get("stats", cluster)
    pct = baseline_for(int(cluster.get("size", 0)), baseline)["pct"]
    signals = []
    for metric in active_anomaly_stats(baseline):
        value = float(cluster[metric])
        threshold = float(pct[metric][f"p{percentile}"])
        if not 0 <= value <= 1 or not 0 <= threshold <= 1:
            raise ValueError(f"{metric} and its baseline must be between zero and one")
        excess = max(0.0, value - threshold) / (1 - threshold) if threshold < 1 else 0.0
        signals.append({
            "metric": metric, "value": value, "threshold": threshold,
            "normalized_excess": round(excess, 4),
        })
    if cluster.get("size", 0) < baseline.get("min_analysis_size", 0):
        for signal in signals:
            signal["normalized_excess"] = 0.0
    values = [signal["normalized_excess"] for signal in signals]
    spread = 40 * (max(values) + sum(values) / len(values)) / 2
    return {
        "score": round(content + spread),
        "content_points": content, "spread_points": round(spread, 1),
        "signals": signals, "method": "review-v2" if "global" in baseline else "review-v1",
        "burstiness_reliable": baseline.get("burstiness_reliable", True),
        "status": "complete" if parsed else "spread_only",
        "explanation": (
            "Review priority, not a probability of propaganda: final concern contributes "
            "0/15/39/60 points (none/low/medium/high); spread adds up to 40 points from "
            "duplicate text, author concentration and burstiness above the chosen baseline. "
            "Spread points = 20 * (largest normalized excess + mean normalized excess). "
            "Confidence and narrative relevance are shown separately."
        ),
    }


def apply_rules(
    analysis: Analysis, cluster: dict[str, Any],
) -> tuple[Analysis, list[dict[str, Any]]]:
    data = analysis.model_dump()
    changes: list[dict[str, Any]] = []

    def change(field: str, value: Any, rule: str) -> None:
        if data[field] != value:
            changes.append({"rule": rule, "field": field, "before": data[field], "after": value})
            data[field] = value

    valid = set(cluster["valid_pids"])
    for field in ("claims", "techniques"):
        kept = []
        for item in data[field]:
            evidence = [pid for pid in item["evidence"] if pid in valid]
            if evidence:
                kept.append({**item, "evidence": evidence})
        change(field, kept, "evidence")
    templating = dict(data["templating"])
    templating["evidence"] = [pid for pid in templating["evidence"] if pid in valid]
    if not templating["evidence"]:
        templating["detected"] = False
    change("templating", templating, "evidence")
    for field in ("relevance", "confidence"):
        change(field, min(1.0, max(0.0, data[field])), "clamp")
    if data["concern_level"] in {"medium", "high"} and not (data["claims"] or data["techniques"]):
        change("concern_level", "insufficient_evidence", "missing_content_evidence")
    if data["relevance"] < 0.2 and not cluster["behavior_anomaly"] and not (data["claims"] or data["techniques"]) and data["concern_level"] not in {"none", "insufficient_evidence"}:
        change("concern_level", "none", "plain_topic")
    if len(valid) < cluster.get("min_sample_for_confidence", 8):
        change("confidence", min(0.4, data["confidence"]), "small_sample")
    return Analysis.model_validate(data), changes


def apply_verdict_rules(
    verdict: Verdict, nearest_cloud: dict[str, Any] | None,
) -> tuple[Verdict, list[dict[str, Any]]]:
    """Contextual spread concern only; never downgrade the independent content assessment."""
    data = verdict.model_dump()
    changes: list[dict[str, Any]] = []

    def change(field: str, value: Any, rule: str) -> None:
        if data[field] != value:
            changes.append({"rule": rule, "field": field, "before": data[field], "after": value})
            data[field] = value

    for field in ("relevance", "confidence"):
        change(field, min(1.0, max(0.0, data[field])), "clamp")
    change("techniques", list(dict.fromkeys(data["techniques"])), "duplicates")
    if nearest_cloud is None:
        change("fits_cloud", False, "no_cloud")
    fits = data["fits_cloud"] and nearest_cloud is not None
    anomalous = bool(fits and nearest_cloud["behavior_anomaly"])
    cloud_concern = bool(fits and nearest_cloud["concern_level"] in {"medium", "high"})
    if data["concern_level"] in {"medium", "high"} and not (anomalous or cloud_concern):
        change("concern_level", "low", "no_spread_evidence")
    if data["relevance"] < 0.2 and not anomalous and data["concern_level"] not in {"none", "insufficient_evidence"}:
        change("concern_level", "none", "plain_topic")
    return Verdict.model_validate(data), changes
