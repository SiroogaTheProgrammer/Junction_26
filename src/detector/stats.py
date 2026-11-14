import hashlib
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from detector.config import Settings
from detector.io import read_json, read_posts, require_artifact, write_json
from detector.schema import AnomalyResult, Baseline, ClusterStats

logger = logging.getLogger(__name__)
STATS_VERSION = "1"
DOMAIN_PATTERN = re.compile(
    r"https?://[^\s<>]+|(?<![\w@/])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,24}(?:/[^\s<>]*)?",
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r"(?<!\w)@[\w.-]+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#(\w+)", re.UNICODE)
FILE_SUFFIXES = {"png", "jpg", "jpeg", "gif", "webp", "svg", "pdf", "txt", "json", "csv", "mp4", "zip", "py", "js"}
ANOMALY_STATS = ("dup_ratio", "top_author_share", "burstiness")
BASELINE_STATS = (
    "dup_ratio", "top_author_share", "top5_author_share", "burstiness",
    "top_domain_share", "top_hashtag_share", "authors_per_post",
)


def _links(text: str) -> list[re.Match[str]]:
    return [
        match for match in DOMAIN_PATTERN.finditer(text)
        if "://" in match.group() or match.group().split("/", 1)[0].rsplit(".", 1)[-1].lower() not in FILE_SUFFIXES
    ]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    for match in reversed(_links(text)):
        text = text[:match.start()] + " <url> " + text[match.end():]
    text = MENTION_PATTERN.sub(" ", text)
    # Preserve the URL marker, not arbitrary angle-bracketed material.
    parts = text.split("<url>")
    return " ".join(" <url> ".join(
        "".join(char if char.isalnum() else " " for char in part) for part in parts
    ).split())


def extract_domains(text: str) -> list[str]:
    result = set()
    for match in _links(unicodedata.normalize("NFKC", str(text))):
        value = match.group().rstrip(".,!?;:)]}")
        try:
            host = urlsplit(value if "://" in value else "https://" + value).hostname
        except ValueError:
            logger.warning("Ignoring malformed URL while extracting aggregate domains")
            continue
        if host:
            result.add(host.lower().removeprefix("www."))
    return sorted(result)


def extract_hashtags(text: str) -> list[str]:
    return sorted(set(HASHTAG_PATTERN.findall(unicodedata.normalize("NFKC", str(text)).lower())))


def near_duplicate_groups(texts: list[str], threshold: float = 0.8, num_perm: int = 64) -> list[int]:
    if not 0 < threshold <= 1 or num_perm < 16:
        raise ValueError("Near-duplicate threshold must be in (0, 1] and num_perm at least 16")
    normalized = [normalize_text(text) for text in texts]
    parents = list(range(len(texts)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    signatures: dict[int, MinHash] = {}
    try:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    except ValueError as exc:
        if str(exc) != "The number of bands are too small (b < 2)":
            raise
        # At very high thresholds the optimiser selects one band, which its constructor rejects.
        logger.debug("Using two LSH bands for threshold %s", threshold)
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, params=(2, num_perm // 2))
    for index, text in enumerate(normalized):
        if not text:
            continue
        if text in exact:
            union(index, exact[text])
            continue
        exact[text] = index
        if len(text) < 15:
            continue
        words = text.split()
        shingles = (
            {" ".join(words[start:start + 3]) for start in range(len(words) - 2)}
            if len(words) >= 8 else {text[start:start + 5] for start in range(len(text) - 4)}
        )
        signature = MinHash(num_perm=num_perm, seed=1)
        signature.update_batch([shingle.encode("utf-8") for shingle in sorted(shingles)])
        for candidate in sorted(lsh.query(signature)):
            if signature.jaccard(signatures[candidate]) >= threshold:
                union(index, candidate)
        signatures[index] = signature
        lsh.insert(index, signature)
    roots = [find(index) for index in range(len(texts))]
    sizes = Counter(roots)
    return [root if sizes[root] > 1 else -1 for root in roots]


def densest_window_share(timestamps: pd.Series, minutes: int = 10) -> float:
    if minutes <= 0:
        raise ValueError("Burst window must be positive")
    valid = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values()
    if valid.empty:
        return 0.0
    values = valid.astype("int64").to_numpy()
    window_ns = minutes * 60 * 1_000_000_000
    left = 0
    max_count = 0
    for right, value in enumerate(values):
        while value - values[left] > window_ns:
            left += 1
        max_count = max(max_count, right - left + 1)
    return float(max_count / len(timestamps))


def cluster_stats(posts: pd.DataFrame, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings if settings is not None else Settings(_env_file=None)
    size = len(posts)
    denominator = size or 1
    texts = posts["text"].fillna("").astype(str).tolist()
    authors = posts["author"].dropna().astype(str)
    author_counts = authors.value_counts()
    normalized = [normalize_text(text) for text in texts]
    counts = Counter(text for text in normalized if text)
    groups = near_duplicate_groups(texts, settings.near_dup_jaccard, settings.minhash_perm)
    duplicate_mask = np.asarray(groups) >= 0
    domains = [extract_domains(text) for text in texts]
    domain_counts = Counter(domain for items in domains for domain in items)
    hashtag_counts = Counter(tag for text in texts for tag in extract_hashtags(text))
    top_domain, domain_count = domain_counts.most_common(1)[0] if domain_counts else (None, 0)
    top_hashtag, hashtag_count = hashtag_counts.most_common(1)[0] if hashtag_counts else (None, 0)
    timestamps = pd.to_datetime(posts["created_at"], utc=True, errors="coerce").dropna()
    first = timestamps.min() if not timestamps.empty else None
    last = timestamps.max() if not timestamps.empty else None
    images = posts["has_images"].dropna() if "has_images" in posts else pd.Series(dtype=bool)
    data = {
        "size": size, "n_authors": int(authors.nunique()),
        "authors_per_post": float(authors.nunique() / denominator),
        "top_author_share": float(author_counts.iloc[0] / denominator) if len(author_counts) else 0.0,
        "top5_author_share": float(author_counts.head(5).sum() / denominator),
        "exact_dup_ratio": sum(bool(text) and counts[text] > 1 for text in normalized) / denominator,
        "dup_ratio": float(duplicate_mask.sum() / denominator),
        "dup_authors": int(posts.loc[duplicate_mask, "author"].dropna().nunique()),
        "largest_dup_group": max(Counter(group for group in groups if group >= 0).values(), default=0),
        "burstiness": densest_window_share(posts["created_at"], settings.burst_window_min),
        "time_span_min": (last - first).total_seconds() / 60 if first is not None else 0.0,
        "first_post_at": first.isoformat() if first is not None else None,
        "last_post_at": last.isoformat() if last is not None else None,
        "top_domain": top_domain, "top_domain_share": domain_count / denominator,
        "link_share": sum(bool(items) for items in domains) / denominator,
        "top_hashtag": top_hashtag, "top_hashtag_share": hashtag_count / denominator,
        "image_share": float(images.astype(bool).mean()) if len(images) == size and size else None,
    }
    return ClusterStats.model_validate({
        key: round(value, 4) if isinstance(value, float) else value for key, value in data.items()
    }).model_dump()


def compute_baseline(clusters: list[dict]) -> dict[str, dict[str, float]]:
    baseline = {}
    for name in BASELINE_STATS:
        values = [float(cluster.get("stats", cluster).get(name, 0.0)) for cluster in clusters]
        percentiles = np.percentile(values, [50, 75, 90]) if values else [0.0, 0.0, 0.0]
        baseline[name] = {key: round(float(value), 4) for key, value in zip(("p50", "p75", "p90"), percentiles)}
    return baseline


def build_baseline(
    clusters: list[dict], settings: Settings, burstiness_reliable: bool = True,
) -> dict[str, Any]:
    global_pct = compute_baseline(clusters)
    bounds = [int(value) for value in settings.baseline_buckets.split(",")]
    buckets = []
    for index, low in enumerate(bounds):
        high = bounds[index + 1] - 1 if index + 1 < len(bounds) else None
        selected = [cluster for cluster in clusters if cluster["size"] >= low and (high is None or cluster["size"] <= high)]
        local = len(selected) >= settings.min_clusters_per_bucket
        buckets.append({
            "min_size": low, "max_size": high, "n_clusters": len(selected),
            "source": "bucket" if local else "global",
            "pct": compute_baseline(selected) if local else global_pct,
        })
    return Baseline.model_validate({
        "global": {"n_clusters": len(clusters), "pct": global_pct}, "buckets": buckets,
        "burstiness_reliable": burstiness_reliable, "burst_window_min": settings.burst_window_min,
        "min_analysis_size": settings.min_analysis_size,
    }).model_dump(by_alias=True)


def baseline_for(size: int, baseline: dict[str, Any]) -> dict[str, Any]:
    if "global" not in baseline:
        return {"pct": baseline, "name": "legacy-global", "source": "global", "n_clusters": None}
    for bucket in baseline["buckets"]:
        if size >= bucket["min_size"] and (bucket["max_size"] is None or size <= bucket["max_size"]):
            high = bucket["max_size"]
            return {
                "pct": bucket["pct"], "name": f"{bucket['min_size']}-{high}" if high is not None else f"{bucket['min_size']}+",
                "source": bucket["source"], "n_clusters": bucket["n_clusters"],
                "reference_n_clusters": bucket["n_clusters"] if bucket["source"] == "bucket" else baseline["global"]["n_clusters"],
            }
    return {**baseline["global"], "name": "global", "source": "global"}


def active_anomaly_stats(baseline: dict[str, Any]) -> tuple[str, ...]:
    return ANOMALY_STATS if baseline.get("burstiness_reliable", True) else ANOMALY_STATS[:2]


def behavior_anomaly(cluster: dict[str, Any], baseline: dict[str, Any], percentile: int = 90) -> AnomalyResult:
    stats = cluster.get("stats", cluster)
    pct = baseline_for(int(stats.get("size", 0)), baseline)["pct"]
    key = f"p{percentile}"
    triggered = []
    for stat in active_anomaly_stats(baseline):
        if stat not in stats or key not in pct.get(stat, {}):
            raise ValueError(f"Behavior anomaly requires {stat} and its baseline {key}")
        if stats[stat] > pct[stat][key]:
            triggered.append(stat)
    if stats.get("size", 0) < baseline.get("min_analysis_size", 0):
        triggered = []
    return AnomalyResult(flag=bool(triggered), triggered_by=triggered)


def is_organic_looking(stats: dict, baseline: dict[str, Any]) -> bool:
    stats = stats.get("stats", stats)
    pct = baseline_for(int(stats.get("size", 0)), baseline)["pct"]
    return all(float(stats[name]) <= float(pct[name]["p75"]) for name in active_anomaly_stats(baseline))


def describe_signals(stats: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    pct = baseline_for(int(stats["size"]), baseline)["pct"]
    labels = {
        "dup_ratio": "near-duplicate posts", "top_author_share": "posts from the top account",
        "burstiness": f"posts in one {baseline.get('burst_window_min', 10)}-minute window",
    }
    lines = [
        f"{stats[name]:.0%} {labels[name]} (similar clusters: p90 {pct[name]['p90']:.0%})"
        for name in active_anomaly_stats(baseline) if stats[name] > pct[name]["p90"]
    ]
    if stats.get("dup_authors", 0) >= 5:
        lines.append(f"near-identical text from {stats['dup_authors']} different accounts")
    return lines


def run(settings: Settings, inspect: bool = False) -> None:
    started = time.monotonic()
    posts = read_posts(settings.data_path)
    assignments = pd.read_parquet(require_artifact(settings.data_path / "assignments.parquet"))
    path = require_artifact(settings.data_path / "clusters.json")
    document = read_json(path)
    if posts["post_id"].duplicated().any() or assignments["post_id"].duplicated().any():
        raise ValueError("Posts and assignments must have unique post ids")
    joined = posts.merge(assignments, on="post_id", validate="one_to_one")
    if len(joined) != len(posts) or len(assignments) != len(posts):
        raise ValueError("Assignments must contain exactly one entry for every post")
    groups = {int(key): group for key, group in joined.groupby("cluster_id") if int(key) != -1}
    ids = [int(cluster["cluster_id"]) for cluster in document["clusters"]]
    if len(set(ids)) != len(ids) or set(groups) != set(ids):
        raise ValueError("Cluster records do not match assignments")
    timestamps = pd.to_datetime(posts["created_at"], utc=True, errors="coerce").dropna()
    span = float((timestamps.max() - timestamps.min()).total_seconds() / 60) if len(timestamps) else 0.0
    reliable = span >= 6 * settings.burst_window_min
    stats_started = time.monotonic()
    for cluster in document["clusters"]:
        group = groups[int(cluster["cluster_id"])]
        if len(group) != cluster["size"]:
            raise ValueError(f"Cluster {cluster['cluster_id']} size does not match assignments")
        cluster["stats"] = cluster_stats(group, settings)
        cluster.update(cluster["stats"])
    stats_seconds = time.monotonic() - stats_started
    baseline_started = time.monotonic()
    baseline = build_baseline(document["clusters"], settings, reliable)
    triggers: Counter[str] = Counter()
    for cluster in document["clusters"]:
        anomaly = behavior_anomaly(cluster, baseline, settings.anomaly_percentile)
        organic = is_organic_looking(cluster, baseline)
        cluster.update({
            "baseline_bucket": baseline_for(cluster["size"], baseline)["name"],
            "flags": {"organic_looking": organic, "behavior_anomaly": anomaly.model_dump()},
            "organic_looking": organic, "signals": describe_signals(cluster["stats"], baseline),
        })
        cluster["stats_hash"] = hashlib.sha256(json.dumps({
            "stats": cluster["stats"], "baseline": baseline_for(cluster["size"], baseline),
            "flags": cluster["flags"], "burstiness_reliable": reliable,
            "burst_window_min": settings.burst_window_min,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        triggers.update(anomaly.triggered_by)
    document["baseline"] = baseline
    document["dataset"] = {
        "n_posts": len(posts), "time_span_min": round(span, 4), "burstiness_reliable": reliable,
        "burst_window_min": settings.burst_window_min,
        "burstiness_note": None if reliable else "Dataset spans less than six burst windows; timing is displayed but excluded from flags and priority.",
        "image_metadata_available": "has_images" in posts and bool(posts["has_images"].notna().all()),
    }
    document["run"]["stats_version"] = STATS_VERSION
    document["run"]["stats_parameters"] = {
        key: getattr(settings, key) for key in (
            "burst_window_min", "near_dup_jaccard", "minhash_perm", "baseline_buckets",
            "min_clusters_per_bucket", "anomaly_percentile", "min_analysis_size",
        )
    }
    write_json(path, document)
    write_json(settings.data_path / "stats_report.json", {
        "n_clusters": len(ids),
        "anomalous": sum(cluster["flags"]["behavior_anomaly"]["flag"] for cluster in document["clusters"]),
        "organic_looking": sum(cluster["organic_looking"] for cluster in document["clusters"]),
        "counts_per_trigger": dict(triggers), **document["dataset"],
        "global_fallback_buckets": [
            baseline_for(bucket["min_size"], baseline)["name"]
            for bucket in baseline["buckets"] if bucket["source"] == "global"
        ],
        "seconds": {
            "cluster_stats": round(stats_seconds, 3),
            "baseline_and_flags": round(time.monotonic() - baseline_started, 3),
            "total": round(time.monotonic() - started, 3),
        },
    })
    logger.info("Updated statistics for %d clusters; burstiness reliable=%s", len(ids), reliable)
    if not reliable:
        logger.warning(document["dataset"]["burstiness_note"])
    if inspect:
        for metric in ANOMALY_STATS:
            print(f"\nTop groups by {metric}:")
            for cluster in sorted(document["clusters"], key=lambda row: (-row[metric], row["cluster_id"]))[:10]:
                print(f"  group {cluster['cluster_id']}: {cluster[metric]:.4f}")
                for text in groups[cluster["cluster_id"]].sort_values("post_id")["text"].head(3):
                    print(f"    {str(text)[:300]}")


# Existing pipeline callers retain these names.
compute_stats = cluster_stats
