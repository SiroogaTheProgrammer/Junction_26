import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detector.io import write_jsonl
from detector.prompts import PROMPT_VERSION
from detector.projection import project_with_model
from detector.rules import behavior_anomaly
from detector.stats import compute_baseline, compute_stats, is_organic_looking


FIXTURE_DIR = Path(__file__).resolve().parent
TOPICS = (
    ("community transit", "Neighbors discuss a local transit change", 60, "none", 0.92),
    ("coastal weather", "Residents share coastal weather updates", 50, "low", 0.82),
    ("public libraries", "Readers talk about library programs", 45, "medium", 0.78),
    ("shared campaign slogans", "Posts repeat a campaign slogan", 40, "high", 0.88),
    ("breaking news headlines", "Posts share similar news headlines", 35, "low", 0.35),
    ("garden updates", "Gardeners compare seasonal updates", 30, "insufficient_evidence", 0.42),
    ("weekend events", "People discuss weekend events", 25, "failed", None),
    ("local cycling", "Cyclists share route suggestions", 15, "not_analysed", None),
)


def generate() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    assignment_rows = []
    for cluster_id, (topic, body, size, _, _) in enumerate(TOPICS):
        for offset in range(size):
            post_id = len(rows)
            rows.append(
                {
                    "post_id": post_id,
                    "uri": f"at://fixture/post/{post_id}",
                    "author": "garden-bot" if cluster_id == 5 else f"user-{offset % 40}",
                    "created_at": start + timedelta(minutes=cluster_id * 60 + (0 if cluster_id == 5 else offset)),
                    "text": body if cluster_id == 5 else f"{body}: update {offset % 17}. #{topic.replace(' ', '')}",
                    "has_images": offset % 8 == 0,
                }
            )
            assignment_rows.append({"post_id": post_id, "cluster_id": cluster_id})

    posts = pd.DataFrame(rows)
    assignments = pd.DataFrame(assignment_rows)
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(len(TOPICS), 384))
    vectors = centers[assignments["cluster_id"].to_numpy()] + rng.normal(size=(len(posts), 384)) * 0.1
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    projection, projection_meta, projection_model = project_with_model(
        posts["post_id"].to_numpy(), vectors, assignments["cluster_id"].to_numpy(), 42,
    )
    projection_meta["fixture_only"] = True
    projection.to_parquet(FIXTURE_DIR / "projection.parquet", index=False)
    projection_model.save(FIXTURE_DIR / "projection_model.npz")
    np.save(FIXTURE_DIR / "embeddings.npy", vectors.astype(np.float32))
    clusters = []
    for cluster_id in range(len(TOPICS)):
        group = posts.loc[assignments["cluster_id"].eq(cluster_id)]
        clusters.append(
            {
                "cluster_id": cluster_id,
                **compute_stats(group),
                "representative_post_ids": group["post_id"].head(10).astype(int).tolist(),
            }
        )
    baseline = compute_baseline(clusters)
    for cluster in clusters:
        cluster["organic_looking"] = is_organic_looking(cluster, baseline)
    clusters.sort(key=lambda item: (-item["size"], item["cluster_id"]))
    posts.to_parquet(FIXTURE_DIR / "posts.parquet", index=False)
    assignments.to_parquet(FIXTURE_DIR / "assignments.parquet", index=False)
    (FIXTURE_DIR / "clusters.json").write_text(
        json.dumps(
            {
                "run": {
                    "n_posts": len(posts),
                    "n_clusters": len(clusters),
                    "noise_ratio": 0.0,
                    "projection": projection_meta,
                    "parameters": {"embedding_model": "intfloat/multilingual-e5-small"},
                    "seconds": {"total": 12.5},
                },
                "baseline": baseline,
                "clusters": clusters,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    analyses = []
    for cluster_id, (topic, body, _, concern, confidence) in enumerate(TOPICS):
        if concern == "not_analysed":
            continue
        sample_ids = assignments.loc[assignments["cluster_id"].eq(cluster_id), "post_id"].head(50).tolist()
        kind, relevance, reason = (
            ("issue_discussion", 0.55, "Mixed opinions about a local public service."),
            ("personal_chatter", 0.1, "Weather updates without a shared public-affairs framing."),
            ("narrative", 0.72, "Repeated claims about public library policy."),
            ("narrative", 0.92, "Campaign framing with repeated calls to action."),
            ("news_relay", 0.5, "Headlines relay an event rather than advocate one shared framing."),
            ("plain_topic", 0.1, "Repeated garden updates have no public-affairs framing."),
            ("unclear", 0.0, "Analysis failed."),
            ("unclear", 0.0, "Not analysed."),
        )[cluster_id]
        parsed = {
            "label": topic.title(),
            "summary": body + ".",
            "relevance_kind": kind,
            "relevance": relevance,
            "relevance_reason": reason,
            "claims": [] if relevance < 0.2 else [{"text": body + ".", "kind": "opinion", "evidence": ["p1", "p2"]}],
            "techniques": (
                [{"name": "call_to_mass_action", "evidence": ["p1"]}] if concern == "high" else []
            ),
            "templating": {"detected": concern == "high", "evidence": ["p1", "p2"] if concern == "high" else []},
            "framing_diversity": "medium",
            "legit_lookalike": "Syndicated news headlines" if cluster_id == 4 else None,
            "concern_level": concern,
            "confidence": confidence,
            "explanation": "The sampled posts share a subject, but this alone does not establish coordination.",
        } if concern != "failed" else None
        analyses.append(
            {
                "cluster_id": cluster_id,
                "input_hash": f"fixture-{cluster_id}",
                "prompt_version": PROMPT_VERSION,
                "status": "failed" if concern == "failed" else "ok",
                "sample": [
                    {"pid": f"p{index + 1}", "post_id": post_id}
                    for index, post_id in enumerate(sample_ids)
                ],
                "system_prompt": "Fixture system prompt",
                "user_prompt": "Fixture user prompt",
                "raw_response": json.dumps(parsed) if parsed else "Fixture: model returned an empty answer.",
                "parsed": parsed,
                "behavior_anomaly": behavior_anomaly(clusters[cluster_id], baseline).model_dump(),
                "concern_level_model": "high" if cluster_id == 4 else concern if parsed else None,
                "relevance_model": relevance if parsed else None,
                "capped_by": "lookalike" if cluster_id == 4 else None,
                "rule_log": (
                    [{"rule": "lookalike", "field": "concern_level", "before": "high", "after": "low"}]
                    if cluster_id == 4 else []
                ),
                "attempts": 1,
                "latency_s": 1.2,
                "model": "nemotron-lightning",
                "created_at": "2025-01-02T00:00:00+00:00",
            }
        )
    write_jsonl(FIXTURE_DIR / "analyses.jsonl", analyses)


if __name__ == "__main__":
    generate()
