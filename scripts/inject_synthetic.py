"""Inject clearly marked coordination/spam examples into an isolated copy, never the served data."""
import argparse
from pathlib import Path

import pandas as pd

from detector.config import Settings
from detector.io import read_json, read_posts, require_artifact, write_json
from detector.stats import run


def inject(source: Path, output: Path) -> list[int]:
    if source.resolve() == output.resolve():
        raise ValueError("Synthetic output must be separate from the source dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Synthetic output directory must be empty")
    posts = read_posts(source)
    assignments = pd.read_parquet(require_artifact(source / "assignments.parquet"))
    document = read_json(require_artifact(source / "clusters.json"))
    output.mkdir(parents=True, exist_ok=True)
    start_id = int(posts["post_id"].max()) + 1
    cluster_id = max((cluster["cluster_id"] for cluster in document["clusters"]), default=-1) + 1
    start = pd.to_datetime(posts["created_at"], utc=True).max() + pd.Timedelta(hours=2)
    synthetic_posts = []
    synthetic_assignments = []
    ids = []
    for offset, kind in enumerate(("swarm", "spam")):
        group_id = cluster_id + offset
        ids.append(group_id)
        representative_ids = []
        for index in range(50):
            post_id = start_id + len(synthetic_posts)
            representative_ids.append(post_id)
            synthetic_posts.append({
                "post_id": post_id, "uri": f"at://synthetic-{kind}/app.bsky.feed.post/{index}",
                "author": f"synthetic-{kind}-{index % 20 if kind == 'swarm' else 0}",
                "created_at": start + pd.Timedelta(seconds=index * 5),
                "text": f"SYNTHETIC TEST ONLY: share the public report about water quality in our local community right now. Test variant {index % 3}.",
                "has_images": False, "lang": "en", "is_reply": False,
            })
            synthetic_assignments.append({"post_id": post_id, "cluster_id": group_id})
        document["clusters"].append({
            "cluster_id": group_id, "size": 50, "synthetic": True,
            "representative_post_ids": representative_ids[:10],
        })
    pd.concat([posts, pd.DataFrame(synthetic_posts)], ignore_index=True).to_parquet(output / "posts.parquet", index=False)
    pd.concat([assignments, pd.DataFrame(synthetic_assignments)], ignore_index=True).to_parquet(output / "assignments.parquet", index=False)
    document["run"].update(synthetic_test_only=True, n_posts=len(posts) + 100, n_clusters=len(document["clusters"]))
    write_json(output / "clusters.json", document)
    run(Settings(_env_file=None, data_dir=output))
    result = read_json(output / "clusters.json")
    for cluster in result["clusters"]:
        if cluster["cluster_id"] in ids and not cluster["flags"]["behavior_anomaly"]["flag"]:
            raise AssertionError(
                f"Synthetic group {cluster['cluster_id']} did not exceed the source-derived baseline. "
                "Inspect the baseline rather than weakening it to force a positive result."
            )
    return ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Flagged synthetic groups: {inject(args.source, args.output)}. Test copy only: {args.output}")
