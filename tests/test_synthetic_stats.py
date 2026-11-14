import numpy as np
import pandas as pd
import pytest

from detector.io import read_json, write_json
from scripts.inject_synthetic import inject


def test_synthetic_swarm_and_spam_are_flagged_without_modifying_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "test-copy"
    source.mkdir()
    rng = np.random.default_rng(42)
    posts, assignments, clusters = [], [], []
    vocabulary = [f"word{index}" for index in range(5000)]
    for group_id in range(25):
        ids = []
        for index in range(50):
            post_id = len(posts)
            ids.append(post_id)
            posts.append({
                "post_id": post_id, "uri": f"at://fixture/{post_id}", "author": f"ordinary-{post_id}",
                "created_at": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=post_id * 12),
                "text": " ".join(rng.choice(vocabulary, 20, replace=False)),
                "has_images": False,
            })
            assignments.append({"post_id": post_id, "cluster_id": group_id})
        clusters.append({"cluster_id": group_id, "size": 50, "representative_post_ids": ids[:3]})
    pd.DataFrame(posts).to_parquet(source / "posts.parquet", index=False)
    pd.DataFrame(assignments).to_parquet(source / "assignments.parquet", index=False)
    write_json(source / "clusters.json", {"run": {}, "clusters": clusters})
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    ids = inject(source, output)
    document = read_json(output / "clusters.json")
    targets = [row for row in document["clusters"] if row["cluster_id"] in ids]
    assert len(targets) == 2
    assert all(row["synthetic"] and row["flags"]["behavior_anomaly"]["flag"] for row in targets)
    assert targets[0]["dup_authors"] == 20
    assert targets[1]["dup_authors"] == 1
    assert targets[1]["top_author_share"] == 1
    assert all(path.read_bytes() == before[path.name] for path in source.iterdir())
    with pytest.raises(ValueError, match="separate"):
        inject(source, source)
    with pytest.raises(ValueError, match="empty"):
        inject(source, output)
