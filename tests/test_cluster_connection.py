from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from detector.config import Settings
from detector.io import read_json


def test_cluster_connection_refused_explains_qdrant_and_closes_client(tmp_path, monkeypatch):
    from detector import cluster

    client = Mock()
    monkeypatch.setattr(cluster, "read_posts", lambda _: pd.DataFrame({"post_id": [0, 1, 2]}))
    monkeypatch.setattr(cluster, "qdrant_client", lambda _: client)
    client.scroll.side_effect = ResponseHandlingException(httpx.ConnectError("refused"))
    with pytest.raises(RuntimeError, match="docker compose up -d qdrant"):
        cluster.run(Settings(_env_file=None, data_dir=tmp_path))
    client.close.assert_called_once()


def test_cluster_runs_statistics_automatically_without_changing_membership(tmp_path, monkeypatch):
    from detector import cluster

    posts = pd.DataFrame({
        "post_id": range(12), "author": [f"synthetic-{index}" for index in range(12)],
        "uri": [f"at://test/{index}" for index in range(12)],
        "text": ["A repeated test message about public transport"] * 12,
        "created_at": pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
    })
    posts.to_parquet(tmp_path / "posts.parquet", index=False)
    vectors = np.random.default_rng(42).normal(size=(12, 384)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([0] * 5 + [1] * 5 + [-1] * 2)
    monkeypatch.setattr(cluster, "_load_vectors", lambda *_: (np.arange(12), vectors))
    reducer, estimator = Mock(), Mock()
    reducer.fit_transform.return_value = np.zeros((12, 5))
    estimator.fit_predict.return_value = labels
    monkeypatch.setattr(cluster.umap, "UMAP", lambda **kwargs: reducer)
    monkeypatch.setattr(cluster, "HDBSCAN", lambda **kwargs: estimator)
    cluster.run(Settings(_env_file=None, data_dir=tmp_path))
    document = read_json(tmp_path / "clusters.json")
    assert document["run"]["stats_version"] == "1"
    assert document["dataset"]["n_posts"] == 12
    assert len(document["clusters"]) == 2
    assert all(row["stats"]["dup_ratio"] == 1 for row in document["clusters"])
    assert (tmp_path / "stats_report.json").is_file()
    np.testing.assert_array_equal(pd.read_parquet(tmp_path / "assignments.parquet")["cluster_id"], labels)
