import numpy as np
import pytest

from detector.projection import project_embeddings


def test_real_projection_is_deterministic_bounded_and_preserves_post_mapping():
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(80, 384))
    ids = np.arange(80)[::-1]
    labels = np.tile([-1, 0, 1, 2], 20)
    points, metadata = project_embeddings(ids, vectors, labels, 42)
    repeated, repeated_meta = project_embeddings(ids, vectors, labels, 42)
    assert points.equals(repeated)
    assert metadata == repeated_meta
    assert points["post_id"].tolist() == ids.tolist()
    assert points["cluster_id"].tolist() == labels.tolist()
    assert np.isfinite(points[["x", "y", "z"]]).all().all()
    assert np.linalg.norm(points[["x", "y", "z"]], axis=1).max() <= 1.00001
    assert metadata["source_dimensions"] == 384
    assert 0 < metadata["explained_variance_2d"] <= metadata["explained_variance_3d"] <= 1
    assert metadata["explained_variance_3d"] < 0.5


def test_projection_positions_follow_embeddings_not_arbitrary_ids():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(40, 8))
    points, _ = project_embeddings(np.arange(40), vectors, np.zeros(40), 1)
    swapped, _ = project_embeddings(np.arange(40) + 99, vectors, np.ones(40), 1)
    np.testing.assert_allclose(points[["x", "y", "z"]], swapped[["x", "y", "z"]])
    permuted, _ = project_embeddings(np.arange(40), vectors[::-1], np.zeros(40), 1)
    np.testing.assert_allclose(points[["x", "y", "z"]], permuted[["x", "y", "z"]].iloc[::-1], atol=1e-5)


@pytest.mark.parametrize("vectors,ids,labels", [
    (np.ones((2, 384)), np.arange(2), np.arange(2)),
    (np.ones((4, 384)), np.arange(3), np.arange(4)),
    (np.full((4, 384), np.nan), np.arange(4), np.arange(4)),
    (np.ones((4, 384)), np.arange(4), np.arange(4)),
    (np.eye(4), np.array([0, 0, 1, 2]), np.arange(4)),
])
def test_projection_rejects_invalid_inputs(vectors, ids, labels):
    with pytest.raises(ValueError):
        project_embeddings(ids, vectors, labels, 42)