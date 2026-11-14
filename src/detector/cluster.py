import logging
import time

import httpx
import numpy as np
import pandas as pd
import umap
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from sklearn.cluster import HDBSCAN

from detector.config import Settings
from detector.io import ensure_data_dir, qdrant_client, read_json, read_posts, require_artifact, write_json
from detector.projection import ProjectionModel, project_with_model
from detector.stats import run as run_stats

logger = logging.getLogger(__name__)


def _read_vectors(client: QdrantClient, settings: Settings, expected: int) -> tuple[np.ndarray, np.ndarray]:
    points = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            offset=offset,
            with_payload=["post_id"],
            with_vectors=True,
        )
        points.extend(page)
        logger.info("Read %d/%d vectors", len(points), expected)
        if offset is None:
            break
    ordered = sorted(points, key=lambda point: int(point.payload["post_id"]))
    post_ids = np.asarray([int(point.payload["post_id"]) for point in ordered], dtype=np.int64)
    vectors = np.asarray([point.vector for point in ordered], dtype=np.float32)
    if len(ordered) != expected or vectors.shape != (expected, 384):
        raise ValueError(
            f"Qdrant contains {len(ordered)} vectors with shape {vectors.shape}; expected {expected} vectors of dimension 384"
        )
    return post_ids, vectors


def _load_vectors(settings: Settings, posts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if len(posts) < 3:
        raise ValueError("Clustering requires at least 3 posts")
    client = None
    try:
        client = qdrant_client(settings)
        post_ids, vectors = _read_vectors(client, settings, len(posts))
    except ResponseHandlingException as exc:
        if not isinstance(exc.source, httpx.ConnectError):
            raise
        raise RuntimeError(
            "Cannot connect to Qdrant. Start it with `docker compose up -d qdrant` "
            "or start your native Qdrant service, then retry with the data folder and "
            "collection that belong together. Serving an existing projection needs no Qdrant."
        ) from exc
    finally:
        if client is not None:
            client.close()
    expected_ids = posts["post_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(post_ids, np.sort(expected_ids)):
        raise ValueError("Qdrant post ids do not match posts.parquet")
    return post_ids, vectors


def _write_projection(
    settings: Settings, points: pd.DataFrame, model: ProjectionModel, vectors: np.ndarray,
) -> None:
    # The projection, its model and the embeddings must describe the same posts in the same order.
    points.to_parquet(settings.data_path / "projection.parquet", index=False)
    model.save(settings.data_path / "projection_model.npz")
    temporary = settings.data_path / "embeddings.tmp"
    with temporary.open("wb") as file:
        np.save(file, np.ascontiguousarray(vectors, dtype=np.float32))
    temporary.replace(settings.data_path / "embeddings.npy")


def project(settings: Settings) -> None:
    """Rebuild the display projection and placement artifacts without changing cloud membership."""
    posts = read_posts(settings.data_path)
    clusters_path = require_artifact(settings.data_path / "clusters.json")
    assignments = pd.read_parquet(require_artifact(settings.data_path / "assignments.parquet"))
    post_ids, vectors = _load_vectors(settings, posts)
    labels = assignments.set_index("post_id")["cluster_id"].reindex(post_ids)
    if labels.isna().any() or len(assignments) != len(post_ids):
        raise ValueError("assignments.parquet does not match the stored vectors; rerun cluster")
    points, projection, model = project_with_model(
        post_ids, vectors, labels.to_numpy(dtype=np.int64), settings.seed,
    )
    _write_projection(settings, points, model, vectors)
    document = read_json(clusters_path)
    document["run"]["projection"] = projection
    write_json(clusters_path, document)
    logger.info("Wrote projection.parquet, projection_model.npz and embeddings.npy for %d posts", len(points))


def run(settings: Settings) -> None:
    ensure_data_dir(settings.data_path)
    started = time.monotonic()
    posts = read_posts(settings.data_path)
    post_ids, vectors = _load_vectors(settings, posts)

    step_started = time.monotonic()
    n_neighbors = min(settings.umap_n_neighbors, len(posts) - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=settings.umap_n_components,
        min_dist=settings.umap_min_dist,
        metric="cosine",
        random_state=settings.seed,
    )
    reduced = reducer.fit_transform(vectors)
    umap_seconds = time.monotonic() - step_started
    logger.info("UMAP reduced %d posts in %.1fs", len(posts), umap_seconds)

    step_started = time.monotonic()
    estimator = HDBSCAN(
        min_cluster_size=min(settings.hdbscan_min_cluster_size, len(posts)),
        min_samples=min(settings.hdbscan_min_samples, len(posts) - 1),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = estimator.fit_predict(reduced).astype(np.int64)
    hdbscan_seconds = time.monotonic() - step_started
    logger.info("HDBSCAN found %d non-noise clusters in %.1fs", len(set(labels) - {-1}), hdbscan_seconds)
    points, projection, model = project_with_model(post_ids, vectors, labels, settings.seed)
    _write_projection(settings, points, model, vectors)

    assignments = pd.DataFrame({"post_id": post_ids, "cluster_id": labels})
    previous_path = settings.data_path / "assignments.parquet"
    if not previous_path.is_file() or not pd.read_parquet(previous_path).sort_values("post_id").reset_index(drop=True).equals(assignments):
        for name in ("analyses.jsonl", "analyze_report.json"):
            (settings.data_path / name).unlink(missing_ok=True)
        logger.info("Cloud membership changed; cleared obsolete analysis records")
    assignments.to_parquet(settings.data_path / "assignments.parquet", index=False)
    aligned_posts = posts.set_index("post_id").loc[post_ids].reset_index()
    clusters: list[dict] = []
    for cluster_id in sorted(set(labels) - {-1}):
        positions = np.flatnonzero(labels == cluster_id)
        group = aligned_posts.iloc[positions]
        centroid = vectors[positions].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm:
            centroid /= norm
        similarities = vectors[positions] @ centroid
        representative_positions = positions[np.argsort(-similarities, kind="stable")[:10]]
        record = {
            "cluster_id": int(cluster_id),
            "size": len(group),
            "representative_post_ids": [int(post_ids[position]) for position in representative_positions],
        }
        clusters.append(record)

    clusters.sort(key=lambda item: (-item["size"], item["cluster_id"]))
    elapsed = time.monotonic() - started
    noise_ratio = float(np.count_nonzero(labels == -1) / len(labels))
    write_json(
        settings.data_path / "clusters.json",
        {
            "run": {
                "n_posts": len(posts),
                "n_clusters": len(clusters),
                "noise_ratio": noise_ratio,
                "projection": projection,
                "parameters": {
                    "embedding_model": settings.embed_model,
                    "umap_n_neighbors": n_neighbors,
                    "umap_n_components": settings.umap_n_components,
                    "umap_min_dist": settings.umap_min_dist,
                    "hdbscan_min_cluster_size": settings.hdbscan_min_cluster_size,
                    "hdbscan_min_samples": settings.hdbscan_min_samples,
                    "seed": settings.seed,
                },
                "seconds": {
                    "read_vectors": round(max(0.0, elapsed - umap_seconds - hdbscan_seconds), 3),
                    "umap": round(umap_seconds, 3),
                    "hdbscan": round(hdbscan_seconds, 3),
                    "total": round(elapsed, 3),
                },
            },
            "clusters": clusters,
        },
    )
    run_stats(settings)
    logger.info("Wrote clusters.json and assignments.parquet")
