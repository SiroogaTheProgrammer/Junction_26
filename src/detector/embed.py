import logging
import time
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient, models

from detector.config import Settings
from detector.embedding import (
    VECTOR_SIZE,
    choose_device,
    clean_for_embedding,
    encode as _encode,
    load_model as _model,
)
from detector.io import (
    dataset_fingerprint,
    ensure_data_dir,
    qdrant_client,
    read_json,
    read_posts,
    write_json,
)

logger = logging.getLogger(__name__)
UPSERT_BATCH = 512


def _embedding_parameters(settings: Settings) -> dict[str, Any]:
    return {
        "model": settings.embed_model, "max_tokens": settings.embed_max_tokens,
        "dimensions": VECTOR_SIZE, "preprocessing": "url-whitespace-query-prefix-v1",
    }


def _existing_ids(client: QdrantClient, collection: str) -> set[int]:
    existing: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=["post_id"],
            with_vectors=False,
        )
        existing.update(int(point.id) for point in points)
        if offset is None:
            return existing


def _ensure_collection(client: QdrantClient, settings: Settings, fingerprint: str, recreate: bool) -> set[int]:
    collection = settings.qdrant_collection
    collections = {item.name for item in client.get_collections().collections}
    if collection in collections and recreate:
        client.delete_collection(collection)
        collections.remove(collection)

    old_fingerprint = None
    if collection in collections:
        metadata = read_json(settings.data_path / "embed_report.json", {})
        old_fingerprint = metadata.get("fingerprint")
        if old_fingerprint != fingerprint or metadata.get("embedding") != _embedding_parameters(settings):
            logger.info("Dataset or embedding configuration changed/unverified; recreating Qdrant collection")
            client.delete_collection(collection)
            collections.remove(collection)

    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        write_json(settings.data_path / "embed_report.json", {
            "fingerprint": fingerprint, "embedding": _embedding_parameters(settings),
        })
        return set()
    return _existing_ids(client, collection)


def benchmark(settings: Settings, number: int) -> None:
    posts = read_posts(settings.data_path)
    if len(posts) < number:
        raise ValueError(f"Benchmark needs {number} posts, but {len(posts)} are available in posts.parquet")
    device = choose_device(settings.embed_device)
    model = _model(settings, device)
    texts = ["query: " + clean_for_embedding(text) for text in posts["text"].head(number)]
    started = time.monotonic()
    _encode(model, texts, settings.embed_batch)
    elapsed = time.monotonic() - started
    rate = number / elapsed if elapsed else 0
    print(f"device={device} posts={number} posts_per_second={rate:.2f}")
    print(f"projected_N_POSTS={settings.n_posts} projected_seconds={settings.n_posts / rate:.1f}")


def run(settings: Settings, recreate: bool = False) -> None:
    started = time.monotonic()
    ensure_data_dir(settings.data_path)
    posts = read_posts(settings.data_path)
    if posts.empty:
        raise ValueError("posts.parquet contains no posts")
    load_report = read_json(settings.data_path / "load_report.json", {})
    fingerprint = load_report.get("fingerprint") or dataset_fingerprint(
        posts["uri"].astype(str).tolist(), settings.n_posts, settings.seed
    )
    device = choose_device(settings.embed_device)
    model = _model(settings, device)
    client = qdrant_client(settings)
    existing = _ensure_collection(client, settings, fingerprint, recreate)
    missing_rows = [index for index, post_id in enumerate(posts["post_id"]) if int(post_id) not in existing]
    logger.info(
        "Embedding %d posts (%d already stored) on %s",
        len(missing_rows),
        len(existing),
        device,
    )

    embedded = 0
    embedding_started = time.monotonic()
    for start in range(0, len(missing_rows), settings.embed_batch):
        indexes = missing_rows[start : start + settings.embed_batch]
        batch = posts.iloc[indexes]
        texts = ["query: " + clean_for_embedding(text) for text in batch["text"]]
        vectors = _encode(model, texts, settings.embed_batch)
        points = [
            models.PointStruct(
                id=int(post_id),
                vector=vector.tolist(),
                payload={"post_id": int(post_id)},
            )
            for post_id, vector in zip(batch["post_id"], vectors, strict=True)
        ]
        for offset in range(0, len(points), UPSERT_BATCH):
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points[offset : offset + UPSERT_BATCH],
                wait=True,
            )
        embedded += len(points)
        logger.info("Embedded %d/%d missing posts", embedded, len(missing_rows))

    point_count = client.count(collection_name=settings.qdrant_collection, exact=True).count
    print(f"Qdrant point count: {point_count} (expected {len(posts)})")
    if point_count != len(posts):
        raise ValueError("Qdrant point count does not equal posts.parquet row count; rerun embed --recreate")
    _print_neighbours(client, settings, posts)

    embedding_elapsed = time.monotonic() - embedding_started
    elapsed = time.monotonic() - started
    rate = embedded / embedding_elapsed if embedding_elapsed else 0.0
    report: dict[str, Any] = {
        "fingerprint": fingerprint,
        "embedding": _embedding_parameters(settings),
        "n_posts": len(posts),
        "n_embedded_this_run": embedded,
        "device": device,
        "posts_per_second": round(rate, 3),
        "elapsed_s": round(embedding_elapsed, 3),
        "total_elapsed_s": round(elapsed, 3),
    }
    write_json(settings.data_path / "embed_report.json", report)
    logger.info("Embedding stage finished in %.1fs (%.2f posts/s)", elapsed, rate)


def _print_neighbours(client: QdrantClient, settings: Settings, posts: pd.DataFrame) -> None:
    if posts.empty:
        return
    random_rows = posts.sample(n=min(3, len(posts)), random_state=settings.seed)
    lookup = posts.set_index("post_id")
    for post_id in random_rows["post_id"]:
        result = client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[int(post_id)],
            with_vectors=True,
        )
        if not result:
            logger.warning("No vector found for post_id=%s", post_id)
            continue
        neighbors = client.query_points(
            collection_name=settings.qdrant_collection,
            query=result[0].vector,
            limit=6,
            with_payload=True,
        ).points
        print(f"\nNearest neighbours for post {post_id}:")
        for point in neighbors:
            neighbor_id = int(point.payload["post_id"])
            if neighbor_id == int(post_id):
                continue
            print(f"  {neighbor_id} score={point.score:.3f}: {lookup.loc[neighbor_id, 'text'][:250]}")
