import hashlib
import logging
import re
import time
from collections import Counter
from itertools import islice
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Features, Value, load_dataset

from detector.config import Settings
from detector.io import ensure_data_dir, read_json, write_json

logger = logging.getLogger(__name__)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
POST_COLUMNS = ["uri", "author", "created_at", "text", "has_images", "lang", "is_reply"]
FEATURES = Features({
    "uri": Value("string"), "author": Value("string"), "created_at": Value("timestamp[us, tz=UTC]"),
    "text": Value("string"), "has_images": Value("bool"), "lang": Value("string"), "is_reply": Value("bool"),
})


def prepare_frame(frame: pd.DataFrame, min_chars: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map both dataset schemas; missing optional metadata stays unknown."""
    frame = frame.copy()
    required = ("text", "created_at")
    missing = [name for name in required if name not in frame]
    if "author" not in frame and "did" not in frame:
        missing.append("author or did")
    if "uri" not in frame and not {"did", "rkey"}.issubset(frame.columns):
        missing.append("uri or did+rkey")
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if "author" not in frame:
        frame["author"] = frame["did"]
    if "uri" not in frame:
        did = frame["did"].astype("string").str.strip()
        key = frame["rkey"].astype("string").str.strip()
        frame["uri"] = ("at://" + did + "/app.bsky.feed.post/" + key).where(did.ne("") & key.ne(""))
    counts = {"source_rows": len(frame)}
    frame = frame.loc[frame["text"].notna()].copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.loc[frame["text"].ne("")].copy()
    counts["nonempty_text"] = len(frame)
    frame = frame.loc[frame["text"].map(lambda text: len(URL_PATTERN.sub("", text).strip())) >= min_chars].copy()
    counts["minimum_text_length"] = len(frame)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce", format="mixed")
    frame = frame.loc[frame["created_at"].notna()].copy()
    counts["valid_timestamps"] = len(frame)
    for column in ("uri", "author"):
        frame[column] = frame[column].astype("string").str.strip()
    frame = frame.loc[frame["uri"].notna() & frame["uri"].ne("") & frame["author"].notna() & frame["author"].ne("")].copy()
    counts["valid_identifiers"] = len(frame)
    for column in ("has_images", "is_reply"):
        if column not in frame:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        else:
            if not frame[column].dropna().isin([True, False, 0, 1]).all():
                raise ValueError(f"Dataset {column} must contain boolean values or null")
            frame[column] = frame[column].astype("boolean")
    if "lang" not in frame:
        frame["lang"] = None
    return frame[POST_COLUMNS], counts


def _arrow_sample(train: Any, settings: Settings) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: Counter[str] = Counter()

    def batch_mapper(batch: dict[str, list]) -> dict[str, list]:
        frame, batch_counts = prepare_frame(pd.DataFrame(dict(batch)), settings.min_post_chars)
        counts.update(batch_counts)
        return frame.to_dict(orient="list")

    # Arrow keeps the text column on disk; only the final sample becomes a pandas frame.
    prepared = train.map(
        batch_mapper, batched=True, batch_size=10_000, remove_columns=train.column_names,
        features=FEATURES, load_from_cache_file=False, desc="Normalising eligible posts",
    )
    if not len(prepared):
        raise ValueError("No eligible posts remain after dataset cleaning")
    uri_index = pa.table({
        "uri": prepared.data.column("uri"),
        "row": pa.array(np.arange(len(prepared), dtype=np.int64)),
    })
    first = uri_index.group_by("uri").aggregate([("row", "min")])["row_min"].to_numpy()
    prepared = prepared.select(np.sort(first))
    counts["unique_uri"] = len(prepared)
    if 0 < settings.n_posts < len(prepared):
        prepared = prepared.shuffle(seed=settings.seed).select(range(settings.n_posts))
    return prepared.to_pandas(), dict(counts)


def run(settings: Settings, scan_limit: int | None = None) -> None:
    if scan_limit is not None and scan_limit < 1:
        raise ValueError("--scan-limit must be a positive number")
    started = time.monotonic()
    ensure_data_dir(settings.data_path)
    logger.info("Loading dataset %s", settings.hf_dataset)
    if scan_limit is None:
        dataset = load_dataset(settings.hf_dataset)
        if "train" not in dataset:
            raise ValueError(f"Dataset {settings.hf_dataset} does not contain a train split")
        frame, counts = _arrow_sample(dataset["train"], settings)
        sampling = "uniform_without_replacement_after_cleaning_and_uri_deduplication"
    else:
        logger.warning("Bounded streaming smoke test: sampling only the first %d source rows, not the whole dataset", scan_limit)
        dataset = load_dataset(settings.hf_dataset, streaming=True)
        if "train" not in dataset:
            raise ValueError(f"Dataset {settings.hf_dataset} does not contain a train split")
        frame, counts = prepare_frame(pd.DataFrame(islice(dataset["train"], scan_limit)), settings.min_post_chars)
        frame = frame.drop_duplicates("uri", keep="first")
        counts["unique_uri"] = len(frame)
        if 0 < settings.n_posts < len(frame):
            frame = frame.sample(n=settings.n_posts, random_state=settings.seed)
        sampling = "uniform_within_scanned_prefix_only_not_global"
    if frame.empty:
        raise ValueError("No eligible posts remain after dataset cleaning")
    if settings.n_posts > len(frame):
        logger.warning("Requested %d posts but only %d eligible unique posts are available", settings.n_posts, len(frame))
    frame = frame.sort_values(["created_at", "uri"], kind="stable").reset_index(drop=True)
    frame.insert(0, "post_id", range(len(frame)))
    counts["selected_rows"] = len(frame)
    payload = f"{settings.hf_dataset}:{settings.seed}:{settings.n_posts}:{sampling}\n"
    fingerprint = hashlib.sha256(
        (payload + frame.to_json(orient="records", date_format="iso")).encode("utf-8")
    ).hexdigest()

    previous = read_json(settings.data_path / "load_report.json", {})
    temporary = settings.data_path / "posts.parquet.tmp"
    frame.to_parquet(temporary, index=False)
    temporary.replace(settings.data_path / "posts.parquet")
    if previous.get("fingerprint") != fingerprint:
        for name in (
            "assignments.parquet", "projection.parquet", "projection_model.npz", "embeddings.npy",
            "clusters.json", "analyses.jsonl", "analyze_report.json", "stats_report.json",
        ):
            (settings.data_path / name).unlink(missing_ok=True)
        logger.info("Sample or metadata changed; cleared derived cloud, placement, statistics and analysis artifacts")
    write_json(settings.data_path / "load_report.json", {
        "counts": counts, "n_posts": len(frame), "n_posts_requested": settings.n_posts,
        "seed": settings.seed, "dataset": settings.hf_dataset, "fingerprint": fingerprint,
        "sampling": sampling, "scan_limit": scan_limit,
        "image_metadata_available": bool(frame["has_images"].notna().all()),
        "language_counts": frame["lang"].fillna("unknown").value_counts().to_dict(),
        "uri_hash": hashlib.sha256("\n".join(sorted(frame["uri"].astype(str))).encode("utf-8")).hexdigest(),
        "elapsed_s": round(time.monotonic() - started, 3),
    })
    logger.info("Wrote %d posts to %s in %.1fs", len(frame), settings.data_path / "posts.parquet", time.monotonic() - started)
