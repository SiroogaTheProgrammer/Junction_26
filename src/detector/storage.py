"""Deletes stored data the running Live view does not need, so a small VPS disk does not fill up."""
import logging
import shutil
import time
from pathlib import Path

from detector.config import Settings

logger = logging.getLogger(__name__)
# The Live cursor (and its in-flight atomic write) is the only state Live keeps on disk.
LIVE_FILES = {"live_progress.json", "live_progress.lock", "live_progress.json.tmp"}
# Only files the app itself writes are removed, so a misconfigured DATA_DIR cannot lose unrelated files.
APP_SUFFIXES = {".parquet", ".json", ".jsonl", ".npy", ".npz", ".tmp"}
XET_LOG_MAX_AGE_S = 24 * 3600


def _size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(item.lstat().st_size for item in path.rglob("*") if item.is_file() or item.is_symlink())


def _remove(path: Path, dry_run: bool, removed: list[tuple[Path, int]]) -> None:
    size = _size(path)
    if not dry_run:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    removed.append((path, size))


def _clean_data_dir(root: Path, dry_run: bool, removed: list[tuple[Path, int]]) -> None:
    root = root.resolve()
    if not root.is_dir():
        return
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError(f"Refusing to clean DATA_DIR {root}: point it at a dedicated data folder")
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        if path.is_file() and path.name not in LIVE_FILES and path.suffix in APP_SUFFIXES:
            _remove(path, dry_run, removed)
        elif path.is_dir() and not dry_run and not any(path.iterdir()):
            path.rmdir()


def _clean_hub(settings: Settings, dry_run: bool, removed: list[tuple[Path, int]]) -> None:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return
    for warning in cache.warnings:
        logger.warning("Skipped an unreadable Hugging Face cache entry: %s", warning)
    keep = {("model", settings.embed_model), ("dataset", settings.live_dataset)}
    hashes = []
    for repo in cache.repos:
        if (repo.repo_type, repo.repo_id) not in keep:
            hashes += [revision.commit_hash for revision in repo.revisions]
            removed.append((repo.repo_path, repo.size_on_disk))
            continue
        # Unreferenced revisions beside a referenced one are leftovers; a commit-pinned download has no refs.
        if not any(revision.refs for revision in repo.revisions):
            continue
        for revision in repo.revisions:
            if not revision.refs:
                hashes.append(revision.commit_hash)
                removed.append((revision.snapshot_path, revision.size_on_disk))
    if hashes and not dry_run:
        cache.delete_revisions(*hashes).execute()


def _clean_datasets_cache(dry_run: bool, removed: list[tuple[Path, int]]) -> None:
    """Live streams its dataset, so prepared Arrow copies and downloads from older runs are never read."""
    from datasets import config

    root = Path(config.HF_DATASETS_CACHE)
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.suffix != ".lock":
            _remove(path, dry_run, removed)


def _clean_xet_logs(dry_run: bool, removed: list[tuple[Path, int]]) -> None:
    from huggingface_hub import constants

    logs = Path(constants.HF_XET_CACHE) / "logs"
    if not logs.is_dir():
        return
    cutoff = time.time() - XET_LOG_MAX_AGE_S
    for path in logs.iterdir():
        if path.is_file() and path.stat().st_mtime < cutoff:
            _remove(path, dry_run, removed)


def cleanup(settings: Settings, dry_run: bool = False) -> list[tuple[Path, int]]:
    """Remove old offline artifacts and caches; returns (path, bytes) for everything removed."""
    removed: list[tuple[Path, int]] = []
    steps = (
        ("data folder", lambda: _clean_data_dir(settings.data_path, dry_run, removed)),
        ("Hugging Face model and dataset cache", lambda: _clean_hub(settings, dry_run, removed)),
        ("prepared dataset cache", lambda: _clean_datasets_cache(dry_run, removed)),
        ("Xet logs", lambda: _clean_xet_logs(dry_run, removed)),
    )
    for name, step in steps:
        try:
            step()
        except ValueError:
            raise
        except OSError as exc:
            logger.warning("Storage cleanup of the %s failed: %s", name, exc)
    action = "Would remove" if dry_run else "Removed"
    for path, size in removed:
        logger.info("%s %s (%.1f MB)", action, path, size / 1e6)
    logger.info("%s %d items, %.1f MB in total", action, len(removed), sum(size for _, size in removed) / 1e6)
    return removed
