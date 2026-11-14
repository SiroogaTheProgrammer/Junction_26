import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from detector import storage
from detector.config import Settings


@pytest.fixture
def caches(tmp_path, monkeypatch):
    datasets_cache = tmp_path / "hf" / "datasets"
    xet = tmp_path / "hf" / "xet"
    monkeypatch.setattr("datasets.config.HF_DATASETS_CACHE", str(datasets_cache))
    monkeypatch.setattr("huggingface_hub.constants.HF_XET_CACHE", str(xet))
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: (_ for _ in ()).throw(
        __import__("huggingface_hub.errors", fromlist=["CacheNotFound"]).CacheNotFound("none", tmp_path)))
    return datasets_cache, xet


def write(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_cleanup_keeps_only_live_cursor_and_unrelated_files(tmp_path, caches):
    data = tmp_path / "data"
    kept = [write(data / "live_progress.json"), write(data / "live_progress.lock"),
            write(data / "trump-2020" / "live_progress.json"), write(data / "live_source.json"),
            write(data / "notes.txt")]
    old = [write(data / "posts.parquet", 1000), write(data / "analyses.previous-2000.jsonl"),
           write(data / "clouds-10000" / "embeddings.npy", 5000), write(data / "clouds-10000" / "clusters.json")]
    datasets_cache, xet = caches
    lock = write(datasets_cache / "old_dataset.lock", 0)
    arrow = write(datasets_cache / "alpindale___two-million-bluesky-posts" / "data.arrow", 2000)
    fresh_log = write(xet / "logs" / "new.log")
    stale_log = write(xet / "logs" / "old.log")
    os.utime(stale_log, (time.time() - 2 * 86400,) * 2)

    preview = storage.cleanup(Settings(_env_file=None, data_dir=data), dry_run=True)
    assert all(path.exists() for path in old + [arrow, stale_log])
    removed = storage.cleanup(Settings(_env_file=None, data_dir=data))
    assert {path for path, _ in removed} == {path for path, _ in preview}
    assert all(path.exists() for path in kept + [lock, fresh_log])
    assert not any(path.exists() for path in old + [arrow.parent, stale_log])
    assert not (data / "clouds-10000").exists()
    assert sum(size for _, size in removed) == 1000 + 10 + 5000 + 10 + 2000 + 10


def test_cleanup_refuses_home_or_root_data_dir(caches):
    with pytest.raises(ValueError, match="dedicated data folder"):
        storage.cleanup(Settings(_env_file=None, data_dir=Path.home()), dry_run=True)


def test_hub_cache_keeps_live_dataset_and_embedding_model_only(monkeypatch, tmp_path):
    def repo(kind, name, *revisions):
        return SimpleNamespace(repo_type=kind, repo_id=name, repo_path=tmp_path / name, size_on_disk=100, revisions=[
            SimpleNamespace(commit_hash=commit, refs=refs, snapshot_path=tmp_path / name / commit, size_on_disk=40)
            for commit, refs in revisions
        ])

    deleted = []

    class Cache:
        warnings = []
        repos = [
            repo("model", "intfloat/multilingual-e5-small", ("new", {"main"}), ("old", set())),
            repo("dataset", "itsmebatuhan/bluesky-10m-posts-15-languages", ("pinned", set())),
            repo("dataset", "alpindale/two-million-bluesky-posts", ("a", {"main"}), ("b", set())),
        ]

        def delete_revisions(self, *hashes):
            deleted.extend(hashes)
            return SimpleNamespace(execute=lambda: None)

    monkeypatch.setattr("huggingface_hub.scan_cache_dir", Cache)
    removed = []
    storage._clean_hub(Settings(_env_file=None), True, removed)
    assert deleted == []
    storage._clean_hub(Settings(_env_file=None), False, [])
    assert deleted == ["old", "a", "b"]
    assert [path.name for path, _ in removed] == ["old", "two-million-bluesky-posts"]
