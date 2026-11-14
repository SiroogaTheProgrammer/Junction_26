from unittest.mock import Mock

from detector.config import Settings
from detector.embed import _embedding_parameters, _ensure_collection
from detector.io import write_json


def existing_collection():
    client = Mock()
    collection = Mock()
    collection.name = "posts"
    client.get_collections.return_value.collections = [collection]
    client.scroll.return_value = ([], None)
    return client


def test_embedding_cache_resumes_only_matching_dataset_and_model(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    write_json(tmp_path / "embed_report.json", {
        "fingerprint": "current", "embedding": _embedding_parameters(settings),
    })
    client = existing_collection()
    assert _ensure_collection(client, settings, "current", False) == set()
    client.delete_collection.assert_not_called()
    client.create_collection.assert_not_called()
    client.scroll.assert_called_once()


def test_embedding_cache_rebuilds_changed_or_unverified_parameters(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    for metadata in (
        {"fingerprint": "previous", "embedding": _embedding_parameters(settings)},
        {"fingerprint": "current"},
        {"fingerprint": "current", "embedding": {"model": "another-model"}},
    ):
        write_json(tmp_path / "embed_report.json", metadata)
        client = existing_collection()
        assert _ensure_collection(client, settings, "current", False) == set()
        client.delete_collection.assert_called_once_with("posts")
        client.create_collection.assert_called_once()
