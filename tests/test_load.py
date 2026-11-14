import pandas as pd
import pytest
from datasets import Dataset

from detector.config import Settings
from detector.io import read_json, write_json
from detector.load import prepare_frame, run


def test_random_10000_sample_is_repeatable_and_invalidates_only_derived_artifacts(tmp_path, monkeypatch):
    source = pd.DataFrame({
        "text": [f"Public dataset post with enough text number {i}" for i in range(10050)],
        "uri": [f"at://test/{i}" for i in range(10050)],
        "created_at": pd.date_range("2025-01-01", periods=10050, freq="min", tz="UTC"),
        "author": ["test-author"] * 10050,
    })
    train = Dataset.from_pandas(source)
    monkeypatch.setattr("detector.load.load_dataset", lambda name: {"train": train})
    settings = Settings(_env_file=None, data_dir=tmp_path)
    derived = (
        "assignments.parquet", "projection.parquet", "projection_model.npz", "embeddings.npy",
        "clusters.json", "analyses.jsonl", "analyze_report.json", "stats_report.json",
    )
    for name in derived:
        (tmp_path / name).write_text("stale")
    write_json(tmp_path / "embed_report.json", {"fingerprint": "old-for-rebuild"})

    run(settings)
    first = pd.read_parquet(tmp_path / "posts.parquet")
    assert len(first) == 10000
    assert first["uri"].nunique() == 10000
    assert set(first["uri"]) != set(source["uri"].head(10000))
    assert all(not (tmp_path / name).exists() for name in derived)
    assert read_json(tmp_path / "embed_report.json")["fingerprint"] == "old-for-rebuild"
    assert read_json(tmp_path / "load_report.json")["seed"] == 42

    (tmp_path / "clusters.json").write_text("same-sample-cache")
    run(settings)
    assert first.equals(pd.read_parquet(tmp_path / "posts.parquet"))
    assert (tmp_path / "clusters.json").read_text() == "same-sample-cache"
    run(settings.model_copy(update={"seed": 43}))
    different = pd.read_parquet(tmp_path / "posts.parquet")
    assert set(first["uri"]) != set(different["uri"])
    assert not (tmp_path / "clusters.json").exists()


def multilingual_rows():
    return [
        {"did": "did:plc:a", "rkey": "same-key", "text": "Dies ist eine öffentliche Nachricht über lokale Nachrichten.", "lang": "de", "created_at": "2026-08-09T12:00:00Z"},
        {"did": "did:plc:b", "rkey": "same-key", "text": "Bu şehirde toplu taşıma hakkında uzun bir gönderi.", "lang": "tr", "created_at": "2026-08-10T12:00:00Z", "is_reply": True},
        {"did": "did:plc:a", "rkey": "same-key", "text": "Duplicate record with long enough text", "lang": "de", "created_at": "2026-08-09T12:00:00Z"},
        {"did": None, "rkey": "missing-did", "text": "Missing author must not become a fake string", "created_at": "2026-08-09T12:00:00Z"},
        {"did": "did:plc:a", "rkey": "", "text": "Empty record key must not invent a valid URI", "created_at": "2026-08-09T12:00:00Z"},
        {"did": "did:plc:a", "rkey": "invalid-time", "text": "Invalid time must be excluded from sample", "created_at": "not a time"},
    ]


def test_new_dataset_schema_and_unknown_image_metadata_are_preserved(tmp_path, monkeypatch):
    rows = multilingual_rows()
    train = Dataset.from_pandas(pd.DataFrame(rows))
    monkeypatch.setattr("detector.load.load_dataset", lambda name: {"train": train})
    settings = Settings(_env_file=None, data_dir=tmp_path)
    run(settings)
    frame = pd.read_parquet(tmp_path / "posts.parquet")
    assert len(frame) == 2
    assert frame["uri"].tolist() == [
        "at://did:plc:a/app.bsky.feed.post/same-key", "at://did:plc:b/app.bsky.feed.post/same-key",
    ]
    assert set(frame["author"]) == {"did:plc:a", "did:plc:b"}
    assert frame["has_images"].isna().all()
    assert frame["is_reply"].isna().iloc[0] and frame["is_reply"].iloc[1]
    report = read_json(tmp_path / "load_report.json")
    assert report["language_counts"] == {"de": 1, "tr": 1}
    assert report["image_metadata_available"] is False
    assert report["counts"]["source_rows"] == 6
    assert report["counts"]["unique_uri"] == 2
    fingerprint = report["fingerprint"]
    rows[0]["text"] = "Updated content with exactly the same post URI."
    train = Dataset.from_pandas(pd.DataFrame(rows))
    run(settings)
    assert read_json(tmp_path / "load_report.json")["fingerprint"] != fingerprint


def test_bounded_streaming_is_explicit_and_never_consumes_the_full_source(tmp_path, monkeypatch):
    seen = []

    def source():
        for index in range(1000):
            seen.append(index)
            yield {"did": "did:a", "rkey": str(index), "text": f"Long enough content in source post {index}", "created_at": "2026-08-09T12:00:00Z"}

    def load(name, **kwargs):
        assert kwargs == {"streaming": True}
        return {"train": source()}

    monkeypatch.setattr("detector.load.load_dataset", load)
    run(Settings(_env_file=None, data_dir=tmp_path, n_posts=5), scan_limit=10)
    assert len(seen) == 10
    assert len(pd.read_parquet(tmp_path / "posts.parquet")) == 5
    report = read_json(tmp_path / "load_report.json")
    assert report["sampling"] == "uniform_within_scanned_prefix_only_not_global"
    assert report["scan_limit"] == 10


@pytest.mark.parametrize("rows,match", [
    ([{"text": "missing schema"}], "missing required columns"),
    ([{"text": "Enough text", "author": "a", "uri": "u", "created_at": "2026-08-09", "has_images": "False"}], "boolean values"),
])
def test_dataset_schema_errors_are_explicit(rows, match):
    with pytest.raises(ValueError, match=match):
        prepare_frame(pd.DataFrame(rows), 0)


def test_trump_2020_rows_map_to_stream_positions_without_inventing_metadata():
    rows = [
        {"Unnamed: 0": 0, "source_id": 361267889, "target_id": 25073877.0, "date": "2020-05-10 07:00:00", "texto": "Pray to be stronger men."},
        {"Unnamed: 0": 1, "source_id": None, "target_id": None, "date": "2020-05-10 06:59:45", "texto": "No author on this one"},
        {"Unnamed: 0": 0, "source_id": 509, "target_id": None, "date": "2020-05-10 06:58:00", "texto": "Same index, other file"},
    ]
    frame, counts = prepare_frame(pd.DataFrame(rows), 0, row_offset=1_000)
    assert frame["uri"].tolist() == ["row:1000", "row:1002"]
    assert frame["author"].tolist() == ["twitter-user:361267889", "twitter-user:509"]
    assert str(frame["created_at"].iloc[0]) == "2020-05-10 07:00:00+00:00"
    assert frame["lang"].isna().all() and frame["is_reply"].isna().all() and frame["has_images"].isna().all()
    assert counts["valid_identifiers"] == 2
    with pytest.raises(ValueError, match="stream row offset"):
        prepare_frame(pd.DataFrame(rows), 0)
    large = [{"source_id": 1165041307326328833, "target_id": None, "date": "2020-05-10 07:00:01", "texto": "Large author id"}]
    assert prepare_frame(pd.DataFrame(large), 0, row_offset=0)[0]["author"].iloc[0] == "twitter-user:1165041307326328833"
