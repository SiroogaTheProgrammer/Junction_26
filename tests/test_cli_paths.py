import sys
from unittest.mock import Mock

from detector import api
from detector.__main__ import main, parser
from detector.config import Settings


def test_each_stage_accepts_explicit_data_folder():
    for command in ("load", "embed", "cluster", "stats", "analyze", "serve", "all"):
        args = parser().parse_args([command, "--data-dir", "data/clouds-10000"])
        assert str(args.data_dir) == "data/clouds-10000"
    for command in ("embed", "cluster", "all"):
        args = parser().parse_args([command, "--collection", "clouds_10000"])
        assert args.collection == "clouds_10000"


def test_dataset_and_statistics_cli_options():
    for command in ("load", "all"):
        args = parser().parse_args([command, "--dataset", "itsmebatuhan/bluesky-10m-posts-15-languages", "--posts", "10000"])
        assert args.dataset == "itsmebatuhan/bluesky-10m-posts-15-languages"
        assert args.posts == 10000
    assert parser().parse_args(["load", "--scan-limit", "300"]).scan_limit == 300
    assert parser().parse_args(["stats", "--inspect"]).inspect


def test_serve_data_folder_is_passed_to_the_actual_api(tmp_path, monkeypatch):
    import uvicorn

    original = api.settings
    monkeypatch.setattr(api, "settings", original)
    monkeypatch.setattr("detector.__main__.get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(sys, "argv", ["detector", "serve", "--data-dir", str(tmp_path), "--port", "8769"])
    serve = Mock()
    monkeypatch.setattr(uvicorn, "run", serve)
    main()
    assert api.settings.data_path == tmp_path
    serve.assert_called_once_with(api.app, host="127.0.0.1", port=8769)
