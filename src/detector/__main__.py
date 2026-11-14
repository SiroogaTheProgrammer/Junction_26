import argparse
import asyncio
import logging
from pathlib import Path

from detector.config import get_settings


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="python -m detector")
    subparsers = command_parser.add_subparsers(dest="command", required=True)
    for command in ("load", "embed", "cluster", "project", "stats", "analyze", "serve"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, help="Artifact folder (overrides DATA_DIR)")
        if command in {"embed", "cluster", "project"}:
            subparser.add_argument("--collection", help="Qdrant collection matching this dataset")
        if command == "load":
            subparser.add_argument("--dataset", help="Hugging Face dataset id (overrides HF_DATASET)")
            subparser.add_argument("--posts", type=int)
            subparser.add_argument("--scan-limit", type=int, help="Stream only this prefix for a bounded smoke test; not a global sample")
        elif command == "stats":
            subparser.add_argument("--inspect", action="store_true", help="Print top spread metrics with example posts")
        elif command == "embed":
            subparser.add_argument("--benchmark", type=int)
            subparser.add_argument("--recreate", action="store_true")
        elif command == "analyze":
            subparser.add_argument("--force", action="store_true")
            subparser.add_argument("--cluster", type=int)
            subparser.add_argument("--dry-run", action="store_true")
            subparser.add_argument("--all-clusters", action="store_true")
        elif command == "serve":
            subparser.add_argument("--host", default="127.0.0.1")
            subparser.add_argument("--port", type=int, default=8000)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--data-dir", type=Path, help="Artifact folder (overrides DATA_DIR)")
    all_parser.add_argument("--collection", help="Qdrant collection for this dataset")
    all_parser.add_argument("--posts", type=int, help="Override N_POSTS for this run")
    all_parser.add_argument("--dataset", help="Hugging Face dataset id (overrides HF_DATASET)")
    return command_parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args()
    settings = get_settings()
    overrides = {}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if getattr(args, "collection", None) is not None:
        overrides["qdrant_collection"] = args.collection
    if getattr(args, "dataset", None) is not None:
        overrides["hf_dataset"] = args.dataset
    if getattr(args, "posts", None) is not None:
        if args.posts <= 0:
            raise ValueError("--posts must be a positive number")
        overrides["n_posts"] = args.posts
    if overrides:
        settings = settings.model_copy(update=overrides)
    logging.info("Using artifact folder: %s", settings.data_path.resolve())

    if args.command == "load":
        from detector.load import run

        run(settings, scan_limit=args.scan_limit)
    elif args.command == "embed":
        from detector.embed import benchmark, run

        if args.benchmark:
            benchmark(settings, args.benchmark)
        else:
            run(settings, recreate=args.recreate)
    elif args.command == "cluster":
        from detector.cluster import run

        run(settings)
    elif args.command == "project":
        from detector.cluster import project

        project(settings)
    elif args.command == "stats":
        from detector.stats import run

        run(settings, inspect=args.inspect)
    elif args.command == "analyze":
        from detector.analyze import run

        asyncio.run(run(
            settings, force=args.force, cluster_id=args.cluster, dry_run=args.dry_run,
            all_clusters=args.all_clusters,
        ))
    elif args.command == "serve":
        import uvicorn
        from detector import api

        api.settings = settings
        uvicorn.run(api.app, host=args.host, port=args.port)
    elif args.command == "all":
        from detector.load import run as load
        from detector.embed import run as embed
        from detector.cluster import run as cluster
        from detector.analyze import run as analyze

        load(settings)
        embed(settings)
        cluster(settings)
        asyncio.run(analyze(settings, all_clusters=True))


if __name__ == "__main__":
    main()
