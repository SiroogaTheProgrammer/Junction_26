import asyncio
import json
import logging
import shlex
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from detector.chat import ChatInputError, ChatUpstreamError, Placer, run_chat
from detector.config import get_settings
from detector.embedding import VECTOR_SIZE
from detector.io import read_json, read_jsonl, require_artifact
from detector.llm import create_client, create_super_client, super_configured
from detector.projection import ProjectionModel
from detector.prompts import PROMPT_VERSION
from detector.rules import behavior_anomaly, display_tier, review_priority
from detector.stats import baseline_for, describe_signals, is_organic_looking

logger = logging.getLogger(__name__)
settings = get_settings()
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def _serialize_timestamp(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _load_dashboard_data() -> dict[str, Any]:
    data_dir = settings.data_path
    logger.info("Loading dashboard artifacts from %s", data_dir.resolve())
    clusters_doc = read_json(require_artifact(data_dir / "clusters.json"))
    posts = pd.read_parquet(require_artifact(data_dir / "posts.parquet"))
    assignments = pd.read_parquet(require_artifact(data_dir / "assignments.parquet"))
    analysis_records = read_jsonl(data_dir / "analyses.jsonl")
    analyses = {int(record["cluster_id"]): record for record in analysis_records}
    projection_path = data_dir / "projection.parquet"
    projection = None
    projection_error = None
    if projection_path.is_file():
        projection = pd.read_parquet(projection_path)
        columns = ["post_id", "cluster_id", "x", "y", "z"]
        if not set(columns).issubset(projection.columns):
            raise ValueError("projection.parquet is missing required columns; rerun cluster")
        if (
            len(projection) != len(posts) or projection["post_id"].duplicated().any()
            or not np.isfinite(projection[["x", "y", "z"]].to_numpy()).all()
        ):
            raise ValueError("projection.parquet must contain one finite point per post; rerun cluster")
        aligned = projection.merge(assignments, on="post_id", validate="one_to_one", suffixes=("", "_expected"))
        if (
            len(aligned) != len(posts) or set(projection["post_id"]) != set(posts["post_id"])
            or not aligned["cluster_id"].eq(aligned["cluster_id_expected"]).all()
        ):
            raise ValueError("Projection does not match the current posts and assignments; rerun cluster")
    else:
        projection_error = (
            f"Embedding projection is missing in {data_dir.resolve()}. "
            "Run python -m detector cluster "
            f"--data-dir {shlex.quote(str(data_dir))} "
            f"--collection {shlex.quote(settings.qdrant_collection)} with Qdrant running, "
            "then restart the website. If another dataset already has a projection, "
            "serve that folder with --data-dir instead; do not rebuild it."
        )
        logger.warning(projection_error)
    post_lookup = posts.set_index("post_id").to_dict(orient="index")
    map_rows: list[list[Any]] = []
    live_rows: list[list[Any]] = []
    live_clock: list[str | None] = []
    if projection is not None:
        map_rows = [
            [int(row.post_id), int(row.cluster_id), float(row.x), float(row.y), float(row.z)]
            for row in projection.itertuples(index=False)
        ]
        arrival = projection.assign(
            created_at=projection["post_id"].map(posts.set_index("post_id")["created_at"]),
            row=range(len(projection)),
        ).sort_values(["created_at", "post_id"], kind="stable")
        live_rows = [map_rows[index] for index in arrival["row"]]
        live_clock = [_serialize_timestamp(value) for value in arrival["created_at"]]
    cluster_post_ids = {
        int(cluster_id): group["post_id"].astype(int).tolist()
        for cluster_id, group in assignments.groupby("cluster_id")
        if int(cluster_id) != -1
    }
    return {
        "clusters_doc": clusters_doc,
        "posts": posts,
        "assignments": assignments,
        "analyses": analyses,
        "post_lookup": post_lookup,
        "cluster_post_ids": cluster_post_ids,
        "projection": projection,
        "projection_error": projection_error,
        "map_rows": map_rows,
        "live_rows": live_rows,
        "live_clock": live_clock,
        "chat": _load_chat(data_dir, posts, projection, projection_error),
    }


def _load_chat(
    data_dir: Path, posts: pd.DataFrame, projection: pd.DataFrame | None, projection_error: str | None,
) -> dict[str, Any]:
    """Chat needs the stored embeddings and the fitted projection to place new text."""
    embeddings_path, model_path = data_dir / "embeddings.npy", data_dir / "projection_model.npz"
    rebuild = (
        f"Run python -m detector project --data-dir {shlex.quote(str(data_dir))} "
        f"--collection {shlex.quote(settings.qdrant_collection)} with Qdrant running, then restart the website."
    )
    if projection is None:
        return {"placer": None, "error": projection_error}
    if not embeddings_path.is_file() or not model_path.is_file():
        error = f"Placement artifacts are missing in {data_dir.resolve()}. {rebuild}"
        logger.warning(error)
        return {"placer": None, "error": error}
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    model = ProjectionModel.load(model_path)
    if embeddings.shape != (len(posts), VECTOR_SIZE):
        raise ValueError(f"embeddings.npy does not match posts.parquet. {rebuild}")
    check = min(len(posts), 50)
    stored = projection[["x", "y", "z"]].to_numpy()[:check]
    if not np.allclose(model.transform(embeddings[:check]), stored, atol=1e-4):
        raise ValueError(f"Embeddings and projection model do not reproduce projection.parquet. {rebuild}")
    if not super_configured(settings):
        error = "Chat is off: set SUPER_LLM_API_KEY and SUPER_LLM_BASE_URL in .env, then restart."
        logger.warning(error)
        return {"placer": None, "error": error}
    placer = Placer(
        settings, embeddings, model,
        projection["post_id"].to_numpy(dtype=np.int64), projection["cluster_id"].to_numpy(dtype=np.int64),
    )
    return {"placer": placer, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from detector.live import LiveProcessor

    app.state.chat_slots = asyncio.Semaphore(settings.chat_concurrency)
    app.state.chat_times = deque()
    app.state.live = LiveProcessor(settings)
    app.state.dashboard = None
    try:
        yield
    finally:
        await app.state.live.close()


app = FastAPI(
    title="Coordinated-behavior detector",
    description="Descriptive cluster analysis of a sampled public Bluesky dataset.",
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


class ChatAttachment(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    type: str = Field(default="", max_length=100)
    data: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=8000)
    attachments: list[ChatAttachment] = Field(default_factory=list)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


def _chat_body_limit() -> int:
    # Base64 inflates files by 4/3; the remainder covers the message, history and JSON framing.
    files = settings.chat_max_attachments * settings.chat_max_attachment_mb * 1024 * 1024
    return int(files * 4 / 3) + 256_000


@app.middleware("http")
async def limit_chat_body(request: Request, call_next):
    if request.method == "POST" and request.url.path in {"/api/chat", "/api/live/chat"}:
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
        if int(length) > _chat_body_limit():
            return JSONResponse({"detail": "The request and its attachments are too large"}, status_code=413)
    response = await call_next(request)
    if request.url.path in {"/", "/explore", "/live"} or request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _data(request: Request) -> dict[str, Any]:
    value = getattr(request.app.state, "dashboard", None)
    if value is None:
        try:
            value = _load_dashboard_data()
        except FileNotFoundError as exc:
            logger.warning("Dashboard artifacts are not ready: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        request.app.state.dashboard = value
    return value


def _thresholds() -> dict[str, float]:
    return {"hide_below": settings.relevance_hide_below, "full_from": settings.relevance_full_from}


def _cluster_status(
    cluster: dict[str, Any], analysis: dict[str, Any] | None, baseline: dict[str, Any],
) -> dict[str, Any]:
    status = analysis.get("status", "not_analysed") if analysis else "not_analysed"
    parsed = analysis.get("parsed") if analysis and status == "ok" else None
    anomaly = behavior_anomaly(cluster, baseline, settings.anomaly_percentile)
    relevance = parsed.get("relevance") if parsed else None
    priority = review_priority(parsed, cluster, baseline, settings.anomaly_percentile)
    organic = is_organic_looking(cluster, baseline)
    stale = bool(analysis and (
        (cluster.get("stats_hash") and analysis.get("stats_hash") != cluster["stats_hash"])
        or analysis.get("prompt_version") != PROMPT_VERSION
        or len(analysis.get("sample") or []) != min(settings.sample_size, int(cluster["size"]))
    ))
    tier = (
        display_tier(relevance, anomaly.flag, _thresholds()) if relevance is not None
        else "faded" if parsed else status
    )
    return {
        "cluster_id": int(cluster["cluster_id"]),
        "size": int(cluster["size"]),
        **{key: value for key, value in cluster.items() if key not in {"cluster_id", "size", "representative_post_ids"}},
        "status": status,
        "label": parsed.get("label") if parsed else None,
        "concern_level": parsed.get("concern_level") if parsed else None,
        "confidence": parsed.get("confidence") if parsed else None,
        "relevance": relevance,
        "relevance_kind": parsed.get("relevance_kind") if parsed else None,
        "relevance_reason": (
            parsed.get("relevance_reason", "Legacy analysis: rerun to score narrative relevance.")
            if parsed else None
        ),
        "behavior_anomaly": anomaly.flag,
        "anomaly_triggered_by": anomaly.triggered_by,
        "flags": {"organic_looking": organic, "behavior_anomaly": anomaly.model_dump()},
        "organic_looking": organic,
        "signals": describe_signals(cluster, baseline),
        "baseline_bucket": baseline_for(int(cluster["size"]), baseline)["name"],
        "bucket_baseline": baseline_for(int(cluster["size"]), baseline),
        "burstiness_reliable": baseline.get("burstiness_reliable", True),
        "burst_window_min": baseline.get("burst_window_min", 10),
        "display_tier": tier,
        "review_priority": priority["score"],
        "review_score": priority,
        "analysis_stale": stale,
        # Only a completed assessment can be dismissed; failed or pending clouds stay visible.
        "discarded": status == "ok" and not stale and priority["score"] < settings.min_review_priority,
    }


def _cluster_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = data["clusters_doc"]["baseline"]
    return [
        _cluster_status(cluster, data["analyses"].get(int(cluster["cluster_id"])), baseline)
        for cluster in data["clusters_doc"].get("clusters", [])
    ]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
def meta(request: Request) -> dict[str, Any]:
    data = _data(request)
    clusters = data["clusters_doc"].get("clusters", [])
    analyses = data["analyses"]
    statuses = _cluster_rows(data)
    run = data["clusters_doc"].get("run", {})
    models = run.get("parameters", {})
    llm_model = next((record.get("model") for record in analyses.values() if record.get("model")), None)
    return {
        "run": run,
        "baseline": data["clusters_doc"].get("baseline", {}),
        "dataset": data["clusters_doc"].get("dataset"),
        "models": {
            "embedding": models.get("embedding_model", settings.embed_model),
            "llm": llm_model or settings.llm_model,
        },
        "n_posts": int(run.get("n_posts", len(data["posts"]))),
        "n_clusters": len(clusters),
        "n_analysed": sum(item["status"] == "ok" for item in statuses),
        "n_stale": sum(item["analysis_stale"] for item in statuses),
        "n_failed": sum(item["status"] == "failed" for item in statuses),
        "n_hidden": sum(item["display_tier"] == "hidden" for item in statuses),
        "relevance_thresholds": _thresholds(),
        "anomaly_percentile": settings.anomaly_percentile,
        "noise_ratio": run.get("noise_ratio", 0.0),
        "map_available": data["projection"] is not None,
        "map_error": data["projection_error"],
        "projection": run.get("projection"),
        "n_pending": sum(item["status"] == "not_analysed" for item in statuses),
        "n_discarded": sum(item["discarded"] for item in statuses),
        "min_review_priority": settings.min_review_priority,
        "chat": {
            "available": data["chat"]["error"] is None,
            "error": data["chat"]["error"],
            "model": settings.super_llm_model,
            "judge": settings.super_llm_model if settings.chat_judge == "super" else settings.llm_model,
            "limits": {
                "attachments": settings.chat_max_attachments,
                "attachment_mb": settings.chat_max_attachment_mb,
                "message_chars": 8000,
            },
        },
        "live": {"posts_per_second": settings.live_posts_per_second},
    }


@app.get("/api/clusters")
def list_clusters(request: Request, include_hidden: bool = False) -> list[dict[str, Any]]:
    data = _data(request)
    rows = _cluster_rows(data)
    return rows if include_hidden else [row for row in rows if row["display_tier"] != "hidden"]


@app.get("/api/map")
def map_points(request: Request) -> dict[str, Any]:
    data = _data(request)
    points = data["projection"]
    if points is None:
        raise HTTPException(status_code=503, detail=data["projection_error"])
    return {
        "columns": ["post_id", "cluster_id", "x", "y", "z"],
        "points": data["map_rows"],
        "projection": data["clusters_doc"]["run"].get("projection"),
    }


LIVE_TICK_S = 0.25


@app.get("/api/live/processed")
async def live_processed(request: Request) -> StreamingResponse:
    processor = request.app.state.live
    processor.start()

    async def events():
        revision = -1
        while not await request.is_disconnected():
            if processor.revision != revision:
                snapshot = processor.snapshot()
                revision = snapshot["revision"]
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                if snapshot["status"] in {"complete", "error"}:
                    break
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@app.get("/api/live/clusters/{cluster_id}")
def live_cluster(cluster_id: int, request: Request) -> dict[str, Any]:
    detail = request.app.state.live.detail(cluster_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Live group is no longer in the rolling window")
    return detail


@app.get("/api/live/clusters/{cluster_id}/posts")
def live_cluster_posts(cluster_id: int, request: Request) -> dict[str, Any]:
    detail = request.app.state.live.detail(cluster_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Live group is no longer in the rolling window")
    return {"posts": detail["analysis"].get("sample", [])[:5]}


@app.get("/api/live/stream")
async def live_stream(request: Request) -> StreamingResponse:
    """Replays the stored sample in arrival order, as server-sent events."""
    data = _data(request)
    if data["projection"] is None:
        raise HTTPException(status_code=503, detail=data["projection_error"])
    rows, clock = data["live_rows"], data["live_clock"]
    rate = settings.live_posts_per_second
    batch = max(1, round(rate * LIVE_TICK_S))

    async def events():
        yield f"event: start\ndata: {json.dumps({'total': len(rows), 'posts_per_second': rate})}\n\n"
        for start in range(0, len(rows), batch):
            if await request.is_disconnected():
                return
            end = min(start + batch, len(rows))
            payload = {"points": rows[start:end], "sent": end, "total": len(rows), "clock": clock[end - 1]}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(LIVE_TICK_S)
        yield f"event: done\ndata: {json.dumps({'total': len(rows)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    data = _data(request)
    state = data["chat"]
    if state["error"]:
        raise HTTPException(status_code=503, detail=state["error"])
    clouds = {}
    for row in _cluster_rows(data):
        parsed = (data["analyses"].get(row["cluster_id"]) or {}).get("parsed") or {}
        clouds[row["cluster_id"]] = {**row, "summary": parsed.get("summary")}
    lookup = data["post_lookup"]
    return await _dispatch_chat(payload, request, lambda: (
        state["placer"], clouds, lambda post_id: str(lookup[post_id].get("text") or ""),
    ))


@app.get("/api/live/chat/status")
def live_chat_status(request: Request) -> dict[str, Any]:
    configured = super_configured(settings)
    return {
        "chat": {
            "available": configured,
            "error": None if configured else "Set SUPER_LLM_API_KEY and SUPER_LLM_BASE_URL to enable chat.",
            "limits": {"attachments": settings.chat_max_attachments,
                       "attachment_mb": settings.chat_max_attachment_mb, "message_chars": 8000},
        },
        "window_posts": sum(len(batch[1]) for batch in request.app.state.live.index),
    }


@app.post("/api/live/chat")
async def live_chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    if not super_configured(settings):
        raise HTTPException(status_code=503, detail="Set SUPER_LLM_API_KEY and SUPER_LLM_BASE_URL to enable chat.")
    def context():
        try:
            placer, clouds, texts = request.app.state.live.chat_context()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return placer, clouds, texts.__getitem__
    return await _dispatch_chat(payload, request, context)


async def _dispatch_chat(
    payload: ChatRequest, request: Request,
    context: Callable[[], tuple[Placer, dict, Callable[[int], str]]],
) -> dict[str, Any]:
    now = time.monotonic()
    times: deque = request.app.state.chat_times
    while times and now - times[0] > 60:
        times.popleft()
    slots: asyncio.Semaphore = request.app.state.chat_slots
    if len(times) >= settings.chat_max_per_minute or slots.locked():
        raise HTTPException(
            status_code=429, detail="The demo is busy. Wait a moment and send again.",
            headers={"Retry-After": "15"},
        )
    judge_client, judge_missing = None, None
    async with slots, create_super_client(settings) as super_client:
        placer, clouds, post_text = context()
        times.append(now)
        if settings.chat_judge == "super":
            judge_client, judge_model = super_client, settings.super_llm_model
        else:
            judge_model = settings.llm_model
            key = settings.llm_api_key
            if key and key.get_secret_value().strip() and "<deployment>" not in settings.llm_base_url:
                judge_client = create_client(settings)
            else:
                judge_missing = "LLM_API_KEY and LLM_BASE_URL for the Lightning judge are not configured"
        try:
            return await run_chat(
                settings=settings, placer=placer, message=payload.message,
                history=[turn.model_dump() for turn in payload.history],
                attachments=[item.model_dump() for item in payload.attachments],
                clouds=clouds, post_text=post_text,
                super_client=super_client, judge_client=judge_client, judge_model=judge_model,
                judge_missing=judge_missing,
            )
        except ChatInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ChatUpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            if judge_client is not None and judge_client is not super_client:
                await judge_client.close()


@app.get("/api/clusters/{cluster_id}")
def get_cluster(cluster_id: int, request: Request) -> dict[str, Any]:
    data = _data(request)
    cluster = next(
        (item for item in data["clusters_doc"].get("clusters", []) if int(item["cluster_id"]) == cluster_id),
        None,
    )
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} was not found")
    analysis = data["analyses"].get(cluster_id)
    sample: list[dict[str, Any]] = []
    if analysis:
        for item in analysis.get("sample", []):
            post = data["post_lookup"].get(int(item["post_id"]))
            if post is not None:
                sample.append(
                    {
                        "pid": item["pid"],
                        "post_id": int(item["post_id"]),
                        "author": post.get("author"),
                        "created_at": _serialize_timestamp(post.get("created_at")),
                        "text": post.get("text"),
                    }
                )
    return {
        "cluster": {
            **cluster,
            **_cluster_status(cluster, analysis, data["clusters_doc"]["baseline"]),
        },
        "analysis": {**(analysis or {"status": "not_analysed"}), "sample": sample},
    }


@app.get("/api/clusters/{cluster_id}/posts")
def cluster_posts(
    cluster_id: int,
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    data = _data(request)
    cluster = next(
        (item for item in data["clusters_doc"].get("clusters", []) if int(item["cluster_id"]) == cluster_id),
        None,
    )
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} was not found")
    ids = data["cluster_post_ids"].get(cluster_id, [])
    selected_ids = ids[offset : offset + limit]
    lookup = data["post_lookup"]
    items = [
        {
            "post_id": post_id,
            "author": lookup[post_id].get("author"),
            "created_at": _serialize_timestamp(lookup[post_id].get("created_at")),
            "text": lookup[post_id].get("text"),
        }
        for post_id in selected_ids
        if post_id in lookup
    ]
    return {"cluster_id": cluster_id, "offset": offset, "limit": limit, "total": len(ids), "posts": items}


def _page(name: str) -> FileResponse:
    page = WEB_DIR / name
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Website files are missing")
    return FileResponse(page)


@app.get("/")
def menu_page() -> FileResponse:
    return _page("index.html")


@app.get("/explore")
def explore_page() -> FileResponse:
    return _page("explore.html")


@app.get("/live")
def live_page() -> FileResponse:
    return _page("live.html")
