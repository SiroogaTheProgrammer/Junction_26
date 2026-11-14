import asyncio
import hashlib
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from openai import APIError, APIStatusError, APITimeoutError, AsyncOpenAI

from detector.config import Settings
from detector.io import (
    CompletionTokenLimitError,
    completion_answer,
    ensure_data_dir,
    read_json,
    read_jsonl,
    read_posts,
    require_artifact,
    write_json,
    write_jsonl,
)
from detector.prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE
from detector.llm import create_client, wait_until_ready as _warm_up
from detector.rules import apply_rules, behavior_anomaly, display_tier
from detector.schema import Analysis, AnalysisRecord
from detector.stats import baseline_for, describe_signals, is_organic_looking
from detector.text import clean_post_text, extract_json

logger = logging.getLogger(__name__)


class AnalysisCallError(Exception):
    def __init__(self, message: str, attempts: int, raw_response: str = "") -> None:
        super().__init__(message)
        self.attempts = attempts
        self.raw_response = raw_response


def sample_posts(
    posts: pd.DataFrame,
    representative_ids: list[int],
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if posts.empty or sample_size <= 0:
        return posts.iloc[0:0].copy()
    indexed = posts.drop_duplicates("post_id").set_index("post_id", drop=False)
    selected: list[int] = []
    for post_id in representative_ids[: min(10, sample_size)]:
        if post_id in indexed.index and post_id not in selected:
            selected.append(int(post_id))

    desired = min(sample_size, len(indexed))
    remaining = desired - len(selected)
    earliest_count = min(5, remaining)
    ordered = posts.sort_values(["created_at", "post_id"], kind="stable")
    if earliest_count:
        for post_id in ordered["post_id"]:
            post_id = int(post_id)
            if post_id not in selected:
                selected.append(post_id)
                earliest_count -= 1
                if earliest_count == 0:
                    break

    remaining = desired - len(selected)
    if remaining:
        available = np.asarray([int(post_id) for post_id in indexed.index if int(post_id) not in selected])
        rng = np.random.default_rng(seed)
        chosen = rng.choice(available, size=min(remaining, len(available)), replace=False)
        selected.extend(int(post_id) for post_id in chosen)

    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    return indexed.loc[selected].reset_index(drop=True)


def _rounded_stats(cluster: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    selected = baseline_for(int(cluster["size"]), baseline)
    pct = selected["pct"]
    keys = (
        "size",
        "n_authors",
        "dup_ratio",
        "top_author_share",
        "burstiness",
        "top_domain_share",
        "top_hashtag_share",
        "time_span_min",
    )
    stats: dict[str, Any] = {}
    for key in keys:
        value = cluster.get(key, 0)
        stats[key] = round(float(value), 4) if isinstance(value, (float, int, np.number)) else value
    for key in ("dup_ratio", "top_author_share", "burstiness", "top_domain_share", "top_hashtag_share"):
        bounds = pct.get(key, {})
        stats[f"{key}_baseline"] = {
            percentile: round(float(bounds.get(percentile, 0.0)), 4) for percentile in ("p50", "p90")
        }
    stats.update({
        "details": cluster.get("stats", {}),
        "baseline_bucket": selected["name"], "baseline_source": selected["source"],
        "burstiness_reliable": baseline.get("burstiness_reliable", True),
        "burst_window_min": baseline.get("burst_window_min", 10),
        "signals": describe_signals(cluster, baseline),
        "stats_hash": cluster.get("stats_hash"),
    })
    return stats


def _input_hash(
    model: str,
    sample_ids: list[int],
    stats: dict[str, Any],
    organic_looking: bool,
) -> str:
    payload = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "sample_post_ids": sample_ids,
        "stats": stats,
        "organic_looking": organic_looking,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _cluster_input_hash(
    settings: Settings,
    cluster: dict[str, Any],
    cluster_posts: pd.DataFrame,
    baseline: dict[str, Any],
) -> str:
    sample = sample_posts(
        cluster_posts,
        [int(item) for item in cluster.get("representative_post_ids", [])],
        settings.sample_size,
        settings.seed + int(cluster["cluster_id"]),
    )
    stats = _rounded_stats(cluster, baseline)
    stats["rules"] = {
        "anomaly": behavior_anomaly(cluster, baseline, settings.anomaly_percentile).model_dump(),
        "anomaly_percentile": settings.anomaly_percentile,
        "min_sample_for_confidence": settings.min_sample_for_confidence,
    }
    stats["sample_text_sha256"] = hashlib.sha256(
        json.dumps([str(text) for text in sample["text"]], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    sample_ids = [int(post_id) for post_id in sample["post_id"]]
    return _input_hash(
        settings.llm_model,
        sample_ids,
        stats,
        is_organic_looking(cluster, baseline),
    )


def build_prompt(
    stats: dict[str, Any],
    sample: pd.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    prompt_posts: list[str] = []
    sample_map: list[dict[str, Any]] = []
    for index, row in enumerate(sample.itertuples(index=False), start=1):
        pid = f"p{index}"
        safe_text = clean_post_text(str(row.text))
        prompt_posts.append(f'<post id="{pid}">{safe_text}</post>')
        sample_map.append({"pid": pid, "post_id": int(row.post_id)})
    fields = dict(stats)
    fields.setdefault("baseline_bucket", "legacy-global")
    fields.setdefault("baseline_source", "global")
    fields.setdefault("burst_window_min", 10)
    fields.setdefault("burstiness_reliable", True)
    fields["signals"] = "\n".join(clean_post_text(line, 500) for line in stats.get("signals", [])) or "No above-p90 signals."
    fields["details"] = clean_post_text(json.dumps(stats.get("details", {}), ensure_ascii=False), 4000)
    for metric, prefix in (
        ("dup_ratio", "dup"), ("top_author_share", "author"), ("burstiness", "burst"),
        ("top_domain_share", "domain"), ("top_hashtag_share", "hashtag"),
    ):
        for percentile in ("p50", "p90"):
            fields[f"{prefix}_{percentile}"] = stats[f"{metric}_baseline"][percentile]
    user_prompt = USER_TEMPLATE.format(**fields, posts="\n".join(prompt_posts))
    return user_prompt, sample_map


async def _request_analysis(
    client: AsyncOpenAI,
    settings: Settings,
    user_prompt: str,
) -> tuple[str, Analysis, int]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "cluster_analysis",
            "strict": True,
            "schema": Analysis.output_json_schema(),
        },
    }
    attempts = 0
    status_retries = 0
    validation_retries = 0
    length_retries = 0
    token_budget = settings.llm_max_tokens
    compact_note = ""
    validation_note = ""
    raw_response = ""
    while True:
        attempts += 1
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_prompt + validation_note + compact_note,
                    },
                ],
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=token_budget,
                response_format=response_format,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except APIStatusError as exc:
            if exc.status_code == 429 or 500 <= exc.status_code <= 599:
                if status_retries < 3:
                    await asyncio.sleep(2**(status_retries + 1))
                    status_retries += 1
                    continue
            response_body = exc.response.text if exc.response is not None else ""
            raise AnalysisCallError(str(exc), attempts, response_body) from exc
        except APITimeoutError as exc:
            if status_retries < 3:
                await asyncio.sleep(2**(status_retries + 1))
                status_retries += 1
                continue
            raise AnalysisCallError("LLM request timed out after retries", attempts) from exc
        except APIError as exc:
            raise AnalysisCallError(str(exc), attempts) from exc

        try:
            raw_response = completion_answer(response)
        except CompletionTokenLimitError as exc:
            if length_retries >= 2:
                raise AnalysisCallError(
                    f"LLM output remained truncated after {length_retries} recovery retries "
                    f"(max_tokens={token_budget}). Check Verda's generation/context limits and "
                    "that its chat template honors enable_thinking=False.",
                    attempts, response.model_dump_json(),
                ) from exc
            length_retries += 1
            token_budget = min(token_budget * 2, settings.llm_max_tokens_ceiling)
            compact_note = (
                "\n\nThe previous answer exceeded its output budget. Return compact final JSON only; "
                "do not output reasoning or repeat the posts. Use short claim text, at most five claims, "
                "only clearly evidenced techniques, and at most three evidence ids per item. "
                "Keep label within 6 words, summary 40, relevance_reason 25 and explanation 60."
            )
            logger.warning(
                "LLM output hit the token limit; recovery %d/2 with max_tokens=%d and compact JSON",
                length_retries, token_budget,
            )
            continue
        except ValueError as exc:
            raise AnalysisCallError(str(exc), attempts, response.model_dump_json()) from exc
        try:
            return raw_response, Analysis.model_validate(extract_json(raw_response)), attempts
        except (json.JSONDecodeError, ValueError) as exc:
            if validation_retries >= 1:
                raise AnalysisCallError(
                    f"LLM returned invalid JSON after retry: {exc}",
                    attempts,
                    raw_response,
                ) from exc
            validation_retries += 1
            validation_note = f"\n\nYour previous response failed validation: {exc}. Return only corrected JSON."


async def _analyze_one(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    settings: Settings,
    cluster: dict[str, Any],
    cluster_posts: pd.DataFrame,
    baseline: dict[str, Any],
    force: bool,
    existing: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    cluster_id = int(cluster["cluster_id"])
    sample = sample_posts(
        cluster_posts,
        [int(item) for item in cluster.get("representative_post_ids", [])],
        settings.sample_size,
        settings.seed + cluster_id,
    )
    stats = _rounded_stats(cluster, baseline)
    input_hash = _cluster_input_hash(settings, cluster, cluster_posts, baseline)
    cached = existing.get(cluster_id)
    if not force and cached and cached.get("status") == "ok" and cached.get("input_hash") == input_hash:
        logger.info("Cluster %d unchanged; using cached analysis", cluster_id)
        return cached

    anomaly = behavior_anomaly(cluster, baseline, settings.anomaly_percentile)
    user_prompt, sample_map = build_prompt(stats, sample)
    started = time.monotonic()
    attempts = 0
    raw_response = ""
    try:
        async with semaphore:
            raw_response, analysis, attempts = await _request_analysis(client, settings, user_prompt)
        valid_pids = {item["pid"] for item in sample_map}
        parsed, rule_log = apply_rules(
            analysis,
            {
                **cluster, "valid_pids": valid_pids, "behavior_anomaly": anomaly.flag,
                "organic_looking": is_organic_looking(cluster, baseline),
                "min_sample_for_confidence": settings.min_sample_for_confidence,
            },
        )
        capped_by = next(
            (entry["rule"] for entry in reversed(rule_log) if entry["field"] == "concern_level"), None,
        )
        return AnalysisRecord.model_validate({
            "cluster_id": cluster_id,
            "input_hash": input_hash,
            "stats_hash": cluster.get("stats_hash"),
            "status": "ok",
            "sample": sample_map,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed": parsed.model_dump(mode="json"),
            "prompt_version": PROMPT_VERSION,
            "behavior_anomaly": anomaly.model_dump(),
            "concern_level_model": analysis.concern_level,
            "relevance_model": analysis.relevance,
            "capped_by": capped_by,
            "rule_log": rule_log,
            "attempts": attempts,
            "latency_s": round(time.monotonic() - started, 3),
            "model": settings.llm_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).model_dump(mode="json")
    except (AnalysisCallError, ValueError) as exc:
        logger.exception("Analysis failed for cluster %d", cluster_id)
        return AnalysisRecord.model_validate({
            "cluster_id": cluster_id,
            "input_hash": input_hash,
            "stats_hash": cluster.get("stats_hash"),
            "status": "failed",
            "sample": sample_map,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "raw_response": getattr(exc, "raw_response", "") or raw_response or str(exc),
            "parsed": None,
            "prompt_version": PROMPT_VERSION,
            "behavior_anomaly": anomaly.model_dump(),
            "concern_level_model": None,
            "relevance_model": None,
            "capped_by": None,
            "rule_log": [],
            "attempts": getattr(exc, "attempts", attempts),
            "latency_s": round(time.monotonic() - started, 3),
            "model": settings.llm_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).model_dump(mode="json")


def select_clusters(
    clusters: list[dict[str, Any]], settings: Settings, cluster_id: int | None = None,
    all_clusters: bool = False,
) -> list[dict[str, Any]]:
    if cluster_id is not None:
        selected = [cluster for cluster in clusters if int(cluster["cluster_id"]) == cluster_id]
        if not selected:
            raise ValueError(f"Cluster {cluster_id} was not found")
        return selected
    selected = [
        cluster for cluster in sorted(clusters, key=lambda item: (-int(item["size"]), int(item["cluster_id"])))
        if all_clusters or int(cluster["size"]) >= settings.min_analysis_size
    ]
    return selected if all_clusters or settings.max_analyzed_clusters == 0 else selected[:settings.max_analyzed_clusters]


def _write_report(
    settings: Settings, records: list[dict[str, Any]], cached_ids: set[int], started: float,
) -> None:
    kinds: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    for record in records:
        if record["status"] == "ok":
            kinds[record["parsed"]["relevance_kind"]] += 1
            tiers[display_tier(record["parsed"]["relevance"], record["behavior_anomaly"]["flag"])] += 1
        else:
            tiers["failed"] += 1
    latencies = [float(record["latency_s"]) for record in records if record["cluster_id"] not in cached_ids]
    write_json(settings.data_path / "analyze_report.json", {
        "selected": len(records), "analysed": len(records) - len(cached_ids),
        "cached": len(cached_ids), "failed": sum(record["status"] == "failed" for record in records),
        "counts_per_relevance_kind": dict(kinds),
        "tier_counts": {tier: tiers[tier] for tier in ("full", "faded", "hidden", "failed")},
        "tier_thresholds": {"hide_below": 0.35, "full_from": 0.65},
        "total_latency_s": round(sum(latencies), 3),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "elapsed_s": round(time.monotonic() - started, 3),
    })


async def run(
    settings: Settings, force: bool = False, cluster_id: int | None = None, dry_run: bool = False,
    all_clusters: bool = False,
) -> None:
    started = time.monotonic()
    if not dry_run and (
        not settings.llm_api_key or not settings.llm_api_key.get_secret_value().strip()
        or not settings.llm_base_url or "<deployment>" in settings.llm_base_url
    ):
        logger.warning("LLM_API_KEY and a real LLM_BASE_URL are required; skipping analysis.")
        return
    clusters_path = require_artifact(settings.data_path / "clusters.json")
    posts = read_posts(settings.data_path)
    cluster_data = read_json(clusters_path)
    baseline = cluster_data["baseline"]
    selected = select_clusters(cluster_data["clusters"], settings, cluster_id, all_clusters)
    logger.info("Selected %d/%d clouds for analysis", len(selected), len(cluster_data["clusters"]))
    assignments = pd.read_parquet(require_artifact(settings.data_path / "assignments.parquet"))
    post_clusters = posts.merge(assignments, on="post_id", how="inner", validate="one_to_one")
    if len(post_clusters) != len(posts) or len(assignments) != len(posts):
        raise ValueError("assignments.parquet must contain exactly one assignment for every post")
    groups = {int(key): group for key, group in post_clusters.groupby("cluster_id")}
    for cluster in selected:
        group = groups.get(int(cluster["cluster_id"]))
        if group is None or len(group) != int(cluster["size"]):
            raise ValueError(f"Cluster {cluster['cluster_id']} size does not match assignments.parquet")
        behavior_anomaly(cluster, baseline, settings.anomaly_percentile)
    if dry_run:
        for cluster in selected:
            sample = sample_posts(
                groups[int(cluster["cluster_id"])], cluster["representative_post_ids"],
                settings.sample_size, settings.seed + int(cluster["cluster_id"]),
            )
            prompt, sample_map = build_prompt(_rounded_stats(cluster, baseline), sample)
            print(json.dumps({
                "cluster_id": cluster["cluster_id"], "prompt_version": PROMPT_VERSION,
                "input_hash": _cluster_input_hash(settings, cluster, groups[int(cluster["cluster_id"])], baseline),
                "system_prompt": SYSTEM_PROMPT, "user_prompt": prompt, "sample": sample_map,
                "behavior_anomaly": behavior_anomaly(cluster, baseline, settings.anomaly_percentile).model_dump(),
            }, ensure_ascii=False))
        return
    ensure_data_dir(settings.data_path)
    path = settings.data_path / "analyses.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if record.get("prompt_version") != PROMPT_VERSION or record.get("model") != settings.llm_model:
            continue
        try:
            validated = AnalysisRecord.model_validate(record).model_dump(mode="json")
        except ValueError:
            logger.warning("Ignoring invalid cached analysis for cluster %s", record.get("cluster_id"))
            continue
        existing[int(validated["cluster_id"])] = validated
    valid_existing = {}
    for cluster in cluster_data["clusters"]:
        key = int(cluster["cluster_id"])
        record = existing.get(key)
        if record and key in groups and record["input_hash"] == _cluster_input_hash(settings, cluster, groups[key], baseline):
            valid_existing[key] = record
    targets = {int(cluster["cluster_id"]) for cluster in selected}
    retained = {key: value for key, value in valid_existing.items() if key not in targets} if cluster_id is not None else {}
    cached = {
        key: record for key, record in valid_existing.items()
        if key in targets and record["status"] == "ok" and not force
    }
    results = {**retained, **cached}
    pending = [cluster for cluster in selected if int(cluster["cluster_id"]) not in cached]
    if pending:
        await _warm_up(settings)
    # Persist a consistent checkpoint before starting requests; each completed result replaces it atomically.
    write_jsonl(path, sorted(results.values(), key=lambda record: record["cluster_id"]))
    if pending:
        semaphore = asyncio.Semaphore(settings.llm_concurrency)
        async with create_client(settings) as client:
            tasks = [
                asyncio.create_task(_analyze_one(
                    semaphore, client, settings, cluster, groups[int(cluster["cluster_id"])],
                    baseline, force, valid_existing,
                ))
                for cluster in pending
            ]
            try:
                for task in asyncio.as_completed(tasks):
                    record = await task
                    results[int(record["cluster_id"])] = record
                    write_jsonl(path, sorted(results.values(), key=lambda item: item["cluster_id"]))
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    selected_results = [results[int(cluster["cluster_id"])] for cluster in selected]
    _write_report(settings, selected_results, set(cached), started)
    logger.info(
        "Analysis complete in %.1fs: %d selected, %d cached, %d failed",
        time.monotonic() - started, len(selected), len(cached),
        sum(record["status"] == "failed" for record in selected_results),
    )
