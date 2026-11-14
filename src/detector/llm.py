import asyncio
import logging
import time

import httpx
from openai import AsyncOpenAI

from detector.config import Settings

logger = logging.getLogger(__name__)


def create_client(settings: Settings) -> AsyncOpenAI:
    if not settings.llm_api_key or not settings.llm_api_key.get_secret_value().strip():
        raise ValueError("LLM_API_KEY is required")
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_s,
        max_retries=0,
    )


def super_configured(settings: Settings) -> bool:
    key = settings.super_llm_api_key
    url = settings.super_llm_base_url.strip()
    return bool(key and key.get_secret_value().strip() and url and "<deployment>" not in url)


def create_super_client(settings: Settings) -> AsyncOpenAI:
    if not super_configured(settings):
        raise ValueError("SUPER_LLM_API_KEY and SUPER_LLM_BASE_URL are required")
    return AsyncOpenAI(
        base_url=settings.super_llm_base_url,
        api_key=settings.super_llm_api_key.get_secret_value(),
        timeout=settings.super_llm_timeout_s,
        max_retries=0,
    )


async def wait_until_ready(settings: Settings) -> None:
    if not settings.llm_api_key or not settings.llm_api_key.get_secret_value().strip():
        raise ValueError("LLM_API_KEY is required")
    if not settings.llm_base_url.strip() or "<deployment>" in settings.llm_base_url:
        raise ValueError("Set LLM_BASE_URL to the deployed OpenAI-compatible API base URL, ending in /v1")
    headers = {"Authorization": "Bearer " + settings.llm_api_key.get_secret_value()}
    started = time.monotonic()
    budget = settings.llm_warmup_timeout_s
    last_status = "no response"
    logger.info(
        "Checking LLM API readiness (budget=%ds, model=%s). "
        "A serverless cold start can take minutes; production should keep a warm replica.",
        budget, settings.llm_model,
    )

    def timeout_error() -> TimeoutError:
        return TimeoutError(
            f"LLM API did not become ready within {budget} seconds; last result: {last_status}. "
            "Check Verda container health, complete engine startup logs, GPU memory, and whether "
            "the configured model is supported by the deployed vLLM image. Keep a warm replica "
            "for production. No analysis requests were sent and existing results were preserved."
        )

    try:
        async with asyncio.timeout(budget):
            async with httpx.AsyncClient(timeout=10) as client:
                while True:
                    remaining = budget - (time.monotonic() - started)
                    if remaining <= 0:
                        raise timeout_error()
                    try:
                        response = await client.get(
                            settings.llm_base_url.rstrip("/") + "/models",
                            headers=headers,
                            timeout=min(10, remaining),
                        )
                    except httpx.RequestError as exc:
                        last_status = type(exc).__name__
                    else:
                        if response.status_code == 200:
                            try:
                                body = response.json()
                            except ValueError as exc:
                                raise RuntimeError("LLM /models returned HTTP 200 but not valid JSON") from exc
                            models = body.get("data") if isinstance(body, dict) else None
                            if not isinstance(models, list) or not all(
                                isinstance(item, dict) and isinstance(item.get("id"), str)
                                for item in models
                            ):
                                raise RuntimeError("LLM /models returned an invalid OpenAI model-list response")
                            names = [item["id"] for item in models]
                            if settings.llm_model not in names:
                                raise RuntimeError(
                                    f"Configured LLM_MODEL={settings.llm_model!r} is not served. "
                                    f"Available models: {names}"
                                )
                            logger.info("LLM API ready after %.1fs; configured model is served", time.monotonic() - started)
                            return
                        last_status = f"HTTP {response.status_code}"
                        if response.status_code != 429 and not 500 <= response.status_code <= 599:
                            raise RuntimeError(
                                f"LLM /models returned {last_status}. Check LLM_BASE_URL "
                                "(including /v1), inference credentials, and deployment routing."
                            )
                    remaining = budget - (time.monotonic() - started)
                    logger.warning(
                        "LLM API not ready: %s; elapsed=%.0fs, remaining=%.0fs",
                        last_status, time.monotonic() - started, max(0, remaining),
                    )
                    if remaining <= 0:
                        raise timeout_error()
                    await asyncio.sleep(min(10, remaining))
    except TimeoutError:
        raise timeout_error() from None
