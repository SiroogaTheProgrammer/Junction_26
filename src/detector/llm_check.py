import asyncio
import logging
import time

from detector.config import get_settings
from detector.io import completion_answer
from detector.llm import create_client, wait_until_ready as _wait_until_ready

async def run() -> None:
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is required for make llm-check")
    await _wait_until_ready(settings)
    client = create_client(settings)
    try:
        started = time.monotonic()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "Reply with the single word READY."}],
            max_tokens=settings.llm_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = time.monotonic() - started
        answer = completion_answer(response)
        print(f"LLM endpoint OK: model={settings.llm_model} latency_s={elapsed:.2f} response={answer.strip()!r}")
    finally:
        await client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run())
