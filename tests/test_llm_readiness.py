import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from detector.config import Settings
from detector.llm import create_client, wait_until_ready


def settings(**overrides):
    return Settings(
        _env_file=None, llm_api_key="test-key", llm_base_url="https://test.invalid/v1",
        **overrides,
    )


def test_shared_client_disables_hidden_sdk_retries():
    with patch("detector.llm.AsyncOpenAI") as factory:
        create_client(settings())
    assert factory.call_args.kwargs == {
        "base_url": "https://test.invalid/v1",
        "api_key": "test-key",
        "timeout": 120,
        "max_retries": 0,
    }


@pytest.mark.parametrize("response,message", [
    (httpx.Response(200, json={"data": [{"id": "different-model"}]}), "is not served"),
    (httpx.Response(200, text="gateway placeholder"), "not valid JSON"),
    (httpx.Response(200, json={"data": "invalid"}), "invalid OpenAI"),
    (httpx.Response(403), "HTTP 403"),
    (httpx.Response(404), "HTTP 404"),
    (httpx.Response(302), "HTTP 302"),
])
def test_invalid_readiness_fails_immediately(response, message):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = response
    with patch("detector.llm.httpx.AsyncClient", return_value=client):
        with pytest.raises(RuntimeError, match=message):
            asyncio.run(wait_until_ready(settings()))
    client.get.assert_awaited_once()


@pytest.mark.parametrize("failure", ["timeout", "connection", 429, 503])
def test_readiness_recovers_from_transient_errors_without_logging_secrets(failure, caplog):
    s = settings()
    client = AsyncMock()
    client.__aenter__.return_value = client
    error = {
        "timeout": httpx.ReadTimeout("sensitive exception details"),
        "connection": httpx.ConnectError("sensitive exception details"),
    }.get(failure)
    client.get.side_effect = [
        error if error is not None else httpx.Response(failure),
        httpx.Response(200, json={"data": [{"id": s.llm_model}]}),
    ]
    with (
        patch("detector.llm.httpx.AsyncClient", return_value=client),
        patch("detector.llm.asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        asyncio.run(wait_until_ready(s))
    assert client.get.await_count == 2
    assert sleep.call_args.args[0] == 10
    assert client.get.call_args.kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert "remaining=" in caplog.text
    assert "test-key" not in caplog.text
    assert "sensitive exception details" not in caplog.text


def test_overall_deadline_cancels_stalled_network_request():
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.side_effect = stalled
    # Shorten the outer deadline while keeping Settings' production integer constraint.
    real_timeout = asyncio.timeout
    with (
        patch("detector.llm.httpx.AsyncClient", return_value=client),
        patch("detector.llm.asyncio.timeout", side_effect=lambda _: real_timeout(0.01)),
    ):
        with pytest.raises(TimeoutError, match="existing results were preserved"):
            asyncio.run(wait_until_ready(settings(llm_warmup_timeout_s=1)))
    client.__aexit__.assert_awaited_once()
