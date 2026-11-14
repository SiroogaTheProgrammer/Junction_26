import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
import httpx
from openai import APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletion

from detector.analyze import AnalysisCallError, _request_analysis, _warm_up
from detector.config import Settings
from detector.io import completion_answer
from detector.llm_check import run as llm_check
from tests.test_analyze_rules import _analysis


def completion(content=None, finish_reason="stop", **message_fields):
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, **message_fields},
            }],
            "usage": {"prompt_tokens": 24, "completion_tokens": 16, "total_tokens": 40},
        }
    )


def test_reasoning_only_truncation_is_not_a_success():
    response = completion(finish_reason="length", reasoning="Still reasoning.")
    with pytest.raises(ValueError, match="finish_reason=length"):
        completion_answer(response)


def test_partial_answer_at_token_limit_is_not_accepted():
    with pytest.raises(ValueError, match="output token limit"):
        completion_answer(completion('{"label":', "length"))


def test_empty_answer_includes_diagnostic_metadata():
    with pytest.raises(ValueError, match="finish_reason=stop, completion_tokens=16"):
        completion_answer(completion(reasoning="No final answer."))


def test_complete_answer_is_returned():
    assert completion_answer(completion("READY")) == "READY"


def test_analysis_disables_reasoning_and_parses_answer():
    client = AsyncMock()
    client.chat.completions.create.return_value = completion(_analysis("low").model_dump_json())
    _, parsed, attempts = asyncio.run(_request_analysis(client, Settings(_env_file=None), "test prompt"))
    assert parsed.label == "Shared local concern"
    assert attempts == 1
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_analysis_failure_preserves_reasoning_and_usage():
    client = AsyncMock()
    client.chat.completions.create.return_value = completion(
        finish_reason="length", reasoning="Still reasoning."
    )
    with pytest.raises(AnalysisCallError) as error:
        asyncio.run(_request_analysis(client, Settings(_env_file=None), "test prompt"))
    raw = json.loads(error.value.raw_response)
    assert raw["choices"][0]["message"]["reasoning"] == "Still reasoning."
    assert raw["usage"]["completion_tokens"] == 16
    assert error.value.attempts == 3
    budgets = [call.kwargs["max_tokens"] for call in client.chat.completions.create.call_args_list]
    assert budgets == [4096, 8192, 16384]


@pytest.mark.parametrize("failures", [1, 2])
def test_truncated_output_recovers_with_larger_budget_and_compact_request(failures):
    client = AsyncMock()
    client.chat.completions.create.side_effect = [
        *[completion('{"incomplete":', "length") for _ in range(failures)],
        completion(_analysis().model_dump_json()),
    ]
    with patch("detector.analyze.asyncio.sleep", new_callable=AsyncMock) as sleep:
        _, parsed, attempts = asyncio.run(_request_analysis(
            client, Settings(_env_file=None), "Original prompt with all sampled posts",
        ))
    assert attempts == failures + 1
    assert parsed.label == "Shared local concern"
    calls = client.chat.completions.create.call_args_list
    assert [call.kwargs["max_tokens"] for call in calls] == [4096, 8192, 16384][:failures + 1]
    for call in calls:
        assert call.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert call.kwargs["messages"][1]["content"].startswith("Original prompt with all sampled posts")
    assert "compact final JSON" in calls[-1].kwargs["messages"][1]["content"]
    sleep.assert_not_called()


def test_length_recovery_obeys_ceiling_and_preserves_final_raw_failure():
    client = AsyncMock()
    client.chat.completions.create.return_value = completion("partial", "length")
    settings = Settings(_env_file=None, llm_max_tokens=4096, llm_max_tokens_ceiling=5000)
    with pytest.raises(AnalysisCallError, match="remained truncated") as error:
        asyncio.run(_request_analysis(client, settings, "prompt"))
    assert error.value.attempts == 3
    assert json.loads(error.value.raw_response)["choices"][0]["finish_reason"] == "length"
    assert [call.kwargs["max_tokens"] for call in client.chat.completions.create.call_args_list] == [4096, 5000, 5000]
    with pytest.raises(ValueError, match="CEILING"):
        Settings(_env_file=None, llm_max_tokens=8192, llm_max_tokens_ceiling=4096)


def test_health_check_disables_reasoning_and_uses_configured_cap():
    settings = Settings(_env_file=None, llm_api_key="test-key", llm_model="test-model", llm_max_tokens=4096)
    client = AsyncMock()
    client.chat.completions.create.return_value = completion("READY")
    with (
        patch("detector.llm_check.get_settings", return_value=settings),
        patch("detector.llm_check._wait_until_ready", new_callable=AsyncMock),
        patch("detector.llm_check.create_client", return_value=client),
    ):
        asyncio.run(llm_check())
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 4096
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    client.close.assert_awaited_once()


@pytest.mark.parametrize("failure", ["timeout", 429, 500, 503])
def test_transient_failures_retry_with_controlled_backoff(failure):
    client = AsyncMock()
    request = httpx.Request("POST", "https://test.invalid/v1/chat/completions")
    error = (
        APITimeoutError(request=request) if failure == "timeout"
        else APIStatusError("Transient", response=httpx.Response(failure, request=request), body=None)
    )
    client.chat.completions.create.side_effect = [
        error, error, error, completion(_analysis().model_dump_json()),
    ]
    with patch("detector.analyze.asyncio.sleep", new_callable=AsyncMock) as sleep:
        _, _, attempts = asyncio.run(_request_analysis(client, Settings(_env_file=None), "prompt"))
    assert attempts == 4
    assert [call.args[0] for call in sleep.call_args_list] == [2, 4, 8]


def test_timeout_exhaustion_and_non_retryable_errors_fail_explicitly():
    client = AsyncMock()
    request = httpx.Request("POST", "https://test.invalid/v1/chat/completions")
    client.chat.completions.create.side_effect = APITimeoutError(request=request)
    with patch("detector.analyze.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(AnalysisCallError) as error:
            asyncio.run(_request_analysis(client, Settings(_env_file=None), "prompt"))
    assert error.value.attempts == 4
    client.chat.completions.create.reset_mock()
    client.chat.completions.create.side_effect = APIStatusError(
        "Invalid", response=httpx.Response(400, request=request, text="Bad request"), body=None,
    )
    with pytest.raises(AnalysisCallError) as error:
        asyncio.run(_request_analysis(client, Settings(_env_file=None), "prompt"))
    assert error.value.attempts == 1
    assert error.value.raw_response == "Bad request"


@pytest.mark.parametrize("invalid", ["not JSON", '{"label": "missing fields"}'])
def test_invalid_output_gets_one_corrective_retry(invalid):
    client = AsyncMock()
    client.chat.completions.create.side_effect = [
        completion(invalid), completion(_analysis().model_dump_json()),
    ]
    _, _, attempts = asyncio.run(_request_analysis(client, Settings(_env_file=None), "prompt"))
    assert attempts == 2
    assert client.chat.completions.create.call_args.kwargs["messages"][-1]["role"] == "user"


def test_second_invalid_output_is_preserved_and_refusal_is_not_retried():
    client = AsyncMock()
    client.chat.completions.create.return_value = completion("broken")
    with pytest.raises(AnalysisCallError) as error:
        asyncio.run(_request_analysis(client, Settings(_env_file=None), "prompt"))
    assert error.value.raw_response == "broken"
    assert error.value.attempts == 2
    client.chat.completions.create.return_value = completion(refusal="Cannot comply")
    with pytest.raises(AnalysisCallError, match="refused") as error:
        asyncio.run(_request_analysis(client, Settings(_env_file=None), "prompt"))
    assert error.value.attempts == 1


@pytest.mark.parametrize("status", [200, 401])
def test_warmup_readiness_and_auth_failure_are_bounded(status):
    settings = Settings(
        _env_file=None, llm_api_key="test-key", llm_base_url="https://test.invalid/v1",
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = httpx.Response(status, json={"data": [{"id": settings.llm_model}]})
    with patch("detector.llm.httpx.AsyncClient", return_value=client):
        if status == 200:
            asyncio.run(_warm_up(settings))
        else:
            with pytest.raises(RuntimeError, match="HTTP 401"):
                asyncio.run(_warm_up(settings))
    client.get.assert_awaited_once()


def test_warmup_deadline_is_enforced_without_a_request():
    settings = Settings(
        _env_file=None, llm_api_key="test-key", llm_base_url="https://test.invalid/v1",
        llm_warmup_timeout_s=1,
    )
    with patch("detector.llm.time") as clock:
        clock.monotonic.side_effect = [0, 2]
        with pytest.raises(TimeoutError, match="within 1 seconds"):
            asyncio.run(_warm_up(settings))
