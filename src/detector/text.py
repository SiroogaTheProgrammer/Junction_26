import html
import json
import re
from typing import Any
from urllib.parse import urlsplit

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
FENCE_PATTERN = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://[^\s<>]+|www\.[^\s<>]+", re.IGNORECASE)


def extract_json(raw_response: str) -> dict[str, Any]:
    cleaned = THINK_PATTERN.sub("", raw_response)
    cleaned = FENCE_PATTERN.sub("", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response did not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Response JSON must be an object")
    return value


def clean_post_text(text: str, limit: int = 400) -> str:
    def domain(match: re.Match[str]) -> str:
        value = match.group().rstrip(".,!?;:)]}")
        try:
            parsed = urlsplit(value if "://" in value else "https://" + value)
            host = parsed.hostname or "unknown"
        except ValueError:
            host = "invalid"
        return f"<url:{host}>"

    text = URL_PATTERN.sub(domain, text)
    return html.escape(" ".join(text.split())[:limit], quote=False)
