import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openai.types.chat import ChatCompletion
from qdrant_client import QdrantClient

from detector.config import Settings


class CompletionTokenLimitError(ValueError):
    pass


def completion_answer(response: ChatCompletion) -> str:
    if not response.choices:
        raise ValueError("LLM returned no choices")
    choice = response.choices[0]
    if choice.message.refusal:
        raise ValueError(f"LLM refused the request: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise CompletionTokenLimitError(
            "LLM reached the output token limit before completing its answer "
            "(finish_reason=length). Check reasoning is disabled or increase LLM_MAX_TOKENS."
        )
    answer = choice.message.content or ""
    if not answer.strip():
        usage = response.usage
        tokens = usage.completion_tokens if usage else "unknown"
        raise ValueError(
            f"LLM returned no answer text (finish_reason={choice.finish_reason}, "
            f"completion_tokens={tokens}). Check the server's reasoning/chat-template configuration."
        )
    return answer


def ensure_data_dir(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def artifact_path(data_dir: Path, name: str) -> Path:
    return data_dir / name


def dataset_fingerprint(uris: list[str], n_posts: int, seed: int) -> str:
    digest = hashlib.sha256()
    for uri in sorted(uris):
        digest.update(uri.encode("utf-8"))
        digest.update(b"\n")
    return hashlib.sha256(f"{n_posts}:{seed}:{digest.hexdigest()}".encode("utf-8")).hexdigest()


def qdrant_client(settings: Settings) -> QdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=30)


def require_artifact(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}. Run the preceding pipeline stage first.")
    return path


def read_posts(data_dir: Path) -> pd.DataFrame:
    path = require_artifact(artifact_path(data_dir, "posts.parquet"))
    return pd.read_parquet(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)
