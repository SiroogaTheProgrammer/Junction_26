import re
from typing import Any

import numpy as np

from detector.config import Settings

VECTOR_SIZE = 384


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EMBED_DEVICE=cuda was requested but CUDA is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("EMBED_DEVICE=mps was requested but MPS is unavailable")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported EMBED_DEVICE: {requested}")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_for_embedding(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "<url>", str(text), flags=re.IGNORECASE)
    return " ".join(text.split())


def load_model(settings: Settings, device: str) -> Any:
    # Imported lazily: loading torch is slow and the API only needs it for chat placement.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model, device=device)
    model.max_seq_length = settings.embed_max_tokens
    return model


def encode(model: Any, texts: list[str], batch_size: int, progress: bool = True) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != VECTOR_SIZE:
        raise ValueError(f"Expected {VECTOR_SIZE}-dimensional embeddings, got shape {result.shape}")
    return result
