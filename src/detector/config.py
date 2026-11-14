from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    hf_dataset: str = "itsmebatuhan/bluesky-10m-posts-15-languages"
    n_posts: int = Field(default=10_000, ge=0)
    seed: int = 42
    min_post_chars: int = Field(default=20, ge=0)
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_max_tokens: int = Field(default=128, ge=1)
    embed_batch: int = Field(default=64, ge=1)
    embed_device: str = "auto"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "posts"
    umap_n_neighbors: int = Field(default=15, ge=2)
    umap_n_components: int = Field(default=5, ge=2)
    umap_min_dist: float = Field(default=0.0, ge=0.0)
    hdbscan_min_cluster_size: int = Field(default=25, ge=2)
    hdbscan_min_samples: int = Field(default=5, ge=1)
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://containers.verda.com/<deployment>/v1"
    llm_model: str = "nemotron-lightning"
    llm_temperature: float = 1.0
    llm_top_p: float = 0.95
    llm_max_tokens: int = Field(default=4096, ge=1)
    llm_max_tokens_ceiling: int = Field(default=16384, ge=1)
    llm_concurrency: int = Field(default=8, ge=1)
    llm_timeout_s: float = Field(default=120, gt=0)
    llm_warmup_timeout_s: int = Field(default=900, gt=0)
    min_analysis_size: int = Field(default=1, ge=1)
    max_analyzed_clusters: int = Field(default=0, ge=0)
    sample_size: int = Field(default=50, ge=1)
    relevance_hide_below: float = Field(default=0.35, ge=0, le=1)
    relevance_full_from: float = Field(default=0.65, ge=0, le=1)
    anomaly_percentile: Literal[50, 75, 90] = 90
    burst_window_min: int = Field(default=10, ge=1)
    near_dup_jaccard: float = Field(default=0.8, gt=0, le=1)
    minhash_perm: int = Field(default=64, ge=16)
    baseline_buckets: str = "25,50,200"
    min_clusters_per_bucket: int = Field(default=10, ge=1)
    min_sample_for_confidence: int = Field(default=8, ge=1)
    site_address: str = ":80"
    super_llm_api_key: SecretStr | None = None
    super_llm_base_url: str = ""
    super_llm_model: str = "nemotron-3.5-super-vl"
    super_llm_max_tokens: int = Field(default=4096, ge=1)
    super_llm_timeout_s: float = Field(default=120, gt=0)
    chat_judge: Literal["lightning", "super"] = "lightning"
    chat_judge_timeout_s: float = Field(default=60, gt=0)
    chat_assessment_max_tokens: int = Field(default=32768, ge=2048, le=32768)
    chat_neighbors: int = Field(default=25, ge=1)
    chat_max_attachments: int = Field(default=4, ge=0)
    chat_max_attachment_mb: float = Field(default=4, gt=0)
    chat_max_per_minute: int = Field(default=20, ge=1)
    chat_concurrency: int = Field(default=2, ge=1)
    min_review_priority: int = Field(default=1, ge=0, le=100)
    live_posts_per_second: int = Field(default=120, ge=1)
    live_dataset: str = "itsmebatuhan/bluesky-10m-posts-15-languages"
    live_dataset_revision: str = "main"
    live_window_posts: int = Field(default=50_000, ge=1)
    live_batch_size: int = Field(default=500, ge=25, le=5000)
    live_min_cluster_size: int = Field(default=10, ge=2)
    live_min_samples: int = Field(default=3, ge=1)
    storage_cleanup: bool = True
    storage_cleanup_hours: float = Field(default=6, gt=0)

    @model_validator(mode="after")
    def validate_relevance_thresholds(self) -> "Settings":
        try:
            buckets = [int(value) for value in self.baseline_buckets.split(",")]
        except ValueError as exc:
            raise ValueError("BASELINE_BUCKETS must be comma-separated positive integers") from exc
        if not buckets or buckets[0] < 1 or buckets != sorted(set(buckets)):
            raise ValueError("BASELINE_BUCKETS must be positive, unique and increasing")
        if self.llm_max_tokens_ceiling < self.llm_max_tokens:
            raise ValueError("LLM_MAX_TOKENS_CEILING must be at least LLM_MAX_TOKENS")
        if self.relevance_hide_below > self.relevance_full_from:
            raise ValueError("RELEVANCE_HIDE_BELOW must not exceed RELEVANCE_FULL_FROM")
        if self.live_batch_size > self.live_window_posts:
            raise ValueError("LIVE_BATCH_SIZE cannot exceed LIVE_WINDOW_POSTS")
        return self

    @property
    def data_path(self) -> Path:
        return self.data_dir.expanduser()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
