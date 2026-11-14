from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TechniqueName = Literal[
    "fear_appeal",
    "scapegoating",
    "false_urgency",
    "loaded_language",
    "false_dichotomy",
    "conspiracy_framing",
    "unverified_authority",
    "call_to_mass_action",
    "whataboutism",
]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    kind: Literal["checkable", "opinion"]
    evidence: list[str]


class Technique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: TechniqueName
    evidence: list[str]


class Templating(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected: bool
    evidence: list[str]


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    summary: str
    relevance_kind: Literal[
        "narrative", "issue_discussion", "news_relay", "plain_topic",
        "personal_chatter", "commercial_or_spam", "unclear",
    ]
    relevance: float = Field(allow_inf_nan=False)
    relevance_reason: str
    claims: list[Claim] = Field(max_length=5)
    techniques: list[Technique]
    templating: Templating
    framing_diversity: Literal["low", "medium", "high"]
    legit_lookalike: str | None
    concern_level: Literal["none", "low", "medium", "high", "insufficient_evidence"]
    confidence: float = Field(allow_inf_nan=False)
    explanation: str

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        return value.strip().removeprefix("## ").strip()

    @classmethod
    def output_json_schema(cls) -> dict:
        schema = cls.model_json_schema()
        for field in ("relevance", "confidence"):
            schema["properties"][field].update(minimum=0, maximum=1)
        return schema


class Verdict(BaseModel):
    """The judge's reading of one submitted item, relative to its nearest cloud."""

    model_config = ConfigDict(extra="forbid")

    label: str
    relevance_kind: Literal[
        "narrative", "issue_discussion", "news_relay", "plain_topic",
        "personal_chatter", "commercial_or_spam", "unclear",
    ]
    relevance: float = Field(allow_inf_nan=False)
    techniques: list[TechniqueName] = Field(max_length=9)
    fits_cloud: bool
    concern_level: Literal["none", "low", "medium", "high", "insufficient_evidence"]
    confidence: float = Field(allow_inf_nan=False)
    explanation: str

    @classmethod
    def output_json_schema(cls) -> dict:
        schema = cls.model_json_schema()
        for field in ("relevance", "confidence"):
            schema["properties"][field].update(minimum=0, maximum=1)
        return schema


class ContentAssessment(BaseModel):
    """Observable persuasion in the original material, not coordination or truth."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "propaganda", "political_persuasion", "commercial_persuasion",
        "informational", "personal_expression", "unclear",
    ]
    concern_level: Literal["none", "low", "medium", "high", "insufficient_evidence"]
    techniques: list[TechniqueName] = Field(max_length=9)
    evidence: list[str] = Field(max_length=5)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    explanation: str = Field(max_length=1200)

    @field_validator("evidence")
    @classmethod
    def clean_evidence(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))

    @model_validator(mode="after")
    def require_evidence(self) -> Self:
        if (self.category == "propaganda" or self.concern_level in {"medium", "high"}) and not self.evidence:
            raise ValueError("Propaganda or raised content concern requires observable evidence")
        return self

    @classmethod
    def output_json_schema(cls) -> dict:
        return cls.model_json_schema()


class AnomalyResult(BaseModel):
    flag: bool
    triggered_by: list[str]


class ClusterStats(BaseModel):
    size: int
    n_authors: int
    authors_per_post: float
    top_author_share: float
    top5_author_share: float
    exact_dup_ratio: float
    dup_ratio: float
    dup_authors: int
    largest_dup_group: int
    burstiness: float
    time_span_min: float
    first_post_at: str | None
    last_post_at: str | None
    top_domain: str | None
    top_domain_share: float
    link_share: float
    top_hashtag: str | None
    top_hashtag_share: float
    image_share: float | None


class Percentiles(BaseModel):
    p50: float
    p75: float
    p90: float


class BaselineGroup(BaseModel):
    n_clusters: int
    pct: dict[str, Percentiles]


class BaselineBucket(BaselineGroup):
    min_size: int
    max_size: int | None
    source: Literal["bucket", "global"]


class Baseline(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    buckets: list[BaselineBucket]
    global_: BaselineGroup = Field(alias="global")
    burstiness_reliable: bool
    burst_window_min: int
    min_analysis_size: int


class SamplePost(BaseModel):
    pid: str
    post_id: int


class AnalysisRecord(BaseModel):
    cluster_id: int
    input_hash: str
    stats_hash: str | None = None
    prompt_version: str
    model: str
    status: Literal["ok", "failed"]
    sample: list[SamplePost]
    behavior_anomaly: AnomalyResult
    parsed: Analysis | None
    concern_level_model: str | None
    relevance_model: float | None
    capped_by: str | None
    rule_log: list[dict[str, Any]]
    system_prompt: str
    user_prompt: str
    raw_response: str
    attempts: int
    latency_s: float
    created_at: datetime
