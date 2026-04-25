from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    weak_model: str = Field(default="gpt-4o-mini", alias="WEAK_MODEL")
    strong_model: str = Field(default="gpt-4o", alias="STRONG_MODEL")

    threshold_self_reported: float = Field(default=0.15, alias="THRESHOLD_SELF_REPORTED")
    threshold_predictive_entropy: float = Field(default=0.40, alias="THRESHOLD_PREDICTIVE_ENTROPY")
    threshold_semantic_entropy: float = Field(default=0.50, alias="THRESHOLD_SEMANTIC_ENTROPY")
    threshold_self_consistency: float = Field(default=0.40, alias="THRESHOLD_SELF_CONSISTENCY")

    sample_n: int = Field(default=5, alias="SAMPLE_N")
    sample_temperature: float = Field(default=0.7, alias="SAMPLE_TEMPERATURE")

    nli_model: str = Field(default="microsoft/deberta-v3-large-mnli", alias="NLI_MODEL")
    enable_nli: bool = Field(default=False, alias="ENABLE_NLI")

    cache_size: int = Field(default=2048, alias="CACHE_SIZE")

    def threshold_for(self, router_name: str) -> float:
        return {
            "self_reported": self.threshold_self_reported,
            "predictive_entropy": self.threshold_predictive_entropy,
            "semantic_entropy": self.threshold_semantic_entropy,
            "self_consistency": self.threshold_self_consistency,
        }.get(router_name, 0.5)


# OpenAI list pricing (USD per 1M tokens) snapshot used for live cost panel.
# Update if prices change; only labels/values displayed in Grafana.
PRICING = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, {"prompt": 0.0, "completion": 0.0})
    return (prompt_tokens * p["prompt"] + completion_tokens * p["completion"]) / 1_000_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
