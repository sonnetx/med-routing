from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class TierSpec:
    """One tier in the cascade: a model + its USD/1M token pricing."""
    name: str
    model: str
    prompt_per_m: float
    completion_per_m: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    # 3-tier cascade. Names match cascading_demo so storyline and pricing are
    # consistent across the two demos.
    tier_nano_model: str = Field(default="gpt-5.4-nano", alias="TIER_NANO_MODEL")
    tier_mini_model: str = Field(default="gpt-5.4-mini", alias="TIER_MINI_MODEL")
    tier_base_model: str = Field(default="gpt-5.4", alias="TIER_BASE_MODEL")

    threshold_self_reported: float = Field(default=0.15, alias="THRESHOLD_SELF_REPORTED")
    threshold_predictive_entropy: float = Field(default=0.40, alias="THRESHOLD_PREDICTIVE_ENTROPY")
    threshold_semantic_entropy: float = Field(default=0.50, alias="THRESHOLD_SEMANTIC_ENTROPY")
    threshold_self_consistency: float = Field(default=0.40, alias="THRESHOLD_SELF_CONSISTENCY")
    threshold_routellm: float = Field(default=0.50, alias="THRESHOLD_ROUTELLM")
    threshold_learned: float = Field(default=0.50, alias="THRESHOLD_LEARNED")
    threshold_auto: float = Field(default=0.50, alias="THRESHOLD_AUTO")
    threshold_semantic_entropy_embed: float = Field(
        default=0.50, alias="THRESHOLD_SEMANTIC_ENTROPY_EMBED"
    )

    embed_model: str = Field(default="text-embedding-3-small", alias="EMBED_MODEL")
    judge_model: str = Field(default="gpt-4o-mini", alias="JUDGE_MODEL")
    demo_data_dir: str = Field(default="demo_data", alias="DEMO_DATA_DIR")

    sample_n: int = Field(default=5, alias="SAMPLE_N")
    sample_temperature: float = Field(default=0.7, alias="SAMPLE_TEMPERATURE")

    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-large",
        alias="NLI_MODEL",
    )
    enable_nli: bool = Field(default=False, alias="ENABLE_NLI")

    enable_routellm: bool = Field(default=False, alias="ENABLE_ROUTELLM")
    routellm_kind: str = Field(default="mf", alias="ROUTELLM_KIND")
    routellm_strong_model: str = Field(default="gpt-4-1106-preview", alias="ROUTELLM_STRONG_MODEL")
    routellm_weak_model: str = Field(
        default="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
        alias="ROUTELLM_WEAK_MODEL",
    )

    enable_learned: bool = Field(default=False, alias="ENABLE_LEARNED")
    learned_router_path: str = Field(
        default="models/learned_router.pkl", alias="LEARNED_ROUTER_PATH"
    )

    cache_size: int = Field(default=2048, alias="CACHE_SIZE")

    home_region: str = Field(default="US", alias="HOME_REGION")
    audit_dir: str = Field(default="audit", alias="AUDIT_DIR")
    db_path: str = Field(default="data/med_routing.db", alias="DB_PATH")

    @property
    def tiers(self) -> list[TierSpec]:
        # ordered cheap → expensive; cascade walks this list
        return [
            TierSpec("nano", self.tier_nano_model, 0.20, 1.25),
            TierSpec("mini", self.tier_mini_model, 0.75, 4.50),
            TierSpec("base", self.tier_base_model, 2.50, 15.00),
        ]

    def threshold_for(self, router_name: str) -> float:
        if router_name in _RUNTIME_OVERRIDES:
            return _RUNTIME_OVERRIDES[router_name]
        return {
            "self_reported": self.threshold_self_reported,
            "predictive_entropy": self.threshold_predictive_entropy,
            "semantic_entropy": self.threshold_semantic_entropy,
            "semantic_entropy_embed": self.threshold_semantic_entropy_embed,
            "self_consistency": self.threshold_self_consistency,
            "routellm": self.threshold_routellm,
            "learned": self.threshold_learned,
            "auto": self.threshold_auto,
        }.get(router_name, 0.5)

    # Back-compat shims so any straggler code referencing the old field names
    # still gets a usable model id (first / last tier). The intent is that all
    # call sites use `tiers` going forward.
    @property
    def weak_model(self) -> str:
        return self.tiers[0].model

    @property
    def strong_model(self) -> str:
        return self.tiers[-1].model


def _build_pricing(s: "Settings") -> dict[str, dict[str, float]]:
    return {t.model: {"prompt": t.prompt_per_m, "completion": t.completion_per_m} for t in s.tiers}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _build_pricing(get_settings())
    p = pricing.get(model, {"prompt": 0.0, "completion": 0.0})
    return (prompt_tokens * p["prompt"] + completion_tokens * p["completion"]) / 1_000_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Read-only snapshot exposed for templates / debug endpoints; rebuilt lazily.
def pricing_table() -> dict[str, dict[str, float]]:
    return _build_pricing(get_settings())


_RUNTIME_OVERRIDES: dict[str, float] = {}


def set_runtime_threshold(router: str, threshold: float) -> None:
    _RUNTIME_OVERRIDES[router] = float(threshold)


def clear_runtime_threshold(router: str) -> None:
    _RUNTIME_OVERRIDES.pop(router, None)


def runtime_overrides() -> dict[str, float]:
    return dict(_RUNTIME_OVERRIDES)
