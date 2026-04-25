from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Processor:
    """A model endpoint's underlying legal processor and data residency.

    Used to drive GDPR Article 30 (records of processing) audit rows and the
    data-residency panel. `region` is a short code (US / EU / UK / DE / ...)
    suitable for grouping in Grafana.
    """

    name: str
    entity: str
    region: str
    dpa_ref: str = ""
    notes: str = ""


_UNKNOWN = Processor(name="unknown", entity="unknown", region="unknown")

DEFAULT_REGISTRY: dict[str, Processor] = {
    "gpt-4o-mini": Processor(
        name="openai-gpt-4o-mini",
        entity="OpenAI, Inc.",
        region="US",
        dpa_ref="OpenAI DPA 2024",
    ),
    "gpt-4o": Processor(
        name="openai-gpt-4o",
        entity="OpenAI, Inc.",
        region="US",
        dpa_ref="OpenAI DPA 2024",
    ),
    "claude-haiku-4-5": Processor(
        name="anthropic-claude-haiku",
        entity="Anthropic PBC",
        region="US",
        dpa_ref="Anthropic DPA 2024",
    ),
    "claude-opus-4-7": Processor(
        name="anthropic-claude-opus",
        entity="Anthropic PBC",
        region="US",
        dpa_ref="Anthropic DPA 2024",
    ),
    "local-llama": Processor(
        name="local-llama",
        entity="self-hosted",
        region="EU",
        notes="On-prem inference, no third-party data transfer.",
    ),
}


def get_processor(model: str) -> Processor:
    return DEFAULT_REGISTRY.get(model, Processor(name=f"unknown-{model}", entity="unknown", region="unknown"))


def is_cross_border(region: str, home_region: str) -> bool:
    """True if `region` is a known foreign region relative to `home_region`."""
    if region in {"unknown", ""}:
        return True  # treat unknown as worst-case for compliance
    return region != home_region
