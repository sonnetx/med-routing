from __future__ import annotations

from ..config import get_settings
from ..llm.openai_client import OpenAIClient
from .auto import AutoRouter
from .base import UncertaintyRouter
from .predictive_entropy import PredictiveEntropyRouter
from .self_consistency import SelfConsistencyRouter
from .self_reported import SelfReportedRouter


def build_routers(client: OpenAIClient) -> dict[str, UncertaintyRouter]:
    s = get_settings()
    routers: dict[str, UncertaintyRouter] = {
        SelfReportedRouter.name: SelfReportedRouter(client),
        SelfConsistencyRouter.name: SelfConsistencyRouter(),
        PredictiveEntropyRouter.name: PredictiveEntropyRouter(),
    }
    if s.enable_nli:
        # Heavy import gated on env flag so default startup stays fast.
        # If model download/load fails (HF rate limiting, missing checkpoint,
        # etc.) we log a warning and continue — the rest of the routers stay
        # available rather than the whole app failing to start.
        try:
            from ..nli import get_nli_scorer
            from .semantic_entropy import SemanticEntropyRouter

            routers[SemanticEntropyRouter.name] = SemanticEntropyRouter(get_nli_scorer(s.nli_model))
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "semantic_entropy router not registered: %s", exc,
            )
    if s.enable_routellm:
        # Heavy import (torch + transformers + HF download); gated on env flag.
        from .routellm_router import RouteLLMRouter

        routers[RouteLLMRouter.name] = RouteLLMRouter(
            kind=s.routellm_kind,
            strong_model=s.routellm_strong_model,
            weak_model=s.routellm_weak_model,
        )
    # auto must be registered last so it can see all the others.
    routers[AutoRouter.name] = AutoRouter(sub_routers=dict(routers))
    return routers


KNOWN_ROUTERS = (
    "self_reported",
    "self_consistency",
    "predictive_entropy",
    "semantic_entropy",
    "routellm",
    "auto",
)
