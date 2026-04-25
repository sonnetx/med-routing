from __future__ import annotations

from ..config import get_settings
from ..llm.openai_client import OpenAIClient
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
        from ..nli import get_nli_scorer
        from .semantic_entropy import SemanticEntropyRouter

        routers[SemanticEntropyRouter.name] = SemanticEntropyRouter(get_nli_scorer(s.nli_model))
    return routers


KNOWN_ROUTERS = ("self_reported", "self_consistency", "predictive_entropy", "semantic_entropy")
