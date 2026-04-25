from __future__ import annotations

import math
import re

from ..llm.openai_client import Completion
from .base import RouterScore, Sampler, UncertaintyRouter

_LETTER_TOK = re.compile(r"^\s*[A-D]\s*$", re.IGNORECASE)


def token_entropy(top_logprobs: list[tuple[str, float]]) -> float:
    """Entropy in nats from a top-k list of (token, logprob)."""
    if not top_logprobs:
        return 0.0
    probs = [math.exp(lp) for _, lp in top_logprobs]
    z = sum(probs)
    if z <= 0:
        return 0.0
    probs = [p / z for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


class PredictiveEntropyRouter(UncertaintyRouter):
    """Predictive entropy on the answer-letter token from the weak call's logprobs.

    Falls back to mean per-token entropy if no letter token is found (e.g., the
    model produced a verbose answer without a clean A/B/C/D token).
    Score is normalized by log(k) where k is the number of top alternatives.
    """

    name = "predictive_entropy"

    async def score(self, *, messages, weak: Completion, sampler: Sampler) -> RouterScore:
        if not weak.logprobs:
            return RouterScore(score=1.0, extras={"reason": "no_logprobs"})

        letter_tokens = [t for t in weak.logprobs if _LETTER_TOK.match(t.token)]
        if letter_tokens:
            target = letter_tokens[0]
            entropies = [token_entropy(target.top)]
            k = max(len(target.top), 2)
        else:
            entropies = [token_entropy(t.top) for t in weak.logprobs if t.top]
            k = max(max((len(t.top) for t in weak.logprobs), default=2), 2)

        if not entropies:
            return RouterScore(score=1.0, extras={"reason": "empty_top_logprobs"})

        mean_h = sum(entropies) / len(entropies)
        norm = math.log(k)
        score = max(0.0, min(1.0, mean_h / norm if norm > 0 else 0.0))
        return RouterScore(
            score=score,
            extras={
                "raw_entropy_nats": mean_h,
                "normalizer_log_k": norm,
                "letter_token_used": bool(letter_tokens),
            },
        )
