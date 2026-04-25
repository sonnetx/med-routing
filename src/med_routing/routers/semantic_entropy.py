from __future__ import annotations

import math

from ..config import get_settings
from ..nli import NLIScorer
from .base import RouterScore, Sampler, UncertaintyRouter


def cluster_by_entailment(samples: list[str], scorer: NLIScorer) -> list[list[int]]:
    """Cluster samples by bidirectional entailment (Kuhn et al. 2023).

    Two responses are equivalent iff each entails the other. Returns lists of
    sample indices grouped by equivalence.
    """
    clusters: list[list[int]] = []
    for i, s in enumerate(samples):
        placed = False
        for cluster in clusters:
            j = cluster[0]
            ref = samples[j]
            if scorer.entails(ref, s) and scorer.entails(s, ref):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def cluster_entropy(clusters: list[list[int]], n: int) -> float:
    """Entropy in nats over cluster size proportions."""
    if n <= 0 or not clusters:
        return 0.0
    return -sum((len(c) / n) * math.log(len(c) / n) for c in clusters if len(c) > 0)


class SemanticEntropyRouter(UncertaintyRouter):
    """Kuhn et al. 2023 — sample N, NLI-cluster, entropy over clusters / log(N)."""

    name = "semantic_entropy"

    def __init__(self, scorer: NLIScorer) -> None:
        self._scorer = scorer

    async def score(self, *, messages, weak, sampler: Sampler) -> RouterScore:
        s = get_settings()
        samples = await sampler(n=s.sample_n, temperature=s.sample_temperature)
        texts = [c.text for c in samples]
        if not texts:
            return RouterScore(score=1.0, extras={"reason": "no_samples"})

        clusters = cluster_by_entailment(texts, self._scorer)
        h = cluster_entropy(clusters, len(texts))
        norm = math.log(len(texts)) if len(texts) > 1 else 1.0
        score = max(0.0, min(1.0, h / norm if norm > 0 else 0.0))

        return RouterScore(
            score=score,
            extras={
                "n": len(texts),
                "clusters": [len(c) for c in clusters],
                "raw_entropy_nats": h,
            },
        )
