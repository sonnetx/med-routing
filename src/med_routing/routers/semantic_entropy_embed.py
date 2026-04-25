from __future__ import annotations

import math
from typing import Iterable

from openai import AsyncOpenAI

from ..config import get_settings
from .base import RouterScore, Sampler, UncertaintyRouter


def _cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


def _cluster_by_cosine(
    embeddings: list[list[float]], threshold: float
) -> list[list[int]]:
    """Greedy clustering: a sample joins the first cluster whose representative
    has cosine similarity >= threshold; otherwise it starts a new cluster."""
    clusters: list[list[int]] = []
    for i, emb in enumerate(embeddings):
        placed = False
        for cluster in clusters:
            ref = embeddings[cluster[0]]
            if _cosine(ref, emb) >= threshold:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def _cluster_entropy(clusters: Iterable[list[int]], n: int) -> float:
    if n <= 0:
        return 0.0
    return -sum((len(c) / n) * math.log(len(c) / n) for c in clusters if len(c) > 0)


class SemanticEntropyEmbedRouter(UncertaintyRouter):
    """Embeddings-backed semantic entropy.

    Same shape as Kuhn et al. 2023 but the equivalence relation is cosine
    similarity over OpenAI text embeddings instead of bidirectional NLI
    entailment. Trades a small amount of conceptual fidelity for: no torch
    dependency, no 1.4 GB DeBERTa download, and ~10ms per cluster decision
    instead of ~500ms.

    Score is normalized cluster-entropy in [0,1]; higher = samples disagree
    semantically more, weak model is uncertain, escalate.
    """

    name = "semantic_entropy_embed"

    def __init__(
        self,
        *,
        embed_model: str = "text-embedding-3-small",
        cluster_threshold: float = 0.78,
    ) -> None:
        self._embed_model = embed_model
        self._threshold = cluster_threshold
        self._client = AsyncOpenAI(
            api_key=get_settings().openai_api_key or None,
            base_url=get_settings().openai_base_url,
        )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        # Replace empty strings with a single space; the embeddings API rejects empty input.
        cleaned = [t if t.strip() else " " for t in texts]
        resp = await self._client.embeddings.create(model=self._embed_model, input=cleaned)
        return [item.embedding for item in resp.data]

    async def score(self, *, messages, weak, sampler: Sampler) -> RouterScore:
        s = get_settings()
        samples = await sampler(n=s.sample_n, temperature=s.sample_temperature)
        texts = [c.text for c in samples]
        if not texts:
            return RouterScore(score=1.0, extras={"reason": "no_samples"})

        try:
            embeddings = await self._embed(texts)
        except Exception as exc:
            return RouterScore(score=1.0, extras={"reason": "embed_failed", "err": str(exc)[:200]})

        clusters = _cluster_by_cosine(embeddings, self._threshold)
        h = _cluster_entropy(clusters, len(texts))
        norm = math.log(len(texts)) if len(texts) > 1 else 1.0
        score = max(0.0, min(1.0, h / norm if norm > 0 else 0.0))

        return RouterScore(
            score=score,
            extras={
                "n": len(texts),
                "clusters": [len(c) for c in clusters],
                "raw_entropy_nats": h,
                "embed_model": self._embed_model,
                "cluster_threshold": self._threshold,
            },
        )
