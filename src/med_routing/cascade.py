from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .cache import CompletionCache
from .config import get_settings
from .llm.openai_client import Completion, OpenAIClient
from .metrics import (
    ESCALATION_RATE,
    LATENCY_SECONDS,
    REQUESTS_TOTAL,
    UNCERTAINTY,
)
from .routers.base import UncertaintyRouter


@dataclass
class CascadeResult:
    text: str
    model_used: str
    router: str
    score: float
    threshold: float
    escalated: bool
    weak_completion: Completion
    strong_completion: Completion | None
    extras: dict[str, Any]


class CascadeController:
    """Runs the weak model, scores uncertainty, escalates if needed, attaches metadata."""

    def __init__(
        self,
        *,
        client: OpenAIClient,
        routers: dict[str, UncertaintyRouter],
        cache: CompletionCache,
    ) -> None:
        self._client = client
        self._routers = routers
        self._cache = cache
        self._counts: dict[str, dict[str, int]] = {}

    def _bump(self, router: str, escalated: bool) -> None:
        bucket = self._counts.setdefault(router, {"total": 0, "escalated": 0})
        bucket["total"] += 1
        if escalated:
            bucket["escalated"] += 1
        if bucket["total"]:
            ESCALATION_RATE.labels(router=router).set(bucket["escalated"] / bucket["total"])

    async def _weak_with_logprobs(
        self, messages: list[dict[str, str]], *, model: str
    ) -> Completion:
        cached = self._cache.get(messages=messages, model=model, temperature=0.0, n=1)
        if cached is not None:
            return cached
        completions = await self._client.complete(
            model=model,
            messages=messages,
            tier="weak",
            temperature=0.0,
            max_tokens=64,
            logprobs=True,
            top_logprobs=5,
        )
        weak = completions[0]
        self._cache.set(messages=messages, model=model, temperature=0.0, n=1, value=weak)
        return weak

    async def _make_sampler(self, messages: list[dict[str, str]], *, model: str):
        async def sampler(*, n: int, temperature: float) -> list[Completion]:
            cached = self._cache.get(messages=messages, model=model, temperature=temperature, n=n)
            if cached is not None:
                return cached
            samples = await self._client.complete(
                model=model,
                messages=messages,
                tier="weak",
                temperature=temperature,
                max_tokens=64,
                n=n,
            )
            self._cache.set(messages=messages, model=model, temperature=temperature, n=n, value=samples)
            return samples

        return sampler

    async def handle(self, messages: list[dict[str, str]], router_name: str) -> CascadeResult:
        s = get_settings()
        if router_name not in self._routers:
            raise ValueError(f"Unknown router: {router_name!r}; have {sorted(self._routers)}")
        router = self._routers[router_name]
        threshold = s.threshold_for(router_name)

        t0 = time.perf_counter()
        weak = await self._weak_with_logprobs(messages, model=s.weak_model)
        sampler = await self._make_sampler(messages, model=s.weak_model)
        rs = await router.score(messages=messages, weak=weak, sampler=sampler)

        UNCERTAINTY.labels(router=router_name).observe(rs.score)
        escalated = rs.score >= threshold

        strong: Completion | None = None
        if escalated:
            results = await self._client.complete(
                model=s.strong_model,
                messages=messages,
                tier="strong",
                temperature=0.0,
                max_tokens=64,
            )
            strong = results[0]
            tier_for_latency = "strong"
            text = strong.text
            model_used = s.strong_model
        else:
            tier_for_latency = "weak"
            text = weak.text
            model_used = s.weak_model

        REQUESTS_TOTAL.labels(router=router_name, escalated=str(escalated).lower()).inc()
        LATENCY_SECONDS.labels(router=router_name, tier=tier_for_latency).observe(time.perf_counter() - t0)
        self._bump(router_name, escalated)

        return CascadeResult(
            text=text,
            model_used=model_used,
            router=router_name,
            score=rs.score,
            threshold=threshold,
            escalated=escalated,
            weak_completion=weak,
            strong_completion=strong,
            extras=rs.extras,
        )
