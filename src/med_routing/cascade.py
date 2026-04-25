from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

from .audit import AuditLogger
from .cache import CompletionCache, hash_messages
from .config import cost_usd, get_settings
from .llm.openai_client import Completion, OpenAIClient
from .metrics import (
    ACTUAL_COST_BY_ROUTER,
    COUNTERFACTUAL_COST_BY_ROUTER,
    CROSS_BORDER_TOTAL,
    ESCALATION_RATE,
    LATENCY_SECONDS,
    REQUESTS_TOTAL,
    UNCERTAINTY,
)
from .processors import get_processor, is_cross_border
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
        audit: AuditLogger | None = None,
    ) -> None:
        self._client = client
        self._routers = routers
        self._cache = cache
        self._audit = audit
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

        # Cost accounting attributed to the router. Actual = weak + any strong escalation.
        # Counterfactual = what we would have paid had we sent the prompt straight to the
        # strong model. We use the strong call's token counts when we have them; otherwise
        # we proxy with the weak call's token counts (close enough — same prompt, similar
        # answer length for MCQ tasks).
        actual = weak.cost + (strong.cost if strong is not None else 0.0)
        if strong is not None:
            counterfactual = cost_usd(s.strong_model, strong.prompt_tokens, strong.completion_tokens)
        else:
            counterfactual = cost_usd(s.strong_model, weak.prompt_tokens, weak.completion_tokens)
        ACTUAL_COST_BY_ROUTER.labels(router=router_name).inc(actual)
        COUNTERFACTUAL_COST_BY_ROUTER.labels(router=router_name).inc(counterfactual)

        # Data residency: identify which processors handled this request and flag
        # any region that is not the configured home region.
        weak_proc = get_processor(s.weak_model)
        final_proc = get_processor(s.strong_model if escalated else s.weak_model)
        regions = {weak_proc.region}
        if escalated:
            regions.add(final_proc.region)
        for region in regions:
            if is_cross_border(region, s.home_region):
                CROSS_BORDER_TOTAL.labels(
                    router=router_name,
                    home_region=s.home_region,
                    foreign_region=region,
                ).inc()

        if self._audit is not None:
            self._audit.log({
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "prompt_sha": hash_messages(messages),
                "router": router_name,
                "score": rs.score,
                "threshold": threshold,
                "escalated": escalated,
                "weak_model": s.weak_model,
                "weak_processor": weak_proc.name,
                "weak_entity": weak_proc.entity,
                "weak_region": weak_proc.region,
                "weak_dpa_ref": weak_proc.dpa_ref,
                "strong_model": s.strong_model if escalated else None,
                "strong_processor": final_proc.name if escalated else None,
                "strong_region": final_proc.region if escalated else None,
                "final_model": model_used,
                "final_region": final_proc.region,
                "regions_touched": sorted(regions),
                "home_region": s.home_region,
                "cross_border": any(is_cross_border(r, s.home_region) for r in regions),
                "tokens_prompt": weak.prompt_tokens + (strong.prompt_tokens if strong else 0),
                "tokens_completion": weak.completion_tokens + (strong.completion_tokens if strong else 0),
                "cost_usd": actual,
                "counterfactual_usd": counterfactual,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            })

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
