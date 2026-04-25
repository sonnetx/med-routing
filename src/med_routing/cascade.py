from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditLogger
from .cache import CompletionCache, hash_messages
from .config import TierSpec, cost_usd, get_settings
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
class TierVisit:
    """One tier's contribution to a single-router cascade decision.

    Each visited tier produces a completion + (optionally) an uncertainty score.
    The last visited tier always commits — its score may still be high if every
    earlier tier said "escalate", but we have nowhere left to go.
    """
    tier_index: int
    tier_name: str
    model: str
    completion: Completion
    score: float | None
    threshold: float | None
    escalated: bool


@dataclass
class CascadeResult:
    text: str
    model_used: str
    router: str
    score: float
    threshold: float
    escalated: bool
    final_tier_index: int
    visits: list[TierVisit] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    # Back-compat shims for any callers still expecting weak/strong handles.
    @property
    def weak_completion(self) -> Completion:
        return self.visits[0].completion

    @property
    def strong_completion(self) -> Completion | None:
        if len(self.visits) <= 1:
            return None
        return self.visits[-1].completion


class CascadeController:
    """3-tier per-router cascade.

    For each request the controller walks `settings.tiers` from cheapest to
    most expensive: at each tier it produces a completion, asks the router for
    an uncertainty score, and either commits (score below threshold) or
    escalates to the next tier. The final tier always commits.
    """

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

    async def _main_with_logprobs(
        self, messages: list[dict[str, str]], *, tier: TierSpec
    ) -> Completion:
        cached = self._cache.get(messages=messages, model=tier.model, temperature=0.0, n=1)
        if cached is not None:
            return cached
        completions = await self._client.complete(
            model=tier.model,
            messages=messages,
            tier=tier.name,
            temperature=0.0,
            max_tokens=64,
            logprobs=True,
            top_logprobs=5,
        )
        out = completions[0]
        self._cache.set(messages=messages, model=tier.model, temperature=0.0, n=1, value=out)
        return out

    def _make_sampler(self, messages: list[dict[str, str]], *, tier: TierSpec):
        async def sampler(*, n: int, temperature: float) -> list[Completion]:
            cached = self._cache.get(messages=messages, model=tier.model, temperature=temperature, n=n)
            if cached is not None:
                return cached
            samples = await self._client.complete(
                model=tier.model,
                messages=messages,
                tier=tier.name,
                temperature=temperature,
                max_tokens=64,
                n=n,
            )
            self._cache.set(messages=messages, model=tier.model, temperature=temperature, n=n, value=samples)
            return samples

        return sampler

    async def handle(self, messages: list[dict[str, str]], router_name: str) -> CascadeResult:
        s = get_settings()
        if router_name not in self._routers:
            raise ValueError(f"Unknown router: {router_name!r}; have {sorted(self._routers)}")
        router = self._routers[router_name]
        threshold = s.threshold_for(router_name)

        tiers = s.tiers
        visits: list[TierVisit] = []
        regions: set[str] = set()
        total_actual = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_extras: dict[str, Any] = {}
        last_score: float = 0.0
        last_threshold: float = threshold
        final_tier_index = 0
        committed = False

        t0 = time.perf_counter()
        for ti, tier in enumerate(tiers):
            is_last = ti == len(tiers) - 1
            comp = await self._main_with_logprobs(messages, tier=tier)
            sampler = self._make_sampler(messages, tier=tier)
            rs = await router.score(messages=messages, weak=comp, sampler=sampler)
            cur_threshold = rs.threshold_override if rs.threshold_override is not None else threshold

            UNCERTAINTY.labels(router=router_name).observe(rs.score)
            escalate = rs.score >= cur_threshold and not is_last

            visits.append(TierVisit(
                tier_index=ti, tier_name=tier.name, model=tier.model,
                completion=comp, score=rs.score, threshold=cur_threshold, escalated=escalate,
            ))
            regions.add(get_processor(tier.model).region)
            total_actual += comp.cost
            total_prompt_tokens += comp.prompt_tokens
            total_completion_tokens += comp.completion_tokens
            last_extras = rs.extras
            last_score = rs.score
            last_threshold = cur_threshold
            final_tier_index = ti

            if not escalate:
                committed = True
                break

        elapsed = time.perf_counter() - t0
        final_visit = visits[final_tier_index]
        escalated_overall = final_tier_index > 0
        text = final_visit.completion.text
        model_used = final_visit.model

        REQUESTS_TOTAL.labels(router=router_name, escalated=str(escalated_overall).lower()).inc()
        LATENCY_SECONDS.labels(router=router_name, tier=final_visit.tier_name).observe(elapsed)
        self._bump(router_name, escalated_overall)

        # Counterfactual = top-tier model on this prompt. Use its actual token
        # counts if we visited it; else estimate from the cheapest call we made.
        top_tier = tiers[-1]
        if final_tier_index == len(tiers) - 1:
            counterfactual = visits[-1].completion.cost
        else:
            ref = visits[0].completion
            counterfactual = cost_usd(top_tier.model, ref.prompt_tokens, ref.completion_tokens)
        ACTUAL_COST_BY_ROUTER.labels(router=router_name).inc(total_actual)
        COUNTERFACTUAL_COST_BY_ROUTER.labels(router=router_name).inc(counterfactual)

        for region in regions:
            if is_cross_border(region, s.home_region):
                CROSS_BORDER_TOTAL.labels(
                    router=router_name,
                    home_region=s.home_region,
                    foreign_region=region,
                ).inc()

        if self._audit is not None:
            tier_chain = []
            for v in visits:
                proc = get_processor(v.model)
                tier_chain.append({
                    "tier_index": v.tier_index,
                    "tier_name": v.tier_name,
                    "model": v.model,
                    "processor": proc.name,
                    "entity": proc.entity,
                    "region": proc.region,
                    "dpa_ref": proc.dpa_ref,
                    "score": v.score,
                    "threshold": v.threshold,
                    "escalated": v.escalated,
                    "prompt_tokens": v.completion.prompt_tokens,
                    "completion_tokens": v.completion.completion_tokens,
                    "cost_usd": v.completion.cost,
                })
            final_proc = get_processor(model_used)
            self._audit.log({
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "prompt_sha": hash_messages(messages),
                "router": router_name,
                "score": last_score,
                "threshold": last_threshold,
                "escalated": escalated_overall,
                "tier_chain": tier_chain,
                "final_model": model_used,
                "final_tier_index": final_tier_index,
                "final_region": final_proc.region,
                "regions_touched": sorted(regions),
                "home_region": s.home_region,
                "cross_border": any(is_cross_border(r, s.home_region) for r in regions),
                "tokens_prompt": total_prompt_tokens,
                "tokens_completion": total_completion_tokens,
                "cost_usd": total_actual,
                "counterfactual_usd": counterfactual,
                "latency_ms": int(elapsed * 1000),
            })

        return CascadeResult(
            text=text,
            model_used=model_used,
            router=router_name,
            score=last_score,
            threshold=last_threshold,
            escalated=escalated_overall,
            final_tier_index=final_tier_index,
            visits=visits,
            extras=last_extras,
        )
