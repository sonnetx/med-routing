"""Multi-metric grid cascade.

Drives the SSE-streamed `/v1/run` endpoint and the `/` grid UI: at each tier we
compute one main answer + one batch of stochastic samples (shared across every
active router), then score every active router against that tier's signals.
Each router commits independently — a router with a low uncertainty score
locks in at the current tier; routers still "high uncertainty" are forwarded
to the next tier. The grid UI lights one cell per (router, tier).
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, AsyncIterator

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

# Routers that need stochastic samples to score. Used to skip the sample call
# when no active router needs it (e.g. self_reported / predictive_entropy).
SAMPLER_ROUTERS = {"self_consistency", "semantic_entropy", "semantic_entropy_embed"}


class StreamingCascade:
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

    async def _main_call(
        self, messages: list[dict[str, str]], tier: TierSpec
    ) -> tuple[Completion, float]:
        cached = self._cache.get(messages=messages, model=tier.model, temperature=0.0, n=1)
        if cached is not None:
            return cached, 0.0
        t0 = time.perf_counter()
        comps = await self._client.complete(
            model=tier.model,
            messages=messages,
            tier=tier.name,
            temperature=0.0,
            max_tokens=128,
            logprobs=True,
            top_logprobs=5,
        )
        dt_s = time.perf_counter() - t0
        out = comps[0]
        self._cache.set(messages=messages, model=tier.model, temperature=0.0, n=1, value=out)
        return out, dt_s

    async def _samples_call(
        self, messages: list[dict[str, str]], tier: TierSpec
    ) -> tuple[list[Completion], float, float]:
        s = get_settings()
        cached = self._cache.get(
            messages=messages, model=tier.model, temperature=s.sample_temperature, n=s.sample_n
        )
        if cached is not None:
            cost = sum(c.cost for c in cached)
            return cached, cost, 0.0
        t0 = time.perf_counter()
        samples = await self._client.complete(
            model=tier.model,
            messages=messages,
            tier=tier.name,
            temperature=s.sample_temperature,
            max_tokens=128,
            n=s.sample_n,
        )
        dt_s = time.perf_counter() - t0
        self._cache.set(
            messages=messages, model=tier.model, temperature=s.sample_temperature,
            n=s.sample_n, value=samples,
        )
        cost = sum(c.cost for c in samples)
        return samples, cost, dt_s

    def _make_sampler(self, samples: list[Completion]):
        async def sampler(*, n: int, temperature: float) -> list[Completion]:
            # The grid UI fixes (n, temperature) per-tier; routers asking for
            # different (n, temperature) get whatever we already produced. This
            # is OK for the UI demo — the routers all use SAMPLE_N samples at
            # SAMPLE_TEMPERATURE, matching what we precomputed.
            return samples[:n] if len(samples) >= n else samples

        return sampler

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        metrics: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        s = get_settings()
        active = list(metrics) if metrics else [r for r in self._routers if r != "auto"]
        active = [m for m in active if m in self._routers]
        active_set = set(active)
        committed: dict[str, dict[str, Any]] = {}

        # Per-router accumulators for audit-logging at the end.
        per_router_actual: dict[str, float] = {m: 0.0 for m in active}
        per_router_visits: dict[str, list[dict[str, Any]]] = {m: [] for m in active}
        per_router_extras: dict[str, dict[str, Any]] = {m: {} for m in active}
        per_router_score: dict[str, float] = {m: 0.0 for m in active}
        per_router_threshold: dict[str, float] = {m: s.threshold_for(m) for m in active}
        regions_touched: set[str] = set()

        request_t0 = time.perf_counter()
        tiers = s.tiers

        # Emit a one-shot prologue so the UI can render the skeleton.
        yield {
            "type": "run_start",
            "tiers": [{"index": i, "name": t.name, "model": t.model} for i, t in enumerate(tiers)],
            "metrics": active,
            "thresholds": {m: per_router_threshold[m] for m in active},
        }

        for ti, tier in enumerate(tiers):
            if not active_set:
                break
            is_last = ti == len(tiers) - 1
            yield {
                "type": "tier_start",
                "tier_index": ti, "tier_name": tier.name, "model": tier.model,
            }

            main, main_dt = await self._main_call(messages, tier)
            yield {
                "type": "tier_main_done",
                "tier_index": ti, "tier_name": tier.name, "model": tier.model,
                "answer": main.text,
                "input_tokens": main.prompt_tokens,
                "output_tokens": main.completion_tokens,
                "cost": main.cost,
                "latency": main_dt,
            }
            regions_touched.add(get_processor(tier.model).region)

            need_samples = bool(active_set & SAMPLER_ROUTERS)
            samples: list[Completion] = []
            if need_samples:
                samples, scost, sdt = await self._samples_call(messages, tier)
                yield {
                    "type": "tier_samples_done",
                    "tier_index": ti, "tier_name": tier.name, "model": tier.model,
                    "n": len(samples), "cost": scost, "latency": sdt,
                }
            sampler = self._make_sampler(samples)

            for metric in list(active_set):
                router = self._routers[metric]
                try:
                    rs = await router.score(messages=messages, weak=main, sampler=sampler)
                except Exception as exc:
                    yield {
                        "type": "metric_error", "tier_index": ti, "metric": metric,
                        "error": str(exc),
                    }
                    active_set.discard(metric)
                    continue
                threshold = (
                    rs.threshold_override
                    if rs.threshold_override is not None
                    else per_router_threshold[metric]
                )
                escalate = rs.score >= threshold and not is_last

                UNCERTAINTY.labels(router=metric).observe(rs.score)
                per_router_actual[metric] += main.cost
                if metric in SAMPLER_ROUTERS:
                    per_router_actual[metric] += sum(c.cost for c in samples)
                per_router_extras[metric] = rs.extras
                per_router_score[metric] = rs.score
                per_router_threshold[metric] = threshold
                per_router_visits[metric].append({
                    "tier_index": ti, "tier_name": tier.name, "model": tier.model,
                    "score": rs.score, "threshold": threshold, "escalated": escalate,
                    "prompt_tokens": main.prompt_tokens,
                    "completion_tokens": main.completion_tokens,
                })

                yield {
                    "type": "metric_score",
                    "tier_index": ti, "tier_name": tier.name, "model": tier.model,
                    "metric": metric,
                    "score": rs.score, "threshold": threshold,
                    "escalate": escalate, "extras": rs.extras,
                }
                if not escalate:
                    committed[metric] = {
                        "model": tier.model, "tier_index": ti, "tier_name": tier.name,
                        "answer": main.text,
                    }
                    yield {
                        "type": "metric_committed",
                        "metric": metric,
                        "model": tier.model, "tier_index": ti, "tier_name": tier.name,
                        "answer": main.text,
                    }
                    active_set.discard(metric)

        elapsed = time.perf_counter() - request_t0

        # Per-router accounting + audit.
        top_model = tiers[-1].model
        for metric in active:
            visits = per_router_visits[metric]
            if not visits:
                continue
            final = visits[-1]
            final_idx = final["tier_index"]
            escalated_overall = final_idx > 0
            self._bump(metric, escalated_overall)
            REQUESTS_TOTAL.labels(router=metric, escalated=str(escalated_overall).lower()).inc()
            LATENCY_SECONDS.labels(router=metric, tier=final["tier_name"]).observe(elapsed)
            actual = per_router_actual[metric]
            ACTUAL_COST_BY_ROUTER.labels(router=metric).inc(actual)
            # counterfactual = cost of running this same router methodology
            # entirely at the top tier (one main call + N samples if sampler).
            # otherwise sampler routers always look bad: actual includes
            # sample calls, but a 1-call counterfactual would be 6× cheaper.
            ref_visit = visits[0]
            calls_per_visit = (1 + s.sample_n) if metric in SAMPLER_ROUTERS else 1
            counterfactual = calls_per_visit * cost_usd(
                top_model,
                ref_visit["prompt_tokens"],
                ref_visit["completion_tokens"],
            )
            COUNTERFACTUAL_COST_BY_ROUTER.labels(router=metric).inc(counterfactual)

        for region in regions_touched:
            if is_cross_border(region, s.home_region):
                # `router` label is meaningless for a multi-router grid; tag with
                # the literal "stream" so callers can filter it out if they want
                # per-router cardinality. Cardinality stays bounded.
                CROSS_BORDER_TOTAL.labels(
                    router="stream",
                    home_region=s.home_region,
                    foreign_region=region,
                ).inc()

        if self._audit is not None:
            for metric in active:
                visits = per_router_visits[metric]
                if not visits:
                    continue
                final = visits[-1]
                self._audit.log({
                    "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "prompt_sha": hash_messages(messages),
                    "router": metric,
                    "score": per_router_score[metric],
                    "threshold": per_router_threshold[metric],
                    "escalated": final["tier_index"] > 0,
                    "tier_chain": [
                        {
                            "tier_index": v["tier_index"],
                            "tier_name": v["tier_name"],
                            "model": v["model"],
                            "processor": get_processor(v["model"]).name,
                            "entity": get_processor(v["model"]).entity,
                            "region": get_processor(v["model"]).region,
                            "dpa_ref": get_processor(v["model"]).dpa_ref,
                            "score": v["score"],
                            "threshold": v["threshold"],
                            "escalated": v["escalated"],
                        }
                        for v in visits
                    ],
                    "final_model": final["model"],
                    "final_tier_index": final["tier_index"],
                    "final_region": get_processor(final["model"]).region,
                    "regions_touched": sorted(regions_touched),
                    "home_region": s.home_region,
                    "cross_border": any(is_cross_border(r, s.home_region) for r in regions_touched),
                    "cost_usd": per_router_actual[metric],
                    "counterfactual_usd": (
                        ((1 + s.sample_n) if metric in SAMPLER_ROUTERS else 1)
                        * cost_usd(
                            top_model,
                            visits[0]["prompt_tokens"],
                            visits[0]["completion_tokens"],
                        )
                    ),
                    "latency_ms": int(elapsed * 1000),
                })

        yield {
            "type": "all_done",
            "elapsed": elapsed,
            "committed": committed,
        }
