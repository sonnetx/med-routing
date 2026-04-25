from __future__ import annotations

from collections import defaultdict
from typing import Any

from .eval.aggregator import EvalAggregator
from .metrics import (
    ACTUAL_COST_BY_ROUTER,
    COST_USD_TOTAL,
    COUNTERFACTUAL_COST_BY_ROUTER,
    CROSS_BORDER_TOTAL,
    ESCALATION_RATE,
    LATENCY_SECONDS,
    PROCESSOR_CALLS_TOTAL,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
    UNCERTAINTY,
)
from .processors import get_processor


def replay_metrics_from_store(
    store: Any,
    aggregator: EvalAggregator,
    audit: Any | None = None,
) -> dict[str, int]:
    """Re-populate Prometheus metrics from SQLite at startup.

    Why this exists: Prometheus client counters live in the FastAPI process
    memory, so a container restart resets them to zero. The DB persists across
    restarts, but unless we replay it the dashboard panels look like the dataset
    shrank. This loops the persisted decisions + eval_rows once at startup and
    bumps the in-process metrics so cumulative state is preserved.

    Idempotent only at the call-site granularity (call once at lifespan startup).
    Calling it twice will double-count.

    Returns a small dict of replay counts for logging.
    """
    decisions = store.query_decisions(limit=1_000_000)
    eval_rows = store.query_eval(limit=1_000_000)

    counts_per_router: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "escalated": 0})

    for d in decisions:
        router = d.get("router") or "unknown"
        escalated = bool(d.get("escalated"))
        score = float(d.get("score") or 0.0)

        REQUESTS_TOTAL.labels(router=router, escalated=str(escalated).lower()).inc()
        UNCERTAINTY.labels(router=router).observe(score)

        actual_cost = d.get("cost_usd")
        if actual_cost:
            ACTUAL_COST_BY_ROUTER.labels(router=router).inc(actual_cost)
        cf_cost = d.get("counterfactual_usd")
        if cf_cost:
            COUNTERFACTUAL_COST_BY_ROUTER.labels(router=router).inc(cf_cost)

        latency_ms = d.get("latency_ms")
        if latency_ms:
            tier = "strong" if escalated else "weak"
            LATENCY_SECONDS.labels(router=router, tier=tier).observe(latency_ms / 1000.0)

        weak_proc = d.get("weak_processor")
        weak_region = d.get("weak_region") or "unknown"
        weak_model = d.get("weak_model") or ""
        if weak_proc:
            entity = get_processor(weak_model).entity if weak_model else "unknown"
            PROCESSOR_CALLS_TOTAL.labels(processor=weak_proc, entity=entity, region=weak_region).inc()

        if escalated and d.get("strong_processor"):
            strong_model = d.get("strong_model") or ""
            entity = get_processor(strong_model).entity if strong_model else "unknown"
            PROCESSOR_CALLS_TOTAL.labels(
                processor=d["strong_processor"],
                entity=entity,
                region=d.get("strong_region") or "unknown",
            ).inc()

        if d.get("cross_border"):
            home = d.get("home_region") or "?"
            for region in (d.get("regions_touched") or []):
                if region != home:
                    CROSS_BORDER_TOTAL.labels(
                        router=router, home_region=home, foreign_region=region,
                    ).inc()

        # Tokens / per-model cost: attribute to the model that actually
        # produced the final answer. This is approximate when escalated (the
        # weak call's tokens leak into gpt-4o's bucket) but the per-router
        # ACTUAL_COST counters above are exact.
        final_model = d.get("final_model") or weak_model
        if final_model:
            if d.get("tokens_prompt"):
                TOKENS_TOTAL.labels(model=final_model, kind="prompt").inc(d["tokens_prompt"])
            if d.get("tokens_completion"):
                TOKENS_TOTAL.labels(model=final_model, kind="completion").inc(d["tokens_completion"])
            if actual_cost:
                COST_USD_TOTAL.labels(model=final_model).inc(actual_cost)

        bucket = counts_per_router[router]
        bucket["total"] += 1
        bucket["escalated"] += int(escalated)

    for router, c in counts_per_router.items():
        if c["total"]:
            ESCALATION_RATE.labels(router=router).set(c["escalated"] / c["total"])

    for e in eval_rows:
        aggregator.observe(
            router=e.get("router") or "unknown",
            score=float(e.get("score") or 0.0),
            escalated=bool(e.get("escalated")),
            correct=bool(e.get("correct")),
            subject=e.get("subject") or None,
            persist=False,
        )

    # Pre-populate the audit-drawer ring buffer with the most recent decisions
    # (in the order they were originally appended) so the frontend doesn't
    # render "no decisions yet" until the next live request lands.
    if audit is not None:
        audit.restore_recent(list(reversed(decisions)))

    return {"decisions": len(decisions), "eval_rows": len(eval_rows)}
