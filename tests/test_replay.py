from __future__ import annotations

from pathlib import Path

import pytest

from med_routing.eval.aggregator import EvalAggregator
from med_routing.metrics import (
    ACCURACY,
    ACTUAL_COST_BY_ROUTER,
    COUNTERFACTUAL_COST_BY_ROUTER,
    CROSS_BORDER_TOTAL,
    ESCALATION_RATE,
    PROCESSOR_CALLS_TOTAL,
    REQUESTS_TOTAL,
)
from med_routing.replay import replay_metrics_from_store
from med_routing.store import Store


def _v(counter_or_gauge) -> float:
    return counter_or_gauge._value.get()


def test_aggregator_persist_false_skips_db(tmp_path: Path):
    store = Store(tmp_path / "p.db")
    agg = EvalAggregator(store=store)
    agg.observe(router="r", score=0.1, escalated=False, correct=True, persist=False)
    assert store.query_eval() == []
    # But Prometheus state still updated.
    assert _v(ACCURACY.labels(router="r")) == 1.0


def test_replay_repopulates_request_and_cost_counters(tmp_path: Path):
    store = Store(tmp_path / "x.db")
    base = {
        "ts": "2026-04-25T18:00:00Z", "prompt_sha": "h",
        "score": 0.1, "threshold": 0.4,
        "weak_model": "gpt-4o-mini", "weak_processor": "openai-gpt-4o-mini",
        "weak_region": "US", "weak_dpa_ref": "OpenAI DPA",
        "final_model": "gpt-4o-mini", "final_region": "US",
        "home_region": "US", "regions_touched": ["US"],
        "tokens_prompt": 50, "tokens_completion": 1,
        "cost_usd": 0.0001, "counterfactual_usd": 0.0005, "latency_ms": 800,
    }
    store.insert_decision({**base, "router": "self_consistency", "escalated": False, "cross_border": False})
    store.insert_decision({**base, "router": "self_consistency", "escalated": False, "cross_border": False})
    store.insert_decision({
        **base, "router": "self_consistency", "escalated": True, "cross_border": False,
        "strong_model": "gpt-4o", "strong_processor": "openai-gpt-4o", "strong_region": "US",
        "final_model": "gpt-4o",
    })

    req_before_kept = _v(REQUESTS_TOTAL.labels(router="self_consistency", escalated="false"))
    req_before_esc = _v(REQUESTS_TOTAL.labels(router="self_consistency", escalated="true"))
    actual_before = _v(ACTUAL_COST_BY_ROUTER.labels(router="self_consistency"))
    cf_before = _v(COUNTERFACTUAL_COST_BY_ROUTER.labels(router="self_consistency"))
    proc_before = _v(PROCESSOR_CALLS_TOTAL.labels(
        processor="openai-gpt-4o-mini", entity="OpenAI, Inc.", region="US"))

    agg = EvalAggregator(store=store)
    counts = replay_metrics_from_store(store, agg)
    assert counts == {"decisions": 3, "eval_rows": 0}

    assert _v(REQUESTS_TOTAL.labels(router="self_consistency", escalated="false")) - req_before_kept == 2
    assert _v(REQUESTS_TOTAL.labels(router="self_consistency", escalated="true")) - req_before_esc == 1
    assert _v(ACTUAL_COST_BY_ROUTER.labels(router="self_consistency")) - actual_before == pytest.approx(0.0003)
    assert _v(COUNTERFACTUAL_COST_BY_ROUTER.labels(router="self_consistency")) - cf_before == pytest.approx(0.0015)
    # 3 weak processor calls (one per decision).
    assert _v(PROCESSOR_CALLS_TOTAL.labels(
        processor="openai-gpt-4o-mini", entity="OpenAI, Inc.", region="US"
    )) - proc_before == 3
    # Escalation rate gauge set to 1/3.
    assert _v(ESCALATION_RATE.labels(router="self_consistency")) == pytest.approx(1 / 3)


def test_replay_increments_cross_border_counter(tmp_path: Path):
    """A decision logged with cross_border=true and an EU region under HOME=US
    should bump the foreign-region counter."""
    store = Store(tmp_path / "xb.db")
    store.insert_decision({
        "ts": "2026-04-25T18:00:00Z", "prompt_sha": "h",
        "router": "self_consistency", "score": 0.1, "threshold": 0.4,
        "escalated": False,
        "weak_model": "local-llama", "weak_processor": "local-llama",
        "weak_region": "EU", "weak_dpa_ref": "",
        "final_model": "local-llama", "final_region": "EU",
        "home_region": "US", "regions_touched": ["EU"], "cross_border": True,
        "tokens_prompt": 10, "tokens_completion": 1,
        "cost_usd": 0.0, "counterfactual_usd": 0.0001, "latency_ms": 50,
    })

    before = _v(CROSS_BORDER_TOTAL.labels(
        router="self_consistency", home_region="US", foreign_region="EU"))

    agg = EvalAggregator(store=store)
    replay_metrics_from_store(store, agg)

    after = _v(CROSS_BORDER_TOTAL.labels(
        router="self_consistency", home_region="US", foreign_region="EU"))
    assert after - before == 1


def test_replay_eval_rows_through_aggregator_no_doubling(tmp_path: Path):
    """Replaying eval rows must not re-insert them into the DB."""
    store = Store(tmp_path / "e.db")
    store.insert_eval_row(router="r", score=0.1, escalated=False, correct=True, subject="A")
    store.insert_eval_row(router="r", score=0.6, escalated=True, correct=False, subject="A")
    n_before = len(store.query_eval())
    assert n_before == 2

    agg = EvalAggregator(store=store)
    counts = replay_metrics_from_store(store, agg)
    assert counts["eval_rows"] == 2

    # Replay must NOT have caused new INSERTs.
    assert len(store.query_eval()) == n_before
    # But Prometheus accuracy gauge should reflect 1/2 = 0.5.
    assert _v(ACCURACY.labels(router="r")) == pytest.approx(0.5)
