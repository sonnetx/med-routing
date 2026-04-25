from __future__ import annotations

from pathlib import Path

import pytest

from med_routing.audit import AuditLogger
from med_routing.cache import CompletionCache
from med_routing.cascade import CascadeController
from med_routing.eval.aggregator import EvalAggregator
from med_routing.routers.self_consistency import SelfConsistencyRouter
from med_routing.store import DECISION_COLS, EVAL_COLS, Store, iter_csv
from tests.conftest import FakeOpenAIClient, make_completion


def test_store_uses_full_synchronous_for_durability(tmp_path: Path):
    """Bind-mounted volumes on Docker Desktop / WSL2 don't honour the default
    NORMAL synchronous reliably, so we force FULL on every connection."""
    store = Store(tmp_path / "x.db")
    with store._connect() as c:
        # synchronous = 2 = FULL, see https://www.sqlite.org/pragma.html#pragma_synchronous
        sync = c.execute("PRAGMA synchronous").fetchone()[0]
        journal = c.execute("PRAGMA journal_mode").fetchone()[0]
    assert sync == 2, f"expected synchronous=FULL (2), got {sync}"
    assert journal.lower() == "wal"


def test_store_inserts_and_queries_decision(tmp_path: Path):
    store = Store(tmp_path / "x.db")
    row = {
        "ts": "2026-04-25T17:00:00Z",
        "prompt_sha": "deadbeef" * 5,
        "router": "self_consistency",
        "score": 0.4,
        "threshold": 0.4,
        "escalated": True,
        "weak_model": "gpt-4o-mini",
        "weak_processor": "openai-gpt-4o-mini",
        "weak_region": "US",
        "strong_model": "gpt-4o",
        "strong_processor": "openai-gpt-4o",
        "strong_region": "US",
        "final_model": "gpt-4o",
        "final_region": "US",
        "home_region": "US",
        "regions_touched": ["US"],
        "cross_border": False,
        "tokens_prompt": 100,
        "tokens_completion": 5,
        "cost_usd": 0.0001,
        "counterfactual_usd": 0.0002,
        "latency_ms": 540,
    }
    store.insert_decision(row)
    rows = store.query_decisions(limit=10)
    assert len(rows) == 1
    assert rows[0]["router"] == "self_consistency"
    assert rows[0]["regions_touched"] == ["US"]
    assert rows[0]["cross_border"] is False  # round-trips as bool


def test_store_filters_by_router_and_cross_border(tmp_path: Path):
    store = Store(tmp_path / "y.db")
    base = {
        "ts": "2026-04-25T17:00:00Z", "prompt_sha": "x", "score": 0.1, "threshold": 0.4,
        "weak_model": "w", "final_model": "w", "home_region": "US", "regions_touched": ["US"],
        "tokens_prompt": 1, "tokens_completion": 1, "cost_usd": 0.0, "counterfactual_usd": 0.0,
        "latency_ms": 1,
    }
    store.insert_decision({**base, "router": "a", "escalated": False, "cross_border": False})
    store.insert_decision({**base, "router": "b", "escalated": False, "cross_border": True, "regions_touched": ["EU"]})
    store.insert_decision({**base, "router": "a", "escalated": True, "cross_border": True, "regions_touched": ["EU"]})

    assert len(store.query_decisions(router="a")) == 2
    assert len(store.query_decisions(router="b")) == 1
    assert len(store.query_decisions(cross_border_only=True)) == 2


def test_store_inserts_eval_row(tmp_path: Path):
    store = Store(tmp_path / "z.db")
    store.insert_eval_row(router="r", score=0.2, escalated=False, correct=True, subject="Anatomy", qid="q1")
    rows = store.query_eval(limit=10)
    assert len(rows) == 1
    assert rows[0]["router"] == "r"
    assert rows[0]["correct"] == 1


def test_iter_csv_quotes_and_escapes(tmp_path: Path):
    rows = [
        {"ts": "2026-04-25", "router": "a, with comma", "score": 0.5},
        {"ts": "2026-04-26", "router": 'b"quoted"', "score": None},
    ]
    out = "".join(iter_csv(rows, ("ts", "router", "score")))
    assert out.splitlines()[0] == "ts,router,score"
    assert '"a, with comma"' in out
    assert '"b""quoted"""' in out
    assert ",\n" in out  # None becomes empty cell


def test_audit_logger_persists_to_store(tmp_path: Path):
    store = Store(tmp_path / "audit.db")
    audit = AuditLogger(root=tmp_path / "audit", store=store)
    audit.log({
        "ts": "x", "prompt_sha": "y", "router": "r", "score": 0.1, "threshold": 0.4,
        "escalated": False, "weak_model": "m", "final_model": "m", "home_region": "US",
        "regions_touched": ["US"], "cross_border": False,
        "tokens_prompt": 1, "tokens_completion": 1,
        "cost_usd": 0.0, "counterfactual_usd": 0.0, "latency_ms": 1,
    })
    assert len(store.query_decisions()) == 1
    audit.close()


def test_aggregator_persists_eval_rows(tmp_path: Path):
    store = Store(tmp_path / "eval.db")
    agg = EvalAggregator(store=store)
    agg.observe(router="r", score=0.2, escalated=False, correct=True, subject="X")
    agg.observe(router="r", score=0.6, escalated=True, correct=False, subject="X")
    rows = store.query_eval()
    assert len(rows) == 2
    assert {r["correct"] for r in rows} == {0, 1}


async def test_full_cascade_writes_to_store(tmp_path: Path):
    """End-to-end: a single cascade.handle() call lands a row in the SQLite store."""
    store = Store(tmp_path / "e2e.db")
    audit = AuditLogger(root=tmp_path / "audit", store=store)

    client = FakeOpenAIClient()
    client.script(model="gpt-4o-mini", n=1, completions=[make_completion("B")])
    client.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    controller = CascadeController(
        client=client,
        routers={"self_consistency": SelfConsistencyRouter()},
        cache=CompletionCache(maxsize=64),
        audit=audit,
    )
    await controller.handle([{"role": "user", "content": "Q"}], "self_consistency")

    rows = store.query_decisions()
    assert len(rows) == 1
    r = rows[0]
    assert r["router"] == "self_consistency"
    assert r["weak_processor"] == "openai-gpt-4o-mini"
    assert r["regions_touched"] == ["US"]

    stats = store.stats()
    assert stats["decisions"] == 1
    assert stats["decisions_by_router"]["self_consistency"] == 1
    audit.close()
