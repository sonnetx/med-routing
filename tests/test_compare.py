from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client():
    """FastAPI TestClient with a stubbed cascade — no real OpenAI traffic."""
    from med_routing import server
    from med_routing.audit import AuditLogger
    from med_routing.cache import CompletionCache
    from med_routing.cascade import CascadeController
    from med_routing.eval.aggregator import EvalAggregator
    from med_routing.routers.predictive_entropy import PredictiveEntropyRouter
    from med_routing.routers.self_consistency import SelfConsistencyRouter
    from med_routing.routers.self_reported import SelfReportedRouter
    from tests.conftest import FakeOpenAIClient, make_completion

    fake = FakeOpenAIClient()
    fake.script(model="gpt-4o-mini", n=1, completions=[make_completion("B")])
    fake.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])
    fake.script(model="gpt-4o-mini", n=1, completions=[make_completion("90")])

    routers = {
        "self_consistency": SelfConsistencyRouter(),
        "self_reported": SelfReportedRouter(fake),
        "predictive_entropy": PredictiveEntropyRouter(),
    }
    cache = CompletionCache(maxsize=64)
    audit = AuditLogger(root="audit_test_tmp")
    controller = CascadeController(client=fake, routers=routers, cache=cache, audit=audit)

    @asynccontextmanager
    async def _stub_lifespan(_app):
        _app.state.controller = controller
        _app.state.routers = routers
        _app.state.aggregator = EvalAggregator()
        _app.state.audit = audit
        class _S: pass
        _app.state.store = _S()
        _app.state.medmcqa_pool = None
        yield

    original = server.app.router.lifespan_context
    server.app.router.lifespan_context = _stub_lifespan
    try:
        with TestClient(server.app) as client:
            yield client
    finally:
        server.app.router.lifespan_context = original


def test_compare_returns_one_row_per_router(app_client):
    payload = {"messages": [{"role": "user", "content": "Q? A) 1 B) 2 C) 3 D) 4"}]}
    r = app_client.post("/v1/compare", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    routers = {row["router"] for row in body["results"] if "error" not in row}
    assert routers == {"self_consistency", "self_reported", "predictive_entropy"}
    for row in body["results"]:
        if "error" in row:
            continue
        assert 0.0 <= row["score"] <= 1.0
        assert "threshold" in row
        assert "escalated" in row
        assert "model_used" in row
        assert "latency_ms" in row


def test_compare_filters_to_requested_routers(app_client):
    payload = {
        "messages": [{"role": "user", "content": "Q? A) 1 B) 2 C) 3 D) 4"}],
        "routers": ["self_consistency"],
    }
    r = app_client.post("/v1/compare", json=payload)
    assert r.status_code == 200
    routers = {row["router"] for row in r.json()["results"]}
    assert routers == {"self_consistency"}


def test_compare_rejects_unknown_router(app_client):
    payload = {
        "messages": [{"role": "user", "content": "Q?"}],
        "routers": ["self_consistency", "totally_made_up"],
    }
    r = app_client.post("/v1/compare", json=payload)
    assert r.status_code == 404
