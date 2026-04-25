from __future__ import annotations

import pytest

from med_routing.cache import CompletionCache
from med_routing.cascade import CascadeController
from med_routing.routers.self_consistency import SelfConsistencyRouter
from tests.conftest import FakeOpenAIClient, make_completion


def _messages():
    return [{"role": "user", "content": "What is 2+2? A) 3 B) 4 C) 5 D) 6"}]


async def test_cascade_does_not_escalate_when_unanimous():
    client = FakeOpenAIClient()
    weak = make_completion("B", model="gpt-4o-mini")
    client.script(model="gpt-4o-mini", n=1, completions=[weak])
    client.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    routers = {"self_consistency": SelfConsistencyRouter()}
    controller = CascadeController(client=client, routers=routers, cache=CompletionCache(maxsize=64))

    result = await controller.handle(_messages(), "self_consistency")

    assert result.escalated is False
    assert result.score == pytest.approx(0.0)
    assert result.text == "B"
    assert all(call["model"] != "gpt-4o" for call in client.calls)


async def test_cascade_escalates_on_disagreement():
    client = FakeOpenAIClient()
    weak = make_completion("A", model="gpt-4o-mini")
    client.script(model="gpt-4o-mini", n=1, completions=[weak])
    client.script(
        model="gpt-4o-mini",
        n=5,
        completions=[make_completion(t) for t in ["A", "B", "C", "D", "A"]],
    )
    client.script(model="gpt-4o", n=1, completions=[make_completion("D", model="gpt-4o")])

    routers = {"self_consistency": SelfConsistencyRouter()}
    controller = CascadeController(client=client, routers=routers, cache=CompletionCache(maxsize=64))

    result = await controller.handle(_messages(), "self_consistency")

    assert result.escalated is True
    assert result.score >= result.threshold
    assert result.text == "D"
    assert result.model_used == "gpt-4o"
    assert any(call["model"] == "gpt-4o" for call in client.calls)


async def test_cascade_caches_weak_call():
    client = FakeOpenAIClient()
    weak = make_completion("B", model="gpt-4o-mini")
    client.script(model="gpt-4o-mini", n=1, completions=[weak])
    client.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    cache = CompletionCache(maxsize=64)
    routers = {"self_consistency": SelfConsistencyRouter()}
    controller = CascadeController(client=client, routers=routers, cache=cache)

    await controller.handle(_messages(), "self_consistency")
    assert len(cache) >= 1
    n_calls_first = len(client.calls)

    # Second handle for same prompt should hit caches and avoid extra weak n=1 + n=5 calls.
    client.script(model="gpt-4o-mini", n=1, completions=[weak])  # would be popped only on miss
    await controller.handle(_messages(), "self_consistency")
    assert len(client.calls) == n_calls_first  # zero new model calls
