from __future__ import annotations

import pytest

from med_routing.cache import CompletionCache
from med_routing.cascade import CascadeController
from med_routing.config import cost_usd, get_settings
from med_routing.metrics import ACTUAL_COST_BY_ROUTER, COUNTERFACTUAL_COST_BY_ROUTER
from med_routing.routers.self_consistency import SelfConsistencyRouter
from tests.conftest import FakeOpenAIClient, make_completion


def _value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


def _messages():
    return [{"role": "user", "content": "What is 2+2? A) 3 B) 4 C) 5 D) 6"}]


async def test_kept_weak_counterfactual_is_strong_price_for_same_tokens():
    """When we keep weak, counterfactual = strong-priced version of the weak token bill."""
    s = get_settings()
    client = FakeOpenAIClient()
    weak = make_completion("B", model="gpt-4o-mini")
    client.script(model="gpt-4o-mini", n=1, completions=[weak])
    client.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    routers = {"self_consistency": SelfConsistencyRouter()}
    controller = CascadeController(client=client, routers=routers, cache=CompletionCache(maxsize=64))

    actual_before = _value(ACTUAL_COST_BY_ROUTER, router="self_consistency")
    cf_before = _value(COUNTERFACTUAL_COST_BY_ROUTER, router="self_consistency")

    res = await controller.handle(_messages(), "self_consistency")
    assert res.escalated is False

    actual_after = _value(ACTUAL_COST_BY_ROUTER, router="self_consistency")
    cf_after = _value(COUNTERFACTUAL_COST_BY_ROUTER, router="self_consistency")

    expected_cf = cost_usd(s.strong_model, weak.prompt_tokens, weak.completion_tokens)
    assert cf_after - cf_before == pytest.approx(expected_cf)

    # Counterfactual must always exceed actual when we kept weak (strong is more expensive).
    assert (cf_after - cf_before) > (actual_after - actual_before)


async def test_escalated_counterfactual_uses_strong_call_tokens():
    """When escalated, counterfactual is computed from the strong call (which actually ran)."""
    s = get_settings()
    client = FakeOpenAIClient()
    weak = make_completion("A", model="gpt-4o-mini")
    client.script(model="gpt-4o-mini", n=1, completions=[weak])
    client.script(
        model="gpt-4o-mini",
        n=5,
        completions=[make_completion(t) for t in ["A", "B", "C", "D", "A"]],
    )
    strong = make_completion("D", model="gpt-4o")
    client.script(model="gpt-4o", n=1, completions=[strong])

    routers = {"self_consistency": SelfConsistencyRouter()}
    controller = CascadeController(client=client, routers=routers, cache=CompletionCache(maxsize=64))

    cf_before = _value(COUNTERFACTUAL_COST_BY_ROUTER, router="self_consistency")
    actual_before = _value(ACTUAL_COST_BY_ROUTER, router="self_consistency")

    res = await controller.handle(_messages(), "self_consistency")
    assert res.escalated is True

    cf_delta = _value(COUNTERFACTUAL_COST_BY_ROUTER, router="self_consistency") - cf_before
    actual_delta = _value(ACTUAL_COST_BY_ROUTER, router="self_consistency") - actual_before

    expected_cf = cost_usd(s.strong_model, strong.prompt_tokens, strong.completion_tokens)
    assert cf_delta == pytest.approx(expected_cf)

    # Actual = weak + strong; counterfactual = strong only. So escalations have negative
    # savings on this single request — the router paid extra to fall back. Across many
    # requests, the savings on kept-weak should dominate.
    assert actual_delta > cf_delta
