from __future__ import annotations

import pytest

from med_routing.routers.routellm_router import RouteLLMRouter, _last_user_prompt
from tests.conftest import make_completion


class FakeController:
    """Stand-in for routellm.controller.Controller; records calls."""

    def __init__(self, win_rate: float = 0.7) -> None:
        self.win_rate = win_rate
        self.calls: list[dict] = []

    def batch_calculate_win_rate(self, *, prompts, router):
        self.calls.append({"prompts": list(prompts), "router": router})
        return [self.win_rate] * len(list(prompts))


def test_last_user_prompt_picks_last_user_message():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert _last_user_prompt(msgs) == "second"


def test_last_user_prompt_empty_when_no_user():
    assert _last_user_prompt([{"role": "system", "content": "x"}]) == ""


async def test_routellm_score_maps_win_rate_to_uncertainty():
    ctrl = FakeController(win_rate=0.7)
    router = RouteLLMRouter(kind="mf", controller=ctrl)
    weak = make_completion("A")
    msgs = [{"role": "user", "content": "What is the diagnosis?"}]

    result = await router.score(messages=msgs, weak=weak, sampler=None)

    assert result.score == pytest.approx(0.7)
    assert result.extras["kind"] == "mf"
    assert ctrl.calls[0]["router"] == "mf"
    assert ctrl.calls[0]["prompts"] == ["What is the diagnosis?"]


async def test_routellm_score_clamped_to_unit_interval():
    ctrl = FakeController(win_rate=1.5)  # bogus over-range value
    router = RouteLLMRouter(kind="mf", controller=ctrl)
    result = await router.score(
        messages=[{"role": "user", "content": "Q"}],
        weak=make_completion("A"),
        sampler=None,
    )
    assert result.score == 1.0


async def test_routellm_no_user_prompt_returns_max_uncertainty():
    ctrl = FakeController(win_rate=0.0)
    router = RouteLLMRouter(kind="mf", controller=ctrl)
    result = await router.score(
        messages=[{"role": "system", "content": "x"}],
        weak=make_completion("A"),
        sampler=None,
    )
    assert result.score == 1.0
    assert result.extras["reason"] == "no_user_prompt"
    assert ctrl.calls == []
