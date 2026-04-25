from __future__ import annotations

import math

import pytest

from med_routing.llm.openai_client import TokenLogprob
from med_routing.routers.predictive_entropy import PredictiveEntropyRouter, token_entropy
from tests.conftest import make_completion


def _uniform(k: int) -> list[tuple[str, float]]:
    p = 1.0 / k
    return [(f"t{i}", math.log(p)) for i in range(k)]


def _onehot(k: int) -> list[tuple[str, float]]:
    out = [(f"t{i}", math.log(1e-9)) for i in range(k)]
    out[0] = ("t0", math.log(1.0))
    return out


def test_token_entropy_uniform_is_log_k():
    assert token_entropy(_uniform(5)) == pytest.approx(math.log(5), rel=1e-3)


def test_token_entropy_onehot_is_zero():
    assert token_entropy(_onehot(5)) == pytest.approx(0.0, abs=1e-3)


async def test_predictive_entropy_letter_token_uniform_scores_one():
    weak = make_completion(
        "A",
        logprobs=[TokenLogprob(token="A", logprob=-1.6, top=_uniform(5))],
    )
    res = await PredictiveEntropyRouter().score(messages=[], weak=weak, sampler=None)
    assert res.score == pytest.approx(1.0, abs=1e-3)
    assert res.extras["letter_token_used"] is True


async def test_predictive_entropy_letter_token_confident_scores_zero():
    weak = make_completion(
        "A",
        logprobs=[TokenLogprob(token="A", logprob=0.0, top=_onehot(5))],
    )
    res = await PredictiveEntropyRouter().score(messages=[], weak=weak, sampler=None)
    assert res.score == pytest.approx(0.0, abs=1e-3)


async def test_predictive_entropy_no_logprobs_returns_max():
    weak = make_completion("A", logprobs=None)
    res = await PredictiveEntropyRouter().score(messages=[], weak=weak, sampler=None)
    assert res.score == 1.0
