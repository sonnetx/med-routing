from __future__ import annotations

import pytest

from med_routing.routers.self_consistency import SelfConsistencyRouter, extract_letter
from tests.conftest import make_completion


@pytest.mark.parametrize(
    "raw,expected",
    [("A", "A"), ("The answer is C.", "C"), ("D)", "D"), ("none", None), ("z", None)],
)
def test_extract_letter(raw, expected):
    assert extract_letter(raw) == expected


async def test_self_consistency_three_of_five():
    router = SelfConsistencyRouter()
    weak = make_completion("A")
    samples = [make_completion(t) for t in ["A", "A", "A", "B", "C"]]

    async def sampler(*, n, temperature):
        return samples[:n]

    res = await router.score(messages=[], weak=weak, sampler=sampler)

    assert res.score == pytest.approx(0.4)
    assert res.extras["modal"] == "A"
    assert res.extras["modal_count"] == 3


async def test_self_consistency_unanimous_zero_uncertainty():
    router = SelfConsistencyRouter()
    weak = make_completion("B")
    samples = [make_completion("B")] * 5

    async def sampler(*, n, temperature):
        return samples[:n]

    res = await router.score(messages=[], weak=weak, sampler=sampler)
    assert res.score == pytest.approx(0.0)
