from __future__ import annotations

import math

import pytest

from med_routing.routers.semantic_entropy import (
    SemanticEntropyRouter,
    cluster_by_entailment,
    cluster_entropy,
)
from tests.conftest import make_completion


class FakeNLI:
    """Two equivalence classes by string membership."""

    def __init__(self, equivalence: dict[str, int]) -> None:
        self.eq = equivalence

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.eq.get(premise) == self.eq.get(hypothesis)


def test_cluster_by_entailment_groups_equivalence_classes():
    samples = ["x1", "x2", "y1", "x3", "y2"]
    eq = {"x1": 0, "x2": 0, "x3": 0, "y1": 1, "y2": 1}
    clusters = cluster_by_entailment(samples, FakeNLI(eq))
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 3]


def test_cluster_entropy_three_two_over_five():
    h = cluster_entropy([[0, 1, 2], [3, 4]], n=5)
    expected = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
    assert h == pytest.approx(expected)


async def test_semantic_entropy_router_three_two_split():
    samples = ["x1", "x2", "y1", "x3", "y2"]
    eq = {"x1": 0, "x2": 0, "x3": 0, "y1": 1, "y2": 1}
    router = SemanticEntropyRouter(FakeNLI(eq))

    async def sampler(*, n, temperature):
        return [make_completion(t) for t in samples[:n]]

    weak = make_completion("x1")
    res = await router.score(messages=[], weak=weak, sampler=sampler)

    expected = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4)) / math.log(5)
    assert res.score == pytest.approx(expected)
    assert sorted(res.extras["clusters"]) == [2, 3]
