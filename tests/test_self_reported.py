from __future__ import annotations

import pytest

from med_routing.routers.self_reported import SelfReportedRouter, parse_confidence
from tests.conftest import FakeOpenAIClient, make_completion


@pytest.mark.parametrize(
    "raw,expected",
    [("85", 85), ("My confidence is 92.", 92), ("about 100%", 100), ("zero", None), ("999", 100)],
)
def test_parse_confidence(raw, expected):
    assert parse_confidence(raw) == expected


async def test_self_reported_score_uses_followup(fake_client: FakeOpenAIClient):
    router = SelfReportedRouter(fake_client)
    weak = make_completion("A")
    fake_client.script(model="gpt-4o-mini", n=1, completions=[make_completion("85")])

    result = await router.score(messages=[{"role": "user", "content": "Q?"}], weak=weak, sampler=None)

    assert result.score == pytest.approx(0.15)
    assert result.extras["confidence"] == 85
    assert fake_client.calls[-1]["temperature"] == 0.0


async def test_self_reported_unparseable_returns_max_uncertainty(fake_client: FakeOpenAIClient):
    router = SelfReportedRouter(fake_client)
    weak = make_completion("A")
    fake_client.script(model="gpt-4o-mini", n=1, completions=[make_completion("not a number")])

    result = await router.score(messages=[{"role": "user", "content": "Q?"}], weak=weak, sampler=None)

    assert result.score == 1.0
    assert result.extras["parsed"] is None
