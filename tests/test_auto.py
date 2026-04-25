from __future__ import annotations

import pytest

from med_routing.cache import CompletionCache
from med_routing.cascade import CascadeController
from med_routing.routers.auto import AutoRouter, detect_format
from med_routing.routers.predictive_entropy import PredictiveEntropyRouter
from med_routing.routers.self_consistency import SelfConsistencyRouter
from med_routing.routers.self_reported import SelfReportedRouter
from tests.conftest import FakeOpenAIClient, make_completion


# ---------- detect_format ----------

@pytest.mark.parametrize("content,expected", [
    ("Question: Foo\nA) one\nB) two\nC) three\nD) four\nReply with a single letter.", "mcq"),
    ("Question: foo\nReply with a single letter (A, B, C, or D).", "mcq"),
    ("What is asthma? Answer in 1-3 sentences.", "free_form"),
    ("List symptoms of pneumonia.", "free_form"),
    ("", "free_form"),
])
def test_detect_format(content, expected):
    msgs = [{"role": "user", "content": content}]
    assert detect_format(msgs) == expected


def test_detect_format_uses_last_user_message():
    msgs = [
        {"role": "user", "content": "Free-form question?"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "A) x B) y C) z D) w\nReply with a single letter."},
    ]
    assert detect_format(msgs) == "mcq"


# ---------- AutoRouter dispatch ----------

class _StubSemanticEntropy:
    """Minimal duck-type to avoid pulling DeBERTa into unit tests. Pretends
    to be the semantic_entropy router so AutoRouter can pick it up."""
    name = "semantic_entropy"
    async def score(self, *, messages, weak, sampler):
        from med_routing.routers.base import RouterScore
        return RouterScore(score=0.5, extras={"stub": True})


def _make_auto(*, with_se=True):
    fake = FakeOpenAIClient()
    sub = {
        "predictive_entropy": PredictiveEntropyRouter(),
        "self_consistency": SelfConsistencyRouter(),
        "self_reported": SelfReportedRouter(fake),
    }
    if with_se:
        sub["semantic_entropy"] = _StubSemanticEntropy()
    return AutoRouter(sub_routers=sub), fake


async def test_auto_delegates_to_predictive_entropy_on_mcq():
    """Cheapest MCQ-suitable router wins for MCQ prompts."""
    auto, fake = _make_auto()
    weak = make_completion("A")
    msgs = [{"role": "user", "content": "Q\nA) x\nB) y\nC) z\nD) w\nSingle letter."}]
    res = await auto.score(messages=msgs, weak=weak, sampler=lambda **kw: [])
    assert res.extras["auto_format"] == "mcq"
    assert res.extras["auto_router"] == "predictive_entropy"


async def test_auto_delegates_to_semantic_entropy_on_free_form():
    auto, fake = _make_auto(with_se=True)
    weak = make_completion("Asthma is a chronic respiratory disease.")
    async def sampler(*, n, temperature):
        return [make_completion(f"answer {i}") for i in range(n)]
    msgs = [{"role": "user", "content": "What is asthma?"}]
    res = await auto.score(messages=msgs, weak=weak, sampler=sampler)
    assert res.extras["auto_format"] == "free_form"
    assert res.extras["auto_router"] == "semantic_entropy"


async def test_auto_falls_back_when_preferred_unavailable():
    """Without semantic_entropy registered, free-form should fall back to
    self_consistency rather than crashing."""
    auto, _ = _make_auto(with_se=False)
    weak = make_completion("free-form answer")
    async def sampler(*, n, temperature):
        return [make_completion(f"variant {i}") for i in range(n)]
    msgs = [{"role": "user", "content": "What is asthma?"}]
    res = await auto.score(messages=msgs, weak=weak, sampler=sampler)
    assert res.extras["auto_format"] == "free_form"
    assert res.extras["auto_router"] == "self_consistency"


async def test_auto_returns_threshold_override_from_subrouter():
    """The cascade should use the picked sub-router's threshold, not auto's own."""
    auto, _ = _make_auto()
    weak = make_completion("A")
    msgs = [{"role": "user", "content": "Q\nA) x\nB) y\nC) z\nD) w\nSingle letter."}]
    res = await auto.score(messages=msgs, weak=weak, sampler=lambda **kw: [])
    # The sub-router picked should be predictive_entropy → its threshold default is 0.40.
    assert res.threshold_override == pytest.approx(0.40)


# ---------- end-to-end through cascade ----------

async def test_cascade_with_auto_uses_subrouter_threshold():
    """Wire AutoRouter into the cascade and verify threshold_override is honored."""
    fake = FakeOpenAIClient()
    weak = make_completion("B", model="gpt-4o-mini")
    fake.script(model="gpt-4o-mini", n=1, completions=[weak])
    fake.script(model="gpt-4o-mini", n=5, completions=[make_completion("B") for _ in range(5)])

    sub = {
        "predictive_entropy": PredictiveEntropyRouter(),
        "self_consistency": SelfConsistencyRouter(),
        "self_reported": SelfReportedRouter(fake),
    }
    auto = AutoRouter(sub_routers=sub)
    routers = {**sub, "auto": auto}
    controller = CascadeController(
        client=fake, routers=routers, cache=CompletionCache(maxsize=64),
    )

    msgs = [{"role": "user", "content": "Question?\nA) 1\nB) 2\nC) 3\nD) 4\nSingle letter."}]
    res = await controller.handle(msgs, "auto")
    # AutoRouter should have picked predictive_entropy on this MCQ prompt.
    assert res.extras["auto_router"] == "predictive_entropy"
    # And the threshold the cascade used must be predictive_entropy's, not auto's.
    assert res.threshold == pytest.approx(0.40)
