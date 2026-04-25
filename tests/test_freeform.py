from __future__ import annotations

import pytest

from med_routing.eval.medquad import _row_to_item
from med_routing.eval.scoring import score_freeform_with_judge
from med_routing.routers.self_consistency import SelfConsistencyRouter, _normalize_for_match
from tests.conftest import FakeOpenAIClient, make_completion


def test_medquad_row_to_item_happy_path():
    row = {
        "id": "q1",
        "question": "  What is asthma?  ",
        "answer": "Asthma is a chronic respiratory disease.",
        "question_type": "information",
    }
    item = _row_to_item(row, idx=0)
    assert item is not None
    assert item.qid == "q1"
    assert item.question == "What is asthma?"  # stripped
    assert item.subject == "information"
    msgs = item.to_messages()
    assert "What is asthma?" in msgs[0]["content"]
    assert "1-3 sentences" in msgs[0]["content"]


def test_medquad_row_skips_empty():
    assert _row_to_item({"question": "", "answer": "x"}, 0) is None
    assert _row_to_item({"question": "x", "answer": ""}, 0) is None


def test_medquad_row_default_subject_when_missing():
    item = _row_to_item({"question": "Q", "answer": "A"}, idx=5)
    assert item is not None
    assert item.subject == "medquad"


@pytest.mark.parametrize("text_a,text_b,same", [
    ("Asthma is a chronic disease.", "asthma is  a chronic disease.", True),  # case + whitespace
    ("Asthma is a chronic disease.", "Asthma is acute.", False),
    ("", "", True),
])
def test_normalize_for_match_clustering_key(text_a, text_b, same):
    assert (_normalize_for_match(text_a) == _normalize_for_match(text_b)) is same


async def test_self_consistency_falls_back_to_string_match_for_free_form():
    """When samples have no MCQ letter, cluster by normalized string."""
    router = SelfConsistencyRouter()
    weak = make_completion("Asthma is a chronic respiratory disease.")
    samples = [
        make_completion("Asthma is a chronic respiratory disease."),
        make_completion("Asthma is a chronic respiratory disease."),
        make_completion("ASTHMA IS A CHRONIC RESPIRATORY DISEASE."),  # same after norm
        make_completion("Asthma is an acute lung problem."),
        make_completion("Asthma can be treated with inhalers."),
    ]
    async def sampler(*, n, temperature):
        return samples[:n]
    res = await router.score(messages=[], weak=weak, sampler=sampler)
    # 3 normalize to the same key, 1 + 1 are unique → modal_count=3, score = 1 - 3/5 = 0.4
    assert res.score == pytest.approx(0.4)
    assert res.extras["mode"] == "string_match"
    assert res.extras["modal_count"] == 3


async def test_score_freeform_judge_correct(fake_client: FakeOpenAIClient):
    fake_client.script(model="gpt-4o", n=1, completions=[make_completion("CORRECT", model="gpt-4o")])
    ok = await score_freeform_with_judge(
        question="What is asthma?",
        predicted="A chronic respiratory illness.",
        reference="A chronic disease of the lungs.",
        judge_client=fake_client,
        judge_model="gpt-4o",
    )
    assert ok is True


async def test_score_freeform_judge_wrong(fake_client: FakeOpenAIClient):
    fake_client.script(model="gpt-4o", n=1, completions=[make_completion("WRONG", model="gpt-4o")])
    ok = await score_freeform_with_judge(
        question="What is asthma?",
        predicted="It's a type of food allergy.",
        reference="A chronic disease of the lungs.",
        judge_client=fake_client,
        judge_model="gpt-4o",
    )
    assert ok is False


async def test_score_freeform_returns_false_on_empty_inputs(fake_client: FakeOpenAIClient):
    """Don't burn judge tokens on an empty prediction or reference."""
    ok = await score_freeform_with_judge(
        question="Q?", predicted="", reference="x",
        judge_client=fake_client, judge_model="gpt-4o",
    )
    assert ok is False
    # No judge call should have been made.
    assert all(c["model"] != "gpt-4o" for c in fake_client.calls)


async def test_score_freeform_returns_false_on_judge_error():
    class Boom:
        async def complete(self, **kw):
            raise RuntimeError("judge unavailable")
    ok = await score_freeform_with_judge(
        question="Q?", predicted="A", reference="B",
        judge_client=Boom(), judge_model="gpt-4o",
    )
    assert ok is False
