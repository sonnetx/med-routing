from __future__ import annotations

import pytest

from med_routing.eval.medmcqa import MedMCQAItem
from med_routing.routers.predictive_entropy import PredictiveEntropyRouter
from med_routing.routers.self_consistency import SelfConsistencyRouter
from med_routing.routers.self_reported import SelfReportedRouter
from med_routing.train.collect import _process_item
from tests.conftest import FakeOpenAIClient, make_completion


def _item(answer: str = "A") -> MedMCQAItem:
    return MedMCQAItem(
        qid="q1",
        question="What is the diagnosis?",
        options=("apple", "banana", "cherry", "date"),
        answer=answer,
        subject="Medicine",
    )


async def test_process_item_emits_row_with_all_signals(fake_client: FakeOpenAIClient):
    # Weak call: returns "A"; sampled batch (n=5): 3×A, 1×B, 1×C; self-report follow-up: "80".
    fake_client.script(
        model="gpt-4o-mini",
        n=1,
        completions=[make_completion("A")],  # weak call
    )
    fake_client.script(
        model="gpt-4o-mini",
        n=5,
        completions=[make_completion(t) for t in ["A", "A", "A", "B", "C"]],
    )
    fake_client.script(
        model="gpt-4o-mini",
        n=1,
        completions=[make_completion("80")],  # self-report follow-up
    )

    row = await _process_item(
        _item(answer="A"),
        client=fake_client,
        pe_router=PredictiveEntropyRouter(),
        sc_router=SelfConsistencyRouter(),
        sr_router=SelfReportedRouter(fake_client),
        se_router=None,
        with_strong=False,
    )

    assert row is not None
    assert row["qid"] == "q1"
    assert row["weak_letter"] == "A"
    assert row["weak_correct"] is True
    # 3-of-5 modal => 1 - 3/5 = 0.4
    assert row["self_consistency"] == pytest.approx(0.4)
    # self_reported = 1 - 80/100 = 0.2
    assert row["self_reported"] == pytest.approx(0.2)
    # No logprobs on the fake completion -> falls back to "no_logprobs", score=1.0
    assert row["predictive_entropy"] == 1.0
    assert row["semantic_entropy"] is None


async def test_process_item_with_strong_labels_escalation_helps(fake_client: FakeOpenAIClient):
    # Weak says "B" (wrong); strong says "A" (correct) => escalation_helps = True.
    fake_client.script(model="gpt-4o-mini", n=1, completions=[make_completion("B")])
    fake_client.script(
        model="gpt-4o-mini",
        n=5,
        completions=[make_completion("B")] * 5,
    )
    fake_client.script(model="gpt-4o-mini", n=1, completions=[make_completion("75")])
    fake_client.script(model="gpt-4o", n=1, completions=[make_completion("A", model="gpt-4o")])

    row = await _process_item(
        _item(answer="A"),
        client=fake_client,
        pe_router=PredictiveEntropyRouter(),
        sc_router=SelfConsistencyRouter(),
        sr_router=SelfReportedRouter(fake_client),
        se_router=None,
        with_strong=True,
    )

    assert row is not None
    assert row["weak_correct"] is False
    assert row["strong_correct"] is True
    assert row["escalation_helps"] is True
