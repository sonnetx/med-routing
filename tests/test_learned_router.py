from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from med_routing.routers.learned import LearnedRouter
from med_routing.routers.predictive_entropy import PredictiveEntropyRouter
from med_routing.routers.self_consistency import SelfConsistencyRouter
from med_routing.routers.self_reported import SelfReportedRouter
from med_routing.train.fit import _save, fit_from_rows
from tests.conftest import FakeOpenAIClient, make_completion
from tests.test_fit import _synthetic_rows


def _train_artifact(tmp_path: Path) -> Path:
    """Fit a real (small, synthetic) artifact and return its path."""
    rows = _synthetic_rows(n=400)
    result = fit_from_rows(rows, label="weak_correct", model="gbm")
    out = tmp_path / "model.pkl"
    _save(result, out)
    return out


async def test_learned_router_produces_score_inverse_to_p_correct(tmp_path: Path):
    artifact_path = _train_artifact(tmp_path)
    fake_client = FakeOpenAIClient()
    sub_routers = {
        "self_reported": SelfReportedRouter(fake_client),
        "predictive_entropy": PredictiveEntropyRouter(),
        "self_consistency": SelfConsistencyRouter(),
    }
    router = LearnedRouter(artifact_path=str(artifact_path), sub_routers=sub_routers)

    fake_client.script(
        model="gpt-4o-mini",
        n=1,
        completions=[make_completion("90")],  # self-report follow-up
    )

    weak = make_completion("A")  # no logprobs => predictive_entropy = 1.0
    samples = [make_completion("A")] * 5

    async def sampler(*, n, temperature):
        return samples[:n]

    result = await router.score(
        messages=[{"role": "user", "content": "Q?"}], weak=weak, sampler=sampler
    )

    assert 0.0 <= result.score <= 1.0
    assert "p_weak_correct" in result.extras
    assert result.score == pytest.approx(1.0 - result.extras["p_weak_correct"])
    # All three signals captured.
    assert set(result.extras["signals"]) == {
        "self_reported",
        "predictive_entropy",
        "self_consistency",
    }


async def test_learned_router_rejects_artifact_missing_signal(tmp_path: Path):
    artifact_path = _train_artifact(tmp_path)
    sub_routers = {
        # Deliberately omit predictive_entropy — fit's _synthetic_rows includes it
        # so the artifact requires it; constructor should raise.
        "self_reported": SelfReportedRouter(FakeOpenAIClient()),
        "self_consistency": SelfConsistencyRouter(),
    }
    with pytest.raises(ValueError, match="predictive_entropy"):
        LearnedRouter(artifact_path=str(artifact_path), sub_routers=sub_routers)
