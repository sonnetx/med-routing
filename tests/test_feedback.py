from __future__ import annotations

from pathlib import Path

import pytest

from med_routing.config import (
    clear_runtime_threshold,
    get_settings,
    runtime_overrides,
    set_runtime_threshold,
)
from med_routing.feedback import (
    recommend_for_all,
    recommend_threshold,
    sweep_thresholds,
)
from med_routing.store import Store


@pytest.fixture(autouse=True)
def _clean_overrides():
    for r in list(runtime_overrides().keys()):
        clear_runtime_threshold(r)
    yield
    for r in list(runtime_overrides().keys()):
        clear_runtime_threshold(r)


def _row(score, escalated, correct):
    return {"score": score, "escalated": escalated, "correct": correct}


def test_sweep_simple_dataset():
    rows = [_row(0.05, False, True), _row(0.45, False, False), _row(0.85, True, True)]
    sweep = sweep_thresholds(rows, candidates=[0.0, 0.5, 1.0])
    by_t = {s.threshold: s for s in sweep}

    # threshold = 0.0 → everything escalates
    assert by_t[0.0].n_would_escalate == 3
    assert by_t[0.0].n_would_keep == 0

    # threshold = 0.5 → kept = 2 (the 0.05 and 0.45), escalated = 1 (the 0.85)
    s = by_t[0.5]
    assert s.n_would_keep == 2
    assert s.n_would_escalate == 1
    # both kept rows were originally not escalated → kept_accuracy = 1/2
    assert s.kept_accuracy == pytest.approx(0.5)
    # the 0.85 row was originally escalated and correct → escalated_accuracy = 1.0
    assert s.escalated_accuracy == pytest.approx(1.0)

    # threshold = 1.0 → nothing escalates
    assert by_t[1.0].n_would_escalate == 0


def test_recommend_picks_highest_threshold_meeting_target():
    """If accuracy is good in low bins and bad in high bins, recommend a high threshold."""
    rows = (
        [_row(0.05, False, True)] * 20      # very confident, all correct
        + [_row(0.4, False, True)] * 8      # mid confidence, mostly correct
        + [_row(0.4, False, False)] * 2
        + [_row(0.8, False, False)] * 10    # high uncertainty, mostly wrong
    )
    rec = recommend_threshold(rows, target_kept_accuracy=0.85, min_kept_n=5)
    assert rec is not None
    # Sweep keeps weak when score < t. The 0.8 rows enter the "kept" bucket only
    # at t > 0.8 — at t=0.85 kept_accuracy drops to 28/40=0.7 (fails target). So
    # the highest qualifying threshold is 0.80, where kept = 30 rows @ 28/30=0.93.
    assert rec["threshold"] == pytest.approx(0.80)
    assert rec["stats_at_threshold"]["kept_accuracy"] >= 0.85


def test_recommend_falls_back_when_target_unreachable():
    rows = [_row(0.5, False, False)] * 10  # everything wrong, no threshold helps
    rec = recommend_threshold(rows, target_kept_accuracy=0.95)
    assert rec is not None
    assert "unreachable" in rec["reason"]


def test_recommend_for_all_skips_routers_with_too_few_samples():
    by_router = {
        "a": [_row(0.1, False, True)] * 5,           # too few
        "b": [_row(0.1, False, True)] * 50,
    }
    out = recommend_for_all(by_router, target_kept_accuracy=0.85, min_samples=30)
    assert out["a"]["skipped"] is True
    assert "skipped" not in out["b"]
    assert out["b"]["threshold"] is not None


def test_runtime_threshold_override_wins_over_settings():
    s = get_settings()
    default = s.threshold_self_consistency
    set_runtime_threshold("self_consistency", 0.99)
    try:
        assert s.threshold_for("self_consistency") == 0.99
    finally:
        clear_runtime_threshold("self_consistency")
    assert s.threshold_for("self_consistency") == default


def test_store_persists_threshold_and_history(tmp_path: Path):
    store = Store(tmp_path / "fb.db")
    store.set_threshold(router="self_consistency", threshold=0.42, reason="test1")
    store.set_threshold(router="self_consistency", threshold=0.55, reason="test2")
    store.set_threshold(router="self_reported", threshold=0.20, reason="test3")

    current = store.get_thresholds()
    assert current == {"self_consistency": 0.55, "self_reported": 0.20}

    # History keeps both updates for self_consistency.
    hist = store.threshold_history(router="self_consistency")
    assert len(hist) == 2
    assert hist[0]["threshold"] == 0.55  # most recent first
    assert hist[1]["threshold"] == 0.42


def test_runtime_overrides_round_trip():
    set_runtime_threshold("self_reported", 0.30)
    set_runtime_threshold("predictive_entropy", 0.45)
    assert runtime_overrides() == {"self_reported": 0.30, "predictive_entropy": 0.45}
    clear_runtime_threshold("self_reported")
    assert "self_reported" not in runtime_overrides()
