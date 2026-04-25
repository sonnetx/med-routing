from __future__ import annotations

from pathlib import Path

import pytest

from med_routing.eval.pareto import (
    CurvePoint,
    accuracy_at_escalation,
    compute_curve,
    curve_auc,
)


def test_compute_curve_endpoints():
    """At threshold 0 we escalate everything (acc = strong_acc); at threshold>1
    we escalate nothing (acc = weak_acc)."""
    scores = [0.1, 0.5, 0.9]
    weak_correct = [True, True, False]
    strong_correct = [True, False, True]
    curve = compute_curve(scores, weak_correct, strong_correct)

    by_threshold = {round(p.threshold, 4): p for p in curve}
    p_zero = by_threshold[0.0]
    assert p_zero.escalation_rate == pytest.approx(1.0)
    assert p_zero.accuracy == pytest.approx(2 / 3)  # strong correct on 2/3

    p_inf = by_threshold[1.0001]
    assert p_inf.escalation_rate == pytest.approx(0.0)
    assert p_inf.accuracy == pytest.approx(2 / 3)  # weak correct on 2/3


def test_compute_curve_perfect_router_recovers_oracle():
    """A router whose score is high iff the weak model is wrong should let the
    cascade hit the oracle accuracy = max(weak_correct[i], strong_correct[i])."""
    weak_correct = [True, True, False, False]
    strong_correct = [False, True, True, False]
    # Score high (=1.0) iff weak_correct is False; low (=0.0) otherwise.
    scores = [0.0 if wc else 1.0 for wc in weak_correct]
    curve = compute_curve(scores, weak_correct, strong_correct)

    # At a threshold between 0.0 and 1.0 we escalate exactly the wrong-weak
    # items; final correctness = (weak right when not escalated) OR
    # (strong right when escalated). Items 0,1 stay weak (right, right);
    # items 2,3 escalate (strong is right on 2, wrong on 4) -> 3/4.
    best = max(p.accuracy for p in curve)
    assert best == pytest.approx(0.75)

    oracle = sum(wc or sc for wc, sc in zip(weak_correct, strong_correct)) / 4
    assert best == pytest.approx(oracle)


def test_compute_curve_random_scores_lie_between_baselines():
    """A useless router can't go below min(weak_acc, strong_acc) at the extremes,
    and the curve over thresholds spans the gap between weak_acc and strong_acc."""
    import random

    rng = random.Random(0)
    weak_correct = [rng.random() < 0.7 for _ in range(200)]
    strong_correct = [rng.random() < 0.85 for _ in range(200)]
    scores = [rng.random() for _ in range(200)]

    curve = compute_curve(scores, weak_correct, strong_correct)
    accuracies = [p.accuracy for p in curve]
    weak_acc = sum(weak_correct) / 200
    strong_acc = sum(strong_correct) / 200
    assert min(accuracies) >= min(weak_acc, strong_acc) - 0.05
    assert max(accuracies) <= max(weak_acc, strong_acc) + 0.05


def test_accuracy_at_escalation_interpolates_linearly():
    pts = [
        CurvePoint(threshold=0.0, escalation_rate=0.0, accuracy=0.7),
        CurvePoint(threshold=0.5, escalation_rate=0.5, accuracy=0.8),
        CurvePoint(threshold=1.0, escalation_rate=1.0, accuracy=0.85),
    ]
    assert accuracy_at_escalation(pts, 0.0) == pytest.approx(0.7)
    assert accuracy_at_escalation(pts, 0.25) == pytest.approx(0.75)
    assert accuracy_at_escalation(pts, 0.5) == pytest.approx(0.8)
    assert accuracy_at_escalation(pts, 1.0) == pytest.approx(0.85)
    # Outside the curve -> clamp to nearest endpoint.
    assert accuracy_at_escalation(pts, -0.5) == pytest.approx(0.7)
    assert accuracy_at_escalation(pts, 2.0) == pytest.approx(0.85)


def test_curve_auc_perfect_curve_is_one():
    pts = [
        CurvePoint(threshold=0.0, escalation_rate=0.0, accuracy=1.0),
        CurvePoint(threshold=1.0, escalation_rate=1.0, accuracy=1.0),
    ]
    assert curve_auc(pts) == pytest.approx(1.0)


def test_score_with_learned_round_trips(tmp_path: Path):
    """Train a tiny model, save it, then run score_with_learned over a few
    rows and confirm it returns probabilities in [0, 1]."""
    pytest.importorskip("sklearn")

    from med_routing.eval.pareto import score_with_learned
    from med_routing.train.fit import _save, fit_from_rows
    from tests.test_fit import _synthetic_rows

    rows = _synthetic_rows(n=200)
    result = fit_from_rows(rows, label="weak_correct", model="logreg")
    pkl = tmp_path / "model.pkl"
    _save(result, pkl)

    test_rows = _synthetic_rows(n=10, seed=42)
    scores = score_with_learned(test_rows, pickle_path=str(pkl))
    assert len(scores) == 10
    assert all(0.0 <= s <= 1.0 for s in scores)