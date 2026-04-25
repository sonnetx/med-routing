from __future__ import annotations

import pytest

from med_routing.eval.aggregator import EvalAggregator
from med_routing.metrics import (
    ACCURACY_BY_SUBJECT,
    CALIBRATION_BIN_TOTAL,
    bin_midpoint,
    score_bin,
)


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "0.0-0.1"),
        (0.05, "0.0-0.1"),
        (0.1, "0.1-0.2"),
        (0.45, "0.4-0.5"),
        (0.99, "0.9-1.0"),
        (1.0, "0.9-1.0"),
        (1.5, "0.9-1.0"),
        (-0.2, "0.0-0.1"),
    ],
)
def test_score_bin(score, expected):
    assert score_bin(score) == expected


def test_bin_midpoint():
    assert bin_midpoint("0.0-0.1") == pytest.approx(0.05)
    assert bin_midpoint("0.9-1.0") == pytest.approx(0.95)


def test_aggregator_kept_weak_vs_escalated_accuracy():
    agg = EvalAggregator()
    agg.observe(router="r", score=0.1, escalated=False, correct=True, subject="Anatomy")
    agg.observe(router="r", score=0.1, escalated=False, correct=False, subject="Anatomy")
    agg.observe(router="r", score=0.8, escalated=True, correct=True, subject="Surgery")
    agg.observe(router="r", score=0.8, escalated=True, correct=True, subject="Surgery")

    summary = agg.observe(router="r", score=0.9, escalated=True, correct=False, subject="Surgery")

    assert summary["weak_accuracy"] == pytest.approx(0.5)
    assert summary["escalated_accuracy"] == pytest.approx(2 / 3)
    assert summary["accuracy"] == pytest.approx(3 / 5)
    assert summary["n"] == 5


def test_aggregator_increments_calibration_counters():
    agg = EvalAggregator()
    before = CALIBRATION_BIN_TOTAL.labels(router="cal", bin="0.4-0.5")._value.get()
    agg.observe(router="cal", score=0.45, escalated=False, correct=True)
    agg.observe(router="cal", score=0.42, escalated=False, correct=False)
    after = CALIBRATION_BIN_TOTAL.labels(router="cal", bin="0.4-0.5")._value.get()
    assert after - before == 2


def test_aggregator_per_subject_gauge_set():
    agg = EvalAggregator()
    agg.observe(router="rs", score=0.2, escalated=False, correct=True, subject="Pediatrics")
    agg.observe(router="rs", score=0.2, escalated=False, correct=False, subject="Pediatrics")
    agg.observe(router="rs", score=0.2, escalated=False, correct=True, subject="Pediatrics")

    val = ACCURACY_BY_SUBJECT.labels(router="rs", subject="Pediatrics")._value.get()
    assert val == pytest.approx(2 / 3)


def test_aggregator_perfectly_calibrated_router_has_low_ece():
    """A router that's right at low scores and wrong at high scores has ECE near
    the discretization floor (~0.05, set by the gap between bin midpoint and the
    true confidence at the bin edges)."""
    agg = EvalAggregator()
    for _ in range(20):
        agg.observe(router="perfect", score=0.05, escalated=False, correct=True)
    summary = None
    for _ in range(20):
        summary = agg.observe(router="perfect", score=0.95, escalated=False, correct=False)
    assert summary is not None
    assert summary["ece"] < 0.1


def test_aggregator_uninformative_router_has_higher_ece():
    """If accuracy is flat 50% across all bins, ECE > 0."""
    agg = EvalAggregator()
    agg.observe(router="bad", score=0.05, escalated=False, correct=True)
    agg.observe(router="bad", score=0.05, escalated=False, correct=False)
    summary = agg.observe(router="bad", score=0.95, escalated=False, correct=True)
    assert summary["ece"] > 0.2
