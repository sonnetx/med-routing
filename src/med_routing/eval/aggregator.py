from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..metrics import (
    ACCURACY,
    ACCURACY_BY_SUBJECT,
    CALIBRATION_BIN_CORRECT,
    CALIBRATION_BIN_TOTAL,
    ECE,
    ESCALATED_ACCURACY,
    KEPT_WEAK_ACCURACY,
    bin_midpoint,
    score_bin,
)


@dataclass
class _RouterTally:
    total: int = 0
    correct: int = 0
    weak_total: int = 0
    weak_correct: int = 0
    esc_total: int = 0
    esc_correct: int = 0
    bin_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bin_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    subject_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    subject_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class EvalAggregator:
    """Server-side, per-router rolling tallies for evaluation metrics.

    Calibration counters are also exposed as Prometheus Counters so reliability
    diagrams can be built directly in Grafana via PromQL division. The other
    quantities are pushed as Gauges every time observe() is called.
    """

    def __init__(self, store: object | None = None) -> None:
        self._t: dict[str, _RouterTally] = defaultdict(_RouterTally)
        self._store = store

    def observe(
        self,
        *,
        router: str,
        score: float,
        escalated: bool,
        correct: bool,
        subject: str | None = None,
    ) -> dict[str, float]:
        t = self._t[router]
        t.total += 1
        t.correct += int(correct)

        if escalated:
            t.esc_total += 1
            t.esc_correct += int(correct)
        else:
            t.weak_total += 1
            t.weak_correct += int(correct)

        bin_label = score_bin(score)
        t.bin_total[bin_label] += 1
        t.bin_correct[bin_label] += int(correct)
        CALIBRATION_BIN_TOTAL.labels(router=router, bin=bin_label).inc()
        if correct:
            CALIBRATION_BIN_CORRECT.labels(router=router, bin=bin_label).inc()

        ACCURACY.labels(router=router).set(t.correct / t.total)
        if t.weak_total:
            KEPT_WEAK_ACCURACY.labels(router=router).set(t.weak_correct / t.weak_total)
        if t.esc_total:
            ESCALATED_ACCURACY.labels(router=router).set(t.esc_correct / t.esc_total)

        if subject:
            t.subject_total[subject] += 1
            t.subject_correct[subject] += int(correct)
            ACCURACY_BY_SUBJECT.labels(router=router, subject=subject).set(
                t.subject_correct[subject] / t.subject_total[subject]
            )

        ece = self._compute_ece(t)
        ECE.labels(router=router).set(ece)

        if self._store is not None:
            try:
                self._store.insert_eval_row(
                    router=router, score=score, escalated=escalated,
                    correct=correct, subject=subject,
                )
            except Exception:
                pass

        return {
            "accuracy": t.correct / t.total if t.total else 0.0,
            "weak_accuracy": (t.weak_correct / t.weak_total) if t.weak_total else 0.0,
            "escalated_accuracy": (t.esc_correct / t.esc_total) if t.esc_total else 0.0,
            "ece": ece,
            "n": t.total,
        }

    @staticmethod
    def _compute_ece(t: _RouterTally) -> float:
        """Weighted gap between bin accuracy and the confidence implied by score.

        We treat `score` as P(answer is wrong), so expected weak-accuracy in a bin
        with midpoint u is (1 - u). ECE = sum_b (n_b/N) * |(1 - u_b) - acc_b|.
        """
        if t.total == 0:
            return 0.0
        ece = 0.0
        for label, n_b in t.bin_total.items():
            if n_b == 0:
                continue
            acc_b = t.bin_correct[label] / n_b
            expected = 1.0 - bin_midpoint(label)
            ece += (n_b / t.total) * abs(expected - acc_b)
        return ece
