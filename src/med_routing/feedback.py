from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThresholdSweepRow:
    threshold: float
    n_total: int
    n_would_keep: int
    n_would_escalate: int
    kept_accuracy: float | None       # simulated weak-only accuracy on kept rows
    escalated_accuracy: float | None  # simulated strong accuracy on escalated rows
    escalation_rate: float

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 3),
            "n_total": self.n_total,
            "n_would_keep": self.n_would_keep,
            "n_would_escalate": self.n_would_escalate,
            "kept_accuracy": self.kept_accuracy,
            "escalated_accuracy": self.escalated_accuracy,
            "escalation_rate": round(self.escalation_rate, 3),
        }


def sweep_thresholds(
    rows: list[dict],
    *,
    candidates: list[float] | None = None,
) -> list[ThresholdSweepRow]:
    """Replay labeled eval rows against each candidate threshold.

    For each row in `rows` (must have keys: score, escalated, correct), simulate
    what the cascade *would* have done at threshold `t`:
      - score >= t  → "would escalate"
      - score <  t  → "would keep weak"

    The simulation uses observed correctness — i.e. we only know what really
    happened to each row under whatever threshold was live when it was logged.
    A row originally kept-weak gives us a real (weak, correct?) data point; a
    row originally escalated gives us a real (strong, correct?) data point.
    Simulated kept_accuracy is computed only over rows that we actually have
    weak-correctness for; same for escalated_accuracy. This is conservative —
    it doesn't pretend to know what weak would have said on previously-escalated
    rows.
    """
    if candidates is None:
        candidates = [round(0.05 * i, 2) for i in range(0, 21)]  # 0.00..1.00

    out: list[ThresholdSweepRow] = []
    for t in candidates:
        kept_correct = kept_total = 0
        esc_correct = esc_total = 0
        n_keep = n_escalate = 0
        for r in rows:
            score = float(r["score"])
            originally_escalated = bool(r["escalated"])
            correct = bool(r["correct"])
            would_escalate = score >= t
            if would_escalate:
                n_escalate += 1
                if originally_escalated:
                    esc_total += 1
                    esc_correct += int(correct)
            else:
                n_keep += 1
                if not originally_escalated:
                    kept_total += 1
                    kept_correct += int(correct)
        out.append(
            ThresholdSweepRow(
                threshold=t,
                n_total=len(rows),
                n_would_keep=n_keep,
                n_would_escalate=n_escalate,
                kept_accuracy=(kept_correct / kept_total) if kept_total else None,
                escalated_accuracy=(esc_correct / esc_total) if esc_total else None,
                escalation_rate=(n_escalate / len(rows)) if rows else 0.0,
            )
        )
    return out


def recommend_threshold(
    rows: list[dict],
    *,
    target_kept_accuracy: float = 0.85,
    min_kept_n: int = 5,
) -> dict | None:
    """Recommend a threshold that keeps weak-only accuracy on the kept-weak
    subset at or above `target_kept_accuracy`, escalating as little as possible.

    Strategy: sweep thresholds; among those whose simulated kept_accuracy meets
    the target (and have enough samples), choose the *highest* threshold (least
    escalation, biggest savings). If no threshold meets the target, return the
    one with the best kept_accuracy.
    """
    if not rows:
        return None

    sweep = sweep_thresholds(rows)

    qualifying = [
        s for s in sweep
        if s.kept_accuracy is not None
        and s.kept_accuracy >= target_kept_accuracy
        and s.n_would_keep >= min_kept_n
    ]
    if qualifying:
        chosen = max(qualifying, key=lambda s: s.threshold)
        reason = f"highest threshold meeting kept_accuracy ≥ {target_kept_accuracy}"
    else:
        # Fall back: pick the threshold with the best simulated kept_accuracy.
        ranked = [s for s in sweep if s.kept_accuracy is not None]
        if not ranked:
            return None
        chosen = max(ranked, key=lambda s: s.kept_accuracy or 0.0)
        reason = f"target {target_kept_accuracy} unreachable; picked best observed kept_accuracy"

    return {
        "threshold": chosen.threshold,
        "reason": reason,
        "target_kept_accuracy": target_kept_accuracy,
        "stats_at_threshold": chosen.as_dict(),
        "sweep": [s.as_dict() for s in sweep],
    }


def recommend_for_all(
    rows_by_router: dict[str, list[dict]],
    *,
    target_kept_accuracy: float = 0.85,
    min_samples: int = 30,
) -> dict[str, dict]:
    """Per-router recommendations. Routers without enough labeled data are skipped."""
    out: dict[str, dict] = {}
    for router, rows in rows_by_router.items():
        if len(rows) < min_samples:
            out[router] = {
                "skipped": True,
                "reason": f"only {len(rows)} labeled rows, need ≥{min_samples}",
            }
            continue
        rec = recommend_threshold(rows, target_kept_accuracy=target_kept_accuracy)
        if rec is None:
            out[router] = {"skipped": True, "reason": "no labeled rows after filtering"}
        else:
            out[router] = rec
    return out
