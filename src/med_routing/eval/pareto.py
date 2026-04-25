"""Pareto-curve evaluation comparing routers on a validation JSONL.

Input: a JSONL produced by `med_routing.train.collect --with-strong`. For each
router (signals already scored by collect; learned router scored in-process
from a pickle), sweep thresholds and emit (escalation_rate, accuracy) points.

The accuracy axis is computed by simulating the cascade: for each item and
threshold tau, escalated = (router_score >= tau). Final answer is the strong
model's answer when escalated, the weak model's answer otherwise. Accuracy is
the fraction of items where the resulting answer matches the gold letter.

Usage:
    python -m med_routing.eval.pareto \\
        --data runs/val_<ts>.jsonl \\
        --learned models/learned_router.pkl \\
        --out reports/pareto.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_BASE_SIGNAL_ROUTERS = (
    "self_reported",
    "predictive_entropy",
    "self_consistency",
    "semantic_entropy",
)


@dataclass
class CurvePoint:
    threshold: float
    escalation_rate: float
    accuracy: float


def compute_curve(
    scores: list[float],
    weak_correct: list[bool],
    strong_correct: list[bool],
) -> list[CurvePoint]:
    """Sweep thresholds, return one CurvePoint per unique score value (plus
    endpoints). Accuracy = simulated cascade accuracy at that threshold."""
    n = len(scores)
    if not (n == len(weak_correct) == len(strong_correct)):
        raise ValueError("scores, weak_correct, strong_correct must align")
    if n == 0:
        return []

    # Unique score values give the exact frontier; +0.0 and slightly>1.0 as
    # endpoints so the curve includes the "escalate everything" and "escalate
    # nothing" extremes.
    thresholds = sorted({0.0, 1.0001, *scores})
    points: list[CurvePoint] = []
    for tau in thresholds:
        escalated = [s >= tau for s in scores]
        correct = [
            sc if e else wc
            for e, wc, sc in zip(escalated, weak_correct, strong_correct)
        ]
        points.append(
            CurvePoint(
                threshold=float(tau),
                escalation_rate=sum(escalated) / n,
                accuracy=sum(correct) / n,
            )
        )
    return points


def score_with_learned(
    rows: list[dict[str, Any]],
    *,
    pickle_path: str,
    use_real_subject: bool = False,
) -> list[float]:
    """Run the fitted learned-router pipeline on each row, return uncertainty
    scores (1 - P(weak_correct)). Default subject="UNKNOWN" matches what the
    runtime cascade passes; --use-real-subject is for an upper-bound comparison."""
    import joblib
    import numpy as np

    artifact = joblib.load(pickle_path)
    pipe = artifact["pipeline"]
    feature_columns: list[str] = list(artifact["feature_columns"])

    out: list[float] = []
    for r in rows:
        record: dict[str, Any] = {}
        for c in feature_columns:
            if c in _BASE_SIGNAL_ROUTERS:
                v = r.get(c)
                record[c] = 0.0 if v is None else float(v)
            elif c == "prompt_len":
                record[c] = float(r.get("prompt_len", 0))
            elif c == "subject":
                record[c] = (
                    r.get("subject", "UNKNOWN") if use_real_subject else "UNKNOWN"
                )
            else:
                record[c] = 0.0
        X = np.array([[record[c] for c in feature_columns]], dtype=object)
        p_correct = float(pipe.predict_proba(X)[0, 1])
        out.append(max(0.0, min(1.0, 1.0 - p_correct)))
    return out


def accuracy_at_escalation(curve: list[CurvePoint], target: float) -> float:
    """Linear interpolation of accuracy at a target escalation rate."""
    if not curve:
        return float("nan")
    sorted_curve = sorted(curve, key=lambda p: p.escalation_rate)
    if target <= sorted_curve[0].escalation_rate:
        return sorted_curve[0].accuracy
    if target >= sorted_curve[-1].escalation_rate:
        return sorted_curve[-1].accuracy
    for a, b in zip(sorted_curve, sorted_curve[1:]):
        if a.escalation_rate <= target <= b.escalation_rate:
            span = b.escalation_rate - a.escalation_rate
            if span <= 0:
                return (a.accuracy + b.accuracy) / 2
            t = (target - a.escalation_rate) / span
            return a.accuracy + t * (b.accuracy - a.accuracy)
    return sorted_curve[-1].accuracy


def curve_auc(curve: list[CurvePoint]) -> float:
    """Trapezoidal area under the accuracy-vs-escalation curve in [0, 1]."""
    if len(curve) < 2:
        return float("nan")
    pts = sorted(curve, key=lambda p: p.escalation_rate)
    area = 0.0
    for a, b in zip(pts, pts[1:]):
        area += 0.5 * (b.escalation_rate - a.escalation_rate) * (a.accuracy + b.accuracy)
    return area


def _run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[fatal] no such file: {data_path}", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in data_path.open(encoding="utf-8") if l.strip()]
    rows = [
        r
        for r in rows
        if r.get("strong_correct") is not None and r.get("weak_correct") is not None
    ]
    if not rows:
        print(
            "[fatal] no rows with both weak_correct and strong_correct;\n"
            "        rerun `med_routing.train.collect --with-strong`",
            file=sys.stderr,
        )
        return 2

    weak_correct = [bool(r["weak_correct"]) for r in rows]
    strong_correct = [bool(r["strong_correct"]) for r in rows]
    weak_acc = sum(weak_correct) / len(rows)
    strong_acc = sum(strong_correct) / len(rows)

    print(f"n={len(rows)}  weak_acc={weak_acc:.3f}  strong_acc={strong_acc:.3f}")

    output: dict[str, Any] = {
        "n_items": len(rows),
        "weak_acc": weak_acc,
        "strong_acc": strong_acc,
        "routers": {},
    }

    for name in args.routers:
        scores_raw = [r.get(name) for r in rows]
        if any(s is None for s in scores_raw):
            n_missing = sum(1 for s in scores_raw if s is None)
            print(f"[skip] {name}: {n_missing}/{len(rows)} rows missing this signal")
            continue
        scores = [float(s) for s in scores_raw]
        curve = compute_curve(scores, weak_correct, strong_correct)
        output["routers"][name] = {
            "points": [asdict(p) for p in curve],
            "auc": curve_auc(curve),
        }

    if args.learned:
        learned_path = Path(args.learned)
        if not learned_path.exists():
            print(f"[skip] learned: pickle not found at {learned_path}")
        else:
            scores = score_with_learned(
                rows,
                pickle_path=str(learned_path),
                use_real_subject=args.use_real_subject,
            )
            curve = compute_curve(scores, weak_correct, strong_correct)
            key = (
                "learned_real_subject" if args.use_real_subject else "learned"
            )
            output["routers"][key] = {
                "points": [asdict(p) for p in curve],
                "auc": curve_auc(curve),
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print(
        f"{'router':<26} {'AUC':>7} {'@10%':>8} {'@25%':>8} {'@50%':>8} {'@75%':>8}"
    )
    print("-" * 70)
    for name, payload in output["routers"].items():
        pts = [CurvePoint(**p) for p in payload["points"]]
        a10 = accuracy_at_escalation(pts, 0.10)
        a25 = accuracy_at_escalation(pts, 0.25)
        a50 = accuracy_at_escalation(pts, 0.50)
        a75 = accuracy_at_escalation(pts, 0.75)
        print(
            f"{name:<26} {payload['auc']:>7.4f} "
            f"{a10:>8.3f} {a25:>8.3f} {a50:>8.3f} {a75:>8.3f}"
        )
    print(f"\nSaved curves to {out_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute Pareto curves (accuracy vs escalation rate) per router."
    )
    p.add_argument(
        "--data",
        required=True,
        help="JSONL from collect.py --with-strong (validation split).",
    )
    p.add_argument(
        "--routers",
        nargs="*",
        default=[
            "self_reported",
            "predictive_entropy",
            "self_consistency",
            "semantic_entropy",
        ],
        help="Names of router-score columns in the JSONL to evaluate.",
    )
    p.add_argument("--learned", default=None, help="Path to learned-router pickle.")
    p.add_argument(
        "--use-real-subject",
        action="store_true",
        help=(
            "When scoring with the learned router, pass the row's actual subject "
            "instead of UNKNOWN. Use for upper-bound comparison; default matches "
            "production cascade behavior."
        ),
    )
    p.add_argument("--out", default="reports/pareto.json")
    args = p.parse_args()
    sys.exit(_run(args))


if __name__ == "__main__":
    main()