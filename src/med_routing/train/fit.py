"""Fit a small classifier on the JSONL emitted by collect.py.

Predicts P(weak_correct) from the four uncertainty signals + cheap prompt
features. The runtime LearnedRouter will use 1 - P(weak_correct) as its
"uncertainty score" so it inherits the existing higher-=-escalate convention.

Usage:
    python -m med_routing.train.fit \
        --data runs/train_<ts>.jsonl \
        --model gbm \
        --out models/learned_router.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "self_reported",
    "predictive_entropy",
    "self_consistency",
    "semantic_entropy",
    "prompt_len",
]
CATEGORICAL_FEATURES = ["subject"]


@dataclass
class FitResult:
    pipeline: Pipeline
    feature_columns: list[str]
    label: str
    n_train: int
    n_test: int
    metrics: dict[str, float]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_xy(
    rows: list[dict[str, Any]], label: str
) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    """Build X (list-of-dicts; ColumnTransformer expects 2D) and y, dropping
    features that are missing on every row."""
    # Drop rows missing the label.
    rows = [r for r in rows if r.get(label) is not None]
    if not rows:
        raise SystemExit(f"No rows with non-null '{label}'.")

    # Drop numeric features that are None on every row (e.g. semantic_entropy
    # when collect ran with --no-semantic).
    numeric_cols = [
        c for c in NUMERIC_FEATURES if any(r.get(c) is not None for r in rows)
    ]
    feature_cols = numeric_cols + CATEGORICAL_FEATURES

    X = [
        {
            **{c: (r.get(c) if r.get(c) is not None else 0.0) for c in numeric_cols},
            **{c: (r.get(c) or "UNKNOWN") for c in CATEGORICAL_FEATURES},
        }
        for r in rows
    ]
    y = np.array([int(bool(r[label])) for r in rows])
    return X, y, feature_cols


def _records_to_array(records: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    """ColumnTransformer wants a 2-D array (or DataFrame). Build an object-dtype
    array indexed by column name to keep the pickle pandas-free."""
    return np.array([[r[c] for c in columns] for r in records], dtype=object)


def _make_pipeline(
    *, model: str, numeric_cols: list[str], categorical_cols: list[str]
) -> Pipeline:
    numeric_idx = list(range(len(numeric_cols)))
    cat_idx = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_idx),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_idx,
            ),
        ]
    )

    if model == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=0,
        )
    elif model == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    else:
        raise ValueError(f"unknown model {model!r}; expected gbm or logreg")

    return Pipeline([("pre", pre), ("clf", clf)])


def fit_from_rows(
    rows: list[dict[str, Any]],
    *,
    label: str = "weak_correct",
    model: str = "gbm",
    test_size: float = 0.2,
    random_state: int = 0,
) -> FitResult:
    X_records, y, feature_cols = _build_xy(rows, label=label)
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]

    X_arr = _records_to_array(X_records, feature_cols)

    # Stratify on the label only when there are at least 2 of each class, else
    # train_test_split errors out on tiny / imbalanced data.
    stratify = y if min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe = _make_pipeline(
        model=model, numeric_cols=numeric_cols, categorical_cols=CATEGORICAL_FEATURES
    )
    pipe.fit(X_tr, y_tr)

    p_tr = pipe.predict_proba(X_tr)[:, 1]
    p_te = pipe.predict_proba(X_te)[:, 1]

    metrics: dict[str, float] = {
        "train_acc": float(((p_tr >= 0.5) == y_tr).mean()),
        "test_acc": float(((p_te >= 0.5) == y_te).mean()),
        "train_brier": float(brier_score_loss(y_tr, p_tr)),
        "test_brier": float(brier_score_loss(y_te, p_te)),
        "train_logloss": float(log_loss(y_tr, p_tr, labels=[0, 1])),
        "test_logloss": float(log_loss(y_te, p_te, labels=[0, 1])),
        "label_pos_rate": float(y.mean()),
    }
    # AUC undefined when test set is single-class (common at very low n).
    if len(np.unique(y_te)) == 2:
        metrics["test_auc"] = float(roc_auc_score(y_te, p_te))
    if len(np.unique(y_tr)) == 2:
        metrics["train_auc"] = float(roc_auc_score(y_tr, p_tr))

    return FitResult(
        pipeline=pipe,
        feature_columns=feature_cols,
        label=label,
        n_train=len(y_tr),
        n_test=len(y_te),
        metrics=metrics,
    )


def _save(result: FitResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": result.pipeline,
        "feature_columns": result.feature_columns,
        "label": result.label,
        "metrics": result.metrics,
    }
    joblib.dump(artifact, out_path)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "label": result.label,
                "feature_columns": result.feature_columns,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "metrics": result.metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[fatal] no such file: {data_path}", file=sys.stderr)
        return 2

    rows = _load_jsonl(data_path)
    print(f"Loaded {len(rows)} rows from {data_path}")

    result = fit_from_rows(
        rows,
        label=args.label,
        model=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"\nFit summary (label={result.label}, model={args.model}):")
    print(f"  features: {result.feature_columns}")
    print(f"  n_train={result.n_train}  n_test={result.n_test}")
    for k, v in result.metrics.items():
        print(f"  {k:<18}{v:.4f}")

    out_path = Path(args.out)
    _save(result, out_path)
    print(f"\nSaved model to {out_path}")
    print(f"Saved metadata to {out_path.with_suffix('.meta.json')}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Fit the learned router from a collect JSONL.")
    p.add_argument("--data", required=True, help="Path to JSONL from collect.py")
    p.add_argument(
        "--label",
        default="weak_correct",
        choices=["weak_correct", "escalation_helps"],
    )
    p.add_argument("--model", default="gbm", choices=["gbm", "logreg"])
    p.add_argument("--out", default="models/learned_router.pkl")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=0)
    args = p.parse_args()
    sys.exit(_run(args))


if __name__ == "__main__":
    main()
