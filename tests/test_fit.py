from __future__ import annotations

import json
from pathlib import Path

import pytest

sklearn = pytest.importorskip("sklearn")
joblib = pytest.importorskip("joblib")

from med_routing.train.fit import _save, fit_from_rows


def _synthetic_rows(n: int = 200, seed: int = 0) -> list[dict]:
    """High predictive_entropy => weak_correct=False with high probability.
    The classifier should learn this trivially."""
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        if i % 2 == 0:
            pe = rng.uniform(0.6, 0.9)
            weak_correct = rng.random() < 0.1
        else:
            pe = rng.uniform(0.0, 0.2)
            weak_correct = rng.random() < 0.9
        rows.append(
            {
                "qid": str(i),
                "subject": rng.choice(["Anatomy", "Physiology", "Surgery"]),
                "prompt_len": rng.randint(100, 500),
                "self_reported": rng.uniform(0.0, 0.2),
                "predictive_entropy": pe,
                "self_consistency": rng.uniform(0.0, 0.3),
                "semantic_entropy": None,  # exercises the all-None drop path
                "weak_correct": weak_correct,
            }
        )
    return rows


def test_fit_learns_predictive_entropy_signal():
    rows = _synthetic_rows(n=300)
    result = fit_from_rows(rows, label="weak_correct", model="gbm")

    # The signal is by construction strong; the classifier should clear 0.85 AUC.
    assert result.metrics["test_auc"] > 0.85
    # semantic_entropy was all-None so it should be dropped.
    assert "semantic_entropy" not in result.feature_columns
    # The other four numeric features + subject should be retained.
    assert "predictive_entropy" in result.feature_columns
    assert "subject" in result.feature_columns


def test_fit_pipeline_round_trips_to_disk(tmp_path: Path):
    rows = _synthetic_rows(n=200)
    result = fit_from_rows(rows, label="weak_correct", model="logreg")

    out = tmp_path / "model.pkl"
    _save(result, out)

    artifact = joblib.load(out)
    pipe = artifact["pipeline"]
    feature_cols = artifact["feature_columns"]

    # Build a single inference row in the same column order; predict_proba
    # should return a probability for the positive class.
    import numpy as np

    rec = {c: 0.1 for c in feature_cols if c != "subject"}
    rec["subject"] = "Anatomy"
    X = np.array([[rec[c] for c in feature_cols]], dtype=object)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (1,)
    assert 0.0 <= float(p[0]) <= 1.0

    # Sidecar metadata file should exist.
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["label"] == "weak_correct"
    assert meta["feature_columns"] == feature_cols
