from __future__ import annotations

import asyncio
from typing import Any

from .base import RouterScore, Sampler, UncertaintyRouter

_SIGNAL_COLUMNS = (
    "self_reported",
    "predictive_entropy",
    "self_consistency",
    "semantic_entropy",
)


class LearnedRouter(UncertaintyRouter):
    """Composite router that runs each base signal, feeds them to a fitted
    sklearn pipeline, and returns 1 - P(weak_correct) as the uncertainty score.

    Loads the artifact produced by `med_routing.train.fit` (joblib pickle of
    {pipeline, feature_columns, label, metrics}). The artifact's feature_columns
    determine which sub-routers actually need to run — if a signal isn't in the
    column list, we skip it.

    Inference-time `subject` is not known to the cascade, so we pass
    "UNKNOWN" — the artifact's OneHotEncoder is configured with
    handle_unknown="ignore", which produces an all-zero one-hot for it.
    """

    name = "learned"

    def __init__(
        self,
        *,
        artifact_path: str,
        sub_routers: dict[str, UncertaintyRouter],
    ) -> None:
        import joblib  # local import keeps it out of cold-start unless enabled

        artifact = joblib.load(artifact_path)
        self._pipeline = artifact["pipeline"]
        self._feature_columns: list[str] = list(artifact["feature_columns"])
        self._sub_routers = sub_routers
        # Only run signals the artifact actually consumes.
        self._needed_signals = [c for c in _SIGNAL_COLUMNS if c in self._feature_columns]
        missing = [c for c in self._needed_signals if c not in self._sub_routers]
        if missing:
            raise ValueError(
                f"LearnedRouter artifact requires signals {missing} "
                f"but they were not provided in sub_routers"
            )

    async def score(self, *, messages, weak, sampler: Sampler) -> RouterScore:
        signal_values: dict[str, float] = {}

        # Run sub-routers concurrently. They share the cached weak completion
        # and the cached sampler, so duplicate API calls are avoided.
        tasks = [
            self._sub_routers[name].score(messages=messages, weak=weak, sampler=sampler)
            for name in self._needed_signals
        ]
        results = await asyncio.gather(*tasks)
        for name, result in zip(self._needed_signals, results):
            signal_values[name] = float(result.score)

        prompt_len = len(messages[-1]["content"]) if messages else 0
        record: dict[str, Any] = {**signal_values, "prompt_len": prompt_len, "subject": "UNKNOWN"}
        # Default any column the artifact wants but we didn't compute.
        for c in self._feature_columns:
            record.setdefault(c, 0.0)

        # Build a 2-D object array in the column order the pipeline expects.
        import numpy as np

        X = np.array([[record[c] for c in self._feature_columns]], dtype=object)
        p_correct = float(self._pipeline.predict_proba(X)[0, 1])
        uncertainty = 1.0 - p_correct

        return RouterScore(
            score=max(0.0, min(1.0, uncertainty)),
            extras={
                "p_weak_correct": p_correct,
                "signals": signal_values,
            },
        )
