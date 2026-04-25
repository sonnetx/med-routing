from __future__ import annotations

import asyncio
from typing import Any

from .base import RouterScore, Sampler, UncertaintyRouter


def _last_user_prompt(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


class RouteLLMRouter(UncertaintyRouter):
    """Wraps lm-sys/RouteLLM as a baseline router.

    RouteLLM is a *pre-call* learned router: it scores the prompt itself, with no
    knowledge of the weak model's output. The score returned by
    `batch_calculate_win_rate` is the probability that the strong model would
    win on this prompt — higher = should escalate, which lines up directly with
    our `RouterScore` convention (higher = more uncertain).

    Note on cost: the cascade always runs the weak model before invoking any
    router, so this wrapper does not capture RouteLLM's true "skip-the-weak-call"
    cost advantage. It exists as an accuracy baseline. A fair cost comparison
    requires an eval-mode bypass path that is intentionally out of scope here.
    """

    name = "routellm"

    def __init__(
        self,
        *,
        kind: str = "mf",
        strong_model: str = "gpt-4-1106-preview",
        weak_model: str = "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",
        controller: Any | None = None,
    ) -> None:
        self._kind = kind
        if controller is None:
            # Heavy import gated on construction so the main process does not
            # pay torch/transformers cost when ENABLE_ROUTELLM=false.
            from routellm.controller import Controller

            controller = Controller(
                routers=[kind],
                strong_model=strong_model,
                weak_model=weak_model,
                progress_bar=False,
            )
        self._controller = controller

    async def score(self, *, messages, weak, sampler: Sampler) -> RouterScore:
        prompt = _last_user_prompt(messages)
        if not prompt:
            return RouterScore(score=1.0, extras={"reason": "no_user_prompt"})

        win_rates = await asyncio.to_thread(
            self._controller.batch_calculate_win_rate,
            prompts=[prompt],
            router=self._kind,
        )
        # batch_calculate_win_rate returns a pandas Series; coerce to float.
        score = float(list(win_rates)[0])
        score = max(0.0, min(1.0, score))
        return RouterScore(
            score=score,
            extras={"kind": self._kind, "strong_win_rate": score},
        )
