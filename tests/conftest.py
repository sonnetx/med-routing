from __future__ import annotations

from typing import Any

import pytest

from med_routing.config import cost_usd
from med_routing.llm.openai_client import Completion, TokenLogprob
from med_routing.metrics import MODEL_CALLS_TOTAL, PROCESSOR_CALLS_TOTAL
from med_routing.processors import get_processor


def make_completion(text: str, *, logprobs: list[TokenLogprob] | None = None, model: str = "gpt-4o-mini") -> Completion:
    prompt_tokens, completion_tokens = 10, 2
    return Completion(
        model=model,
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost_usd(model, prompt_tokens, completion_tokens),
        logprobs=logprobs,
    )


class FakeOpenAIClient:
    """Records calls and returns scripted completions in order.

    `script` maps a (model, n) bucket to a deque of canned text lists; if no
    script matches, returns a single canned default.
    """

    def __init__(self, default_text: str = "A") -> None:
        self.calls: list[dict[str, Any]] = []
        self.default_text = default_text
        self._scripts: dict[tuple[str, int], list[list[Completion]]] = {}

    def script(self, *, model: str, n: int, completions: list[Completion]) -> None:
        self._scripts.setdefault((model, n), []).append(completions)

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tier: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        n: int = 1,
        seed: int | None = None,
    ) -> list[Completion]:
        self.calls.append(
            {
                "model": model,
                "tier": tier,
                "temperature": temperature,
                "n": n,
                "logprobs": logprobs,
                "messages": messages,
            }
        )
        bucket = self._scripts.get((model, n))
        if bucket:
            out = bucket.pop(0)
        else:
            out = [make_completion(self.default_text, model=model) for _ in range(n)]
        # Mirror the real client's metric side-effects so cascade-level tests reflect
        # the actual call-boundary contract.
        MODEL_CALLS_TOTAL.labels(model=model, tier=tier).inc(len(out))
        proc = get_processor(model)
        PROCESSOR_CALLS_TOTAL.labels(processor=proc.name, entity=proc.entity, region=proc.region).inc(len(out))
        return out


@pytest.fixture
def fake_client() -> FakeOpenAIClient:
    return FakeOpenAIClient()
