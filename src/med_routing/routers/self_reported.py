from __future__ import annotations

import re

from ..llm.openai_client import Completion, OpenAIClient
from ..config import get_settings
from .base import RouterScore, Sampler, UncertaintyRouter

_INT_RE = re.compile(r"\b(\d{1,3})\b")


def parse_confidence(text: str) -> int | None:
    m = _INT_RE.search(text)
    if not m:
        return None
    v = int(m.group(1))
    return max(0, min(100, v))


class SelfReportedRouter(UncertaintyRouter):
    name = "self_reported"

    def __init__(self, client: OpenAIClient) -> None:
        self._client = client

    async def score(self, *, messages, weak: Completion, sampler: Sampler) -> RouterScore:
        followup = list(messages) + [
            {"role": "assistant", "content": weak.text},
            {
                "role": "user",
                "content": (
                    "On a scale from 0 to 100, how confident are you that your answer is correct? "
                    "Reply with just the integer."
                ),
            },
        ]
        s = get_settings()
        completions = await self._client.complete(
            model=s.weak_model,
            messages=followup,
            tier="weak",
            temperature=0.0,
            max_tokens=8,
        )
        raw = completions[0].text if completions else ""
        conf = parse_confidence(raw)
        if conf is None:
            return RouterScore(score=1.0, extras={"raw": raw, "parsed": None})
        return RouterScore(score=1.0 - conf / 100.0, extras={"raw": raw, "confidence": conf})
