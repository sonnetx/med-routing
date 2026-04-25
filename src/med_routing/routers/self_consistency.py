from __future__ import annotations

import re
from collections import Counter

from ..llm.openai_client import Completion
from ..config import get_settings
from .base import RouterScore, Sampler, UncertaintyRouter

_LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def extract_letter(text: str) -> str | None:
    m = _LETTER_RE.search(text)
    return m.group(1).upper() if m else None


class SelfConsistencyRouter(UncertaintyRouter):
    """Sample N weak generations; uncertainty = 1 - (modal answer count / N)."""

    name = "self_consistency"

    async def score(self, *, messages, weak: Completion, sampler: Sampler) -> RouterScore:
        s = get_settings()
        samples = await sampler(n=s.sample_n, temperature=s.sample_temperature)

        letters = [extract_letter(c.text) for c in samples]
        letters = [letter for letter in letters if letter is not None]
        if not letters:
            return RouterScore(score=1.0, extras={"samples": [], "parsed": 0})

        counts = Counter(letters)
        top, top_count = counts.most_common(1)[0]
        score = 1.0 - top_count / len(letters)
        return RouterScore(
            score=score,
            extras={
                "samples": [c.text for c in samples],
                "letters": letters,
                "modal": top,
                "modal_count": top_count,
                "n": len(samples),
            },
        )
