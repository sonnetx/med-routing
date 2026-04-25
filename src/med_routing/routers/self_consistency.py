from __future__ import annotations

import re
from collections import Counter

from ..llm.openai_client import Completion
from ..config import get_settings
from .base import RouterScore, Sampler, UncertaintyRouter

# Tighter than scoring.parse_letter: this is used to *detect* whether a sample
# is an MCQ letter response or free-form. Requires the letter to be uppercase
# and bounded by punctuation/whitespace/EOS — so "Asthma is a chronic..." won't
# spuriously match the standalone "a" in the middle and trick the router into
# letter-clustering long-form answers.
_LETTER_RE = re.compile(r"(?:^|[\s\(\[])([A-D])(?=[\s.)\],!?:;]|$)")


def extract_letter(text: str) -> str | None:
    m = _LETTER_RE.search(text or "")
    return m.group(1) if m else None


def _normalize_for_match(text: str) -> str:
    """Coarse string normalization: lowercase, collapse whitespace, strip
    common boilerplate. Used as a fallback equivalence key on free-form
    answers when no MCQ letter is present."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t[:200]  # cap length so trivial trailing differences don't split clusters


class SelfConsistencyRouter(UncertaintyRouter):
    """Sample N weak generations; uncertainty = 1 - (modal answer count / N).

    On MCQ inputs the equivalence key is the extracted A/B/C/D letter. On
    free-form inputs (no letter parseable) we fall back to a normalized full-
    text match, which is a much weaker signal — for free-form, prefer
    semantic_entropy.
    """

    name = "self_consistency"

    async def score(self, *, messages, weak: Completion, sampler: Sampler) -> RouterScore:
        s = get_settings()
        samples = await sampler(n=s.sample_n, temperature=s.sample_temperature)
        if not samples:
            return RouterScore(score=1.0, extras={"reason": "no_samples"})

        letters = [extract_letter(c.text) for c in samples]
        parsed_letters = [letter for letter in letters if letter is not None]

        # Letter mode only when a clear majority of samples look like an MCQ
        # answer. A stray "A" as a word in a long sentence shouldn't tip us
        # into letter-clustering free-form responses.
        if len(parsed_letters) >= max(2, int(0.6 * len(samples))):
            counts = Counter(parsed_letters)
            top, top_count = counts.most_common(1)[0]
            score = 1.0 - top_count / len(parsed_letters)
            return RouterScore(
                score=score,
                extras={
                    "mode": "letter",
                    "letters": parsed_letters,
                    "modal": top,
                    "modal_count": top_count,
                    "n": len(samples),
                },
            )

        # Free-form fallback: cluster by normalized exact match.
        keys = [_normalize_for_match(c.text) for c in samples]
        counts = Counter(keys)
        _, top_count = counts.most_common(1)[0]
        score = 1.0 - top_count / len(keys)
        return RouterScore(
            score=score,
            extras={
                "mode": "string_match",
                "modal_count": top_count,
                "n": len(samples),
                "n_unique": len(counts),
                "warning": "free-form fallback; semantic_entropy gives a stronger signal",
            },
        )
