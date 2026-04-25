from __future__ import annotations

import re

_LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def parse_letter(text: str) -> str | None:
    m = _LETTER_RE.search(text)
    return m.group(1).upper() if m else None


def is_correct(predicted_text: str, gold_letter: str) -> bool:
    p = parse_letter(predicted_text)
    return p is not None and p == gold_letter.upper()
