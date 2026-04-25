from __future__ import annotations

import re
from typing import Any

_LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def parse_letter(text: str) -> str | None:
    m = _LETTER_RE.search(text)
    return m.group(1).upper() if m else None


def is_correct(predicted_text: str, gold_letter: str) -> bool:
    p = parse_letter(predicted_text)
    return p is not None and p == gold_letter.upper()


_JUDGE_PROMPT = (
    "You are grading an AI's answer against a reference answer for a consumer "
    "health question.\n\n"
    "Question: {question}\n\n"
    "REFERENCE answer:\n{reference}\n\n"
    "AI answer:\n{predicted}\n\n"
    "Is the AI answer medically consistent with the reference? "
    "Paraphrasing is acceptable; the AI does not need to copy exact wording, "
    "but it must not contradict the reference and must address the question. "
    "Reply with a single token: CORRECT or WRONG."
)


async def score_freeform_with_judge(
    *,
    question: str,
    predicted: str,
    reference: str,
    judge_client: Any,
    judge_model: str,
    max_chars: int = 2000,
) -> bool:
    """LLM-as-judge scoring for free-form answers.

    Truncates each side to `max_chars` to keep the judge prompt cheap. The
    judge call uses tier="strong" so its cost rolls into the strong-tier
    accounting — visible in Grafana but not attributed to either router.
    """
    if not predicted.strip() or not reference.strip():
        return False
    msgs = [
        {
            "role": "user",
            "content": _JUDGE_PROMPT.format(
                question=question[:1000],
                reference=reference[:max_chars],
                predicted=predicted[:max_chars],
            ),
        }
    ]
    try:
        completions = await judge_client.complete(
            model=judge_model,
            messages=msgs,
            tier="strong",
            temperature=0.0,
            max_tokens=4,
        )
    except Exception:
        return False
    txt = (completions[0].text if completions else "").strip().upper()
    return txt.startswith("CORRECT")
