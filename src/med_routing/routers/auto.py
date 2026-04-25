from __future__ import annotations

import re
from typing import ClassVar

from ..config import get_settings
from .base import RouterScore, Sampler, UncertaintyRouter


_MCQ_OPTIONS_RE = re.compile(r"^\s*[ABCD]\)", re.MULTILINE)
_MCQ_INSTRUCTION_RE = re.compile(r"single\s+letter|reply with .*letter", re.IGNORECASE)


def detect_format(messages: list[dict]) -> str:
    """Return 'mcq' or 'free_form' from a chat-completion-style messages list.

    Heuristic: if the last user message has either four A)/B)/C)/D) option lines
    OR an explicit "single letter" instruction, treat it as MCQ. Otherwise
    free-form. Conservative — false positives on free-form prompts that happen
    to mention "letter" are fine because MCQ-tuned routers still work on free
    text (they just degenerate to lower-quality signal).
    """
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = str(m.get("content") or "")
            break
    if not last_user:
        return "free_form"
    if _MCQ_INSTRUCTION_RE.search(last_user):
        return "mcq"
    options = _MCQ_OPTIONS_RE.findall(last_user)
    if len(set(options)) >= 4:
        return "mcq"
    return "free_form"


class AutoRouter(UncertaintyRouter):
    """Format-aware meta-router. Picks an appropriate sub-router from the
    available registered routers based on the prompt's format.

    - mcq      → predictive_entropy if available (zero extra cost), else
                 self_consistency, else self_reported.
    - free_form → semantic_entropy if available (NLI handles paraphrases),
                  else self_consistency (string-match fallback), else
                  self_reported.

    The chosen sub-router's calibrated threshold is returned via
    `threshold_override` so the cascade compares against the right cutoff.
    """

    name: ClassVar[str] = "auto"

    # In preference order. First match that's actually registered wins.
    _MCQ_PREFERENCE = ("predictive_entropy", "self_consistency", "self_reported")
    _FREEFORM_PREFERENCE = ("semantic_entropy", "self_consistency", "self_reported")

    def __init__(self, sub_routers: dict[str, UncertaintyRouter]) -> None:
        # Avoid an infinite loop if `auto` is somehow handed itself.
        self._sub = {k: v for k, v in sub_routers.items() if k != self.name}

    def _pick(self, fmt: str) -> UncertaintyRouter | None:
        order = self._FREEFORM_PREFERENCE if fmt == "free_form" else self._MCQ_PREFERENCE
        for n in order:
            if n in self._sub:
                return self._sub[n]
        return next(iter(self._sub.values()), None)

    async def score(self, *, messages, weak, sampler: Sampler) -> RouterScore:
        fmt = detect_format(messages)
        sub = self._pick(fmt)
        if sub is None:
            return RouterScore(score=1.0, extras={"reason": "no_sub_routers", "format": fmt})

        rs = await sub.score(messages=messages, weak=weak, sampler=sampler)
        s = get_settings()
        # The cascade will use this threshold instead of `auto`'s own default.
        rs.threshold_override = s.threshold_for(sub.name)
        rs.extras = {
            **rs.extras,
            "auto_format": fmt,
            "auto_router": sub.name,
            "auto_threshold_used": rs.threshold_override,
        }
        return rs
