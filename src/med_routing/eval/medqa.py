from __future__ import annotations

from typing import Any, Iterator

from .medmcqa import MedMCQAItem  # reuse the dataclass — same A/B/C/D shape

LETTERS = ("A", "B", "C", "D")


def _coerce_options(opts: Any) -> tuple[str, str, str, str] | None:
    """MedQA HF distributions vary: dict, list-of-dicts, or list-of-strings."""
    if isinstance(opts, dict):
        return tuple(str(opts.get(k, "")) for k in LETTERS)  # type: ignore[return-value]
    if isinstance(opts, list) and opts:
        if isinstance(opts[0], dict) and "key" in opts[0] and "value" in opts[0]:
            d = {str(x["key"]): str(x["value"]) for x in opts}
            return tuple(d.get(k, "") for k in LETTERS)  # type: ignore[return-value]
        if isinstance(opts[0], str) and len(opts) >= 4:
            return tuple(opts[:4])  # type: ignore[return-value]
    return None


def _row_to_item(row: dict[str, Any], idx: int) -> MedMCQAItem | None:
    opts = _coerce_options(row.get("options"))
    if opts is None:
        return None
    ans = row.get("answer_idx") or row.get("answer_letter")
    if ans not in LETTERS:
        return None
    return MedMCQAItem(
        qid=f"medqa-{row.get('id', idx)}",
        question=str(row.get("question", "")),
        options=opts,
        answer=str(ans),
        subject=str(row.get("meta_info") or "medqa-usmle"),
    )


def load_medqa(split: str = "test", limit: int | None = None) -> Iterator[MedMCQAItem]:
    """USMLE-style 4-option medical MCQs from MedQA (Apache 2.0).

    Harder than MedMCQA: GPT-4o-mini sits around ~60-65% vs ~75-80% on MedMCQA.
    The extra failures are typically multi-step reasoning questions, where
    sample-based uncertainty signals (self_consistency, semantic_entropy)
    actually start to grip.
    """
    from datasets import load_dataset  # local import — heavy

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        item = _row_to_item(row, i)
        if item is not None:
            yield item
