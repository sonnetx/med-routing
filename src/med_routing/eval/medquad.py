from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


_PROMPT = (
    "You are a careful medical expert answering a consumer health question. "
    "Be accurate and concise — 1-3 sentences.\n\n"
    "Question: {question}"
)


@dataclass
class FreeFormItem:
    """Free-form medical Q&A item — no letter answer, scored by LLM-as-judge."""

    qid: str
    question: str
    reference_answer: str
    subject: str

    def to_messages(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": _PROMPT.format(question=self.question)}]


def _row_to_item(row: dict[str, Any], idx: int) -> FreeFormItem | None:
    q = row.get("question") or ""
    a = row.get("answer") or ""
    if not q or not a:
        return None
    return FreeFormItem(
        qid=str(row.get("id", idx)),
        question=str(q).strip(),
        reference_answer=str(a).strip(),
        subject=str(row.get("question_type") or row.get("category") or "medquad"),
    )


def load_medquad(split: str = "train", limit: int | None = None) -> Iterator[FreeFormItem]:
    """Free-form medical Q&A from MedQuAD (NLM consumer health, public domain).

    Tagged with question_type (treatment, symptoms, prevention, ...) which we
    use as the `subject` label for the per-subject Grafana panel. Answers are
    typically 50-300 words, so we score with LLM-as-judge instead of exact match.
    """
    from datasets import load_dataset  # heavy

    ds = load_dataset("lavita/MedQuAD", split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        item = _row_to_item(row, i)
        if item is not None:
            yield item
