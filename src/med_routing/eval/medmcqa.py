from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

PROMPT_TEMPLATE = (
    "You are a careful medical expert answering a multiple-choice question.\n\n"
    "Question: {question}\n"
    "A) {opa}\n"
    "B) {opb}\n"
    "C) {opc}\n"
    "D) {opd}\n\n"
    "Reply with a single letter (A, B, C, or D) and nothing else."
)

LETTERS = ("A", "B", "C", "D")


@dataclass
class MedMCQAItem:
    qid: str
    question: str
    options: tuple[str, str, str, str]
    answer: str  # 'A'..'D'
    subject: str

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    question=self.question,
                    opa=self.options[0],
                    opb=self.options[1],
                    opc=self.options[2],
                    opd=self.options[3],
                ),
            }
        ]


def load_medmcqa(split: str = "validation", limit: int | None = None) -> Iterator[MedMCQAItem]:
    """Stream MedMCQA items. Requires `datasets` and network on first use."""
    from datasets import load_dataset  # local import — heavy

    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        cop = row.get("cop")
        if cop is None or cop < 0 or cop > 3:
            continue
        yield MedMCQAItem(
            qid=str(row.get("id", i)),
            question=row["question"],
            options=(row["opa"], row["opb"], row["opc"], row["opd"]),
            answer=LETTERS[cop],
            subject=row.get("subject_name", ""),
        )


def iter_pairs(items: Iterable[MedMCQAItem]) -> Iterator[tuple[MedMCQAItem, list[dict[str, str]]]]:
    for it in items:
        yield it, it.to_messages()
