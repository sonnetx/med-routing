from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class NLIScorer(Protocol):
    def entails(self, premise: str, hypothesis: str) -> bool: ...


class DebertaNLIScorer:
    """Lazy wrapper around DeBERTa-v3-MNLI. Loaded once at startup.

    Heavy import (transformers + torch) is intentionally inside __init__ so the
    main server process doesn't pay the cost when ENABLE_NLI=false.
    """

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        # MNLI label order: 0=contradiction, 1=neutral, 2=entailment for DeBERTa-v3-MNLI.
        self._entail_idx = 2

    def entails(self, premise: str, hypothesis: str) -> bool:
        torch = self._torch
        inputs = self._tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        pred = int(logits.argmax().item())
        return pred == self._entail_idx


@lru_cache(maxsize=1)
def get_nli_scorer(model_name: str) -> NLIScorer:
    return DebertaNLIScorer(model_name)
