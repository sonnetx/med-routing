from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import LRUCache

from .config import get_settings


def _key(*, prompt_hash: str, model: str, temperature: float, n: int) -> str:
    return f"{prompt_hash}|{model}|t={temperature:.2f}|n={n}"


def hash_messages(messages: list[dict[str, str]]) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


class CompletionCache:
    """LRU cache keyed by (messages, model, temperature, n).

    Used so semantic-entropy and self-consistency don't pay for samples twice
    during a live demo replay, and so a router can read the weak response without
    re-issuing the call when escalation is decided.
    """

    def __init__(self, maxsize: int | None = None) -> None:
        s = get_settings()
        self._store: LRUCache[str, Any] = LRUCache(maxsize=maxsize or s.cache_size)

    def get(self, *, messages: list[dict[str, str]], model: str, temperature: float, n: int) -> Any | None:
        return self._store.get(_key(prompt_hash=hash_messages(messages), model=model, temperature=temperature, n=n))

    def set(self, *, messages: list[dict[str, str]], model: str, temperature: float, n: int, value: Any) -> None:
        self._store[_key(prompt_hash=hash_messages(messages), model=model, temperature=temperature, n=n)] = value

    def __len__(self) -> int:  # for test introspection
        return len(self._store)
