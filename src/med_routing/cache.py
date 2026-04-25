from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
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
    during a live demo replay, and so a router can read the weak response
    without re-issuing the call when escalation is decided. Optionally
    persists to disk so cache survives container restarts.
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

    # ---------- Disk persistence ----------
    # Values can be Completion dataclasses (with a TokenLogprob list) or plain
    # lists thereof. We round-trip via dict <-> dataclass since dataclasses
    # aren't directly JSON-serializable.

    @staticmethod
    def _serialize(v: Any) -> Any:
        if is_dataclass(v):
            return {"__dc__": v.__class__.__name__, **asdict(v)}
        if isinstance(v, list):
            return [CompletionCache._serialize(x) for x in v]
        return v

    @staticmethod
    def _deserialize(v: Any) -> Any:
        # Local import to avoid circular dep at module load.
        from .llm.openai_client import Completion, TokenLogprob

        if isinstance(v, list):
            return [CompletionCache._deserialize(x) for x in v]
        if isinstance(v, dict) and "__dc__" in v:
            kind = v["__dc__"]
            data = {k: val for k, val in v.items() if k != "__dc__"}
            if kind == "Completion":
                lps = data.get("logprobs")
                if isinstance(lps, list):
                    data["logprobs"] = [
                        TokenLogprob(token=lp["token"], logprob=lp["logprob"], top=[tuple(t) for t in lp.get("top", [])])
                        for lp in lps
                    ]
                return Completion(**data)
            if kind == "TokenLogprob":
                return TokenLogprob(**data)
        return v

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob = {k: CompletionCache._serialize(v) for k, v in self._store.items()}
            p.write_text(json.dumps(blob, ensure_ascii=False, default=str))
        except Exception:
            pass

    def load(self, path: Path | str) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        try:
            blob = json.loads(p.read_text())
        except Exception:
            return 0
        n = 0
        for k, v in blob.items():
            try:
                self._store[k] = CompletionCache._deserialize(v)
                n += 1
            except Exception:
                continue
        return n
