from __future__ import annotations

import datetime as dt
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, TextIO


class AuditLogger:
    """Append-only JSONL audit log with daily rotation, intended for GDPR Art. 30
    records of processing. Stores prompt SHA only — never the raw prompt — so the
    log itself can be retained safely without containing PHI.

    Also keeps the most recent N rows in-memory so the demo UI can show them
    without parsing the file on every request.
    """

    def __init__(
        self,
        root: Path | None = None,
        recent_size: int = 200,
        store: Any | None = None,
    ) -> None:
        self.root = Path(root) if root else Path("audit")
        self.root.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO | None = None
        self._date: str | None = None
        self._lock = threading.Lock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_size)
        self._store = store  # optional Store; persisted across restarts

    @staticmethod
    def _today() -> str:
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    def _path_for(self, date: str) -> Path:
        return self.root / f"decisions-{date}.jsonl"

    def _ensure_open(self) -> TextIO:
        today = self._today()
        if self._date != today:
            if self._fp is not None:
                try:
                    self._fp.close()
                except Exception:
                    pass
            self._fp = self._path_for(today).open("a", encoding="utf-8", buffering=1)
            self._date = today
        return self._fp

    def log(self, row: dict[str, Any]) -> None:
        with self._lock:
            fp = self._ensure_open()
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self._recent.append(row)
        if self._store is not None:
            try:
                self._store.insert_decision(row)
            except Exception:
                # Storing should never break the request; the JSONL log is the
                # durable source of truth, the DB is a convenience replica.
                pass

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent)[-limit:]

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                finally:
                    self._fp = None
                    self._date = None
