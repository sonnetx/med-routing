from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    prompt_sha TEXT NOT NULL,
    router TEXT NOT NULL,
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    escalated INTEGER NOT NULL,
    tier_chain TEXT,
    final_tier_index INTEGER,
    weak_model TEXT,
    weak_processor TEXT,
    weak_region TEXT,
    weak_dpa_ref TEXT,
    strong_model TEXT,
    strong_processor TEXT,
    strong_region TEXT,
    final_model TEXT,
    final_region TEXT,
    home_region TEXT,
    regions_touched_json TEXT,
    cross_border INTEGER NOT NULL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    cost_usd REAL,
    counterfactual_usd REAL,
    latency_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_router ON decisions(router);
CREATE INDEX IF NOT EXISTS idx_decisions_xb ON decisions(cross_border);

CREATE TABLE IF NOT EXISTS eval_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    router TEXT NOT NULL,
    qid TEXT,
    subject TEXT,
    score REAL,
    escalated INTEGER NOT NULL,
    correct INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_router ON eval_rows(router);
CREATE INDEX IF NOT EXISTS idx_eval_ts ON eval_rows(ts);

CREATE TABLE IF NOT EXISTS runtime_thresholds (
    router TEXT PRIMARY KEY,
    threshold REAL NOT NULL,
    applied_at TEXT NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS threshold_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    router TEXT NOT NULL,
    threshold REAL NOT NULL,
    applied_at TEXT NOT NULL,
    reason TEXT
);
"""

DECISION_COLS = (
    "ts", "prompt_sha", "router", "score", "threshold", "escalated",
    "tier_chain", "final_tier_index",
    "weak_model", "weak_processor", "weak_region", "weak_dpa_ref",
    "strong_model", "strong_processor", "strong_region",
    "final_model", "final_region", "home_region", "regions_touched_json",
    "cross_border", "tokens_prompt", "tokens_completion",
    "cost_usd", "counterfactual_usd", "latency_ms",
)


def _ensure_columns(c: sqlite3.Connection) -> None:
    """Add tier_chain / final_tier_index to legacy DBs that pre-date 3-tier."""
    have = {row["name"] for row in c.execute("PRAGMA table_info(decisions)").fetchall()}
    if "tier_chain" not in have:
        c.execute("ALTER TABLE decisions ADD COLUMN tier_chain TEXT")
    if "final_tier_index" not in have:
        c.execute("ALTER TABLE decisions ADD COLUMN final_tier_index INTEGER")

EVAL_COLS = ("ts", "router", "qid", "subject", "score", "escalated", "correct")


class Store:
    """Lightweight SQLite store for audit decisions and eval rows.

    Single file, shippable. Use a fresh connection per call so we don't fight
    SQLite's threading model under FastAPI; with the WAL journal mode, this is
    fine for the kind of low-volume traffic a hackathon demo produces.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as c:
            c.executescript(SCHEMA)
            _ensure_columns(c)
            c.execute("PRAGMA journal_mode = WAL")
            # Bounded WAL: checkpoint into the main DB every 100 pages so a lost
            # WAL never strands more than a small batch of recent rows.
            c.execute("PRAGMA wal_autocheckpoint = 100")

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        c.row_factory = sqlite3.Row
        # synchronous is per-connection. FULL forces fsync on every commit so
        # writes survive a container SIGKILL or host crash, at a cost of ~5-10ms
        # per insert. Worth it — Docker Desktop bind mounts on Windows have
        # buffered writes that NORMAL durability does not flush in time.
        c.execute("PRAGMA synchronous = FULL")
        return c

    def insert_decision(self, row: dict[str, Any]) -> None:
        prepared = _prepare_decision(row)
        placeholders = ",".join("?" for _ in DECISION_COLS)
        cols = ",".join(DECISION_COLS)
        sql = f"INSERT INTO decisions ({cols}) VALUES ({placeholders})"
        with self._lock, self._connect() as c:
            c.execute(sql, [prepared[col] for col in DECISION_COLS])

    def insert_eval_row(
        self, *, router: str, score: float, escalated: bool, correct: bool,
        subject: str | None = None, qid: str | None = None,
    ) -> None:
        sql = (
            "INSERT INTO eval_rows (ts, router, qid, subject, score, escalated, correct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        with self._lock, self._connect() as c:
            c.execute(
                sql,
                (
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                    router, qid, subject, score, int(escalated), int(correct),
                ),
            )

    def query_decisions(
        self, *, limit: int = 100, since: str | None = None,
        router: str | None = None, cross_border_only: bool = False,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM decisions WHERE 1=1"
        params: list[Any] = []
        if since:
            sql += " AND ts >= ?"
            params.append(since)
        if router:
            sql += " AND router = ?"
            params.append(router)
        if cross_border_only:
            sql += " AND cross_border = 1"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as c:
            return [_row_to_dict(r) for r in c.execute(sql, params).fetchall()]

    def query_eval(
        self, *, limit: int = 1000, router: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM eval_rows WHERE 1=1"
        params: list[Any] = []
        if router:
            sql += " AND router = ?"
            params.append(router)
        if since:
            sql += " AND ts >= ?"
            params.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def set_threshold(self, *, router: str, threshold: float, reason: str = "") -> None:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock, self._connect() as c:
            c.execute(
                "INSERT INTO runtime_thresholds (router, threshold, applied_at, reason) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(router) DO UPDATE SET threshold=excluded.threshold, "
                "applied_at=excluded.applied_at, reason=excluded.reason",
                (router, float(threshold), ts, reason),
            )
            c.execute(
                "INSERT INTO threshold_history (router, threshold, applied_at, reason) "
                "VALUES (?, ?, ?, ?)",
                (router, float(threshold), ts, reason),
            )

    def get_thresholds(self) -> dict[str, float]:
        with self._connect() as c:
            return {
                r["router"]: float(r["threshold"])
                for r in c.execute("SELECT router, threshold FROM runtime_thresholds").fetchall()
            }

    def threshold_history(self, *, router: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM threshold_history"
        params: list[Any] = []
        if router:
            sql += " WHERE router = ?"
            params.append(router)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    def stats(self) -> dict[str, Any]:
        with self._connect() as c:
            n_dec = c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            n_eval = c.execute("SELECT COUNT(*) FROM eval_rows").fetchone()[0]
            n_xb = c.execute("SELECT COUNT(*) FROM decisions WHERE cross_border=1").fetchone()[0]
            by_router = {
                r["router"]: r["n"]
                for r in c.execute(
                    "SELECT router, COUNT(*) AS n FROM decisions GROUP BY router"
                ).fetchall()
            }
        return {
            "decisions": n_dec,
            "eval_rows": n_eval,
            "cross_border_decisions": n_xb,
            "decisions_by_router": by_router,
            "db_path": str(self.path),
        }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("regions_touched_json"):
        try:
            d["regions_touched"] = json.loads(d["regions_touched_json"])
        except Exception:
            d["regions_touched"] = []
    if d.get("tier_chain"):
        try:
            d["tier_chain"] = json.loads(d["tier_chain"])
        except Exception:
            d["tier_chain"] = []
    d["escalated"] = bool(d.get("escalated"))
    d["cross_border"] = bool(d.get("cross_border"))
    return d


def _prepare_decision(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise the dict produced by the cascade for SQLite insertion."""
    out: dict[str, Any] = {col: row.get(col) for col in DECISION_COLS}
    out["regions_touched_json"] = json.dumps(row.get("regions_touched") or [])
    chain = row.get("tier_chain")
    if isinstance(chain, list):
        out["tier_chain"] = json.dumps(chain, default=str)
    elif isinstance(chain, str):
        out["tier_chain"] = chain
    else:
        out["tier_chain"] = None
    out["escalated"] = int(bool(row.get("escalated")))
    out["cross_border"] = int(bool(row.get("cross_border")))
    return out


def iter_csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> Iterator[str]:
    """Yield CSV lines (header first) for streaming responses."""
    yield ",".join(columns) + "\n"
    for row in rows:
        cells = []
        for col in columns:
            v = row.get(col)
            if v is None:
                cells.append("")
            else:
                s = str(v).replace('"', '""')
                if "," in s or "\n" in s or '"' in s:
                    s = f'"{s}"'
                cells.append(s)
        yield ",".join(cells) + "\n"
