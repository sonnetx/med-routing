"""Company-facing dataset workflow: upload questions, generate ground truth,
run eval sweep across routers and thresholds, return a Pareto-frontier report.

Design note on efficiency: a naive sweep would call the cascade once per
(question, router, threshold) combination, which is N*R*T API rounds. Instead
this module pre-computes weak/strong answers once per question, judges each
once, and then varies thresholds purely in-memory. That makes a 25-question
sweep across 2 routers × 6 thresholds cost ~30 weak calls + ~25 strong calls +
~300 sampler calls + ~50 judge calls — and almost all of those hit
CompletionCache on demo replay.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .cache import CompletionCache
from .config import cost_usd, get_settings
from .eval.scoring import score_freeform_with_judge
from .llm.openai_client import OpenAIClient
from .routers.base import UncertaintyRouter

DatasetStatus = Literal["uploaded", "generating", "ready", "evaluating", "evaluated"]


@dataclass
class DatasetRow:
    qid: str
    question: str
    ground_truth: str | None = None
    subject: str | None = None
    difficulty: str | None = None


@dataclass
class Dataset:
    id: str
    name: str
    rows: list[DatasetRow]
    status: DatasetStatus = "uploaded"
    has_ground_truth_on_upload: bool = False
    generation_model: str | None = None
    created_at: float = field(default_factory=time.time)
    last_report: dict[str, Any] | None = None


# ---------- CSV parsing ----------

REQUIRED_COL = "question"


def parse_csv(content: bytes | str, name_hint: str = "") -> list[DatasetRow]:
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None or REQUIRED_COL not in reader.fieldnames:
        raise ValueError(
            f"CSV missing required column 'question'. Found: {reader.fieldnames}"
        )
    rows: list[DatasetRow] = []
    for i, raw in enumerate(reader, start=1):
        q = (raw.get("question") or "").strip()
        if not q:
            continue
        rows.append(
            DatasetRow(
                qid=str(raw.get("qid") or i),
                question=q,
                ground_truth=(raw.get("ground_truth") or None),
                subject=(raw.get("subject") or None),
                difficulty=(raw.get("difficulty") or None),
            )
        )
    return rows


# ---------- Cost estimation (uses pricing from config) ----------

# Loose token estimate: 1 token ≈ 4 chars of English. Good enough for a UI
# pre-flight estimate; the actual /metrics counters use real reported usage.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# Generation model pricing — we only have OpenAI rates locally, but the UI
# wants to compare across providers. Snapshot of public list prices in USD per
# 1M tokens as of April 2026; treat as estimates only.
GEN_PRICING = {
    "claude-opus-4-7": {"prompt": 15.0, "completion": 75.0},
    "gpt-5": {"prompt": 5.0, "completion": 20.0},
    "claude-sonnet-4-6": {"prompt": 3.0, "completion": 15.0},
}


def estimate_generation_cost(rows: list[DatasetRow], model: str, expected_answer_tokens: int = 200) -> float:
    p = GEN_PRICING.get(model, {"prompt": 5.0, "completion": 20.0})
    total = 0.0
    for r in rows:
        prompt_t = estimate_tokens(r.question) + 30  # plus system overhead
        total += (prompt_t * p["prompt"] + expected_answer_tokens * p["completion"]) / 1_000_000
    return total


# ---------- Fake generation: load cached "Opus" answers from disk ----------

DEMO_DATASET_NAME = "clinical_eval_25"


def load_cached_opus_answers() -> dict[str, str]:
    s = get_settings()
    path = Path(s.demo_data_dir) / f"{DEMO_DATASET_NAME}.opus.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    return blob.get("answers", {})


async def fake_generate(dataset: Dataset, sleep_s: float = 5.0, store: "DatasetStore | None" = None) -> None:
    """Simulate a frontier-model ground-truth generation pass. Loads the cached
    Opus answers keyed by qid; if a row's qid is missing from the cache, leaves
    its ground_truth as a stub so the demo doesn't silently mis-score."""
    dataset.status = "generating"
    answers = load_cached_opus_answers()
    # Sleep proportional to row count to feel real, capped at sleep_s.
    await asyncio.sleep(min(sleep_s, max(1.0, 0.15 * len(dataset.rows))))
    for r in dataset.rows:
        if not r.ground_truth:
            r.ground_truth = answers.get(
                r.qid,
                "[Demo mode: no cached answer for this question; ground truth not generated.]",
            )
    dataset.generation_model = "claude-opus-4-7"
    dataset.status = "ready"
    if store is not None:
        store.save(dataset)


# ---------- Judge wrapper with on-disk cache ----------


class CachedJudge:
    """Memoizes (question, predicted_answer) -> bool.

    Cache key uses sha1 of the predicted answer so two routers/thresholds that
    produce identical outputs are judged once. Optionally persisted to disk so
    eval replays after a container restart return instantly.
    """

    def __init__(self, *, client: OpenAIClient, model: str) -> None:
        self._client = client
        self._model = model
        self._cache: dict[str, bool] = {}

    @staticmethod
    def _key(question: str, predicted: str) -> str:
        h = hashlib.sha1((question + "|" + predicted).encode("utf-8")).hexdigest()
        return h

    def load(self, path: Path | str) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        try:
            blob = json.loads(p.read_text())
            self._cache.update({k: bool(v) for k, v in blob.items()})
            return len(blob)
        except Exception:
            return 0

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(json.dumps(self._cache, ensure_ascii=False))
        except Exception:
            pass

    async def judge(self, *, question: str, predicted: str, reference: str) -> bool:
        k = self._key(question, predicted)
        if k in self._cache:
            return self._cache[k]
        ok = await score_freeform_with_judge(
            question=question,
            predicted=predicted,
            reference=reference,
            judge_client=self._client,
            judge_model=self._model,
        )
        self._cache[k] = ok
        return ok


# ---------- Eval sweep ----------


@dataclass
class _PerQuestion:
    """Pre-computed per-question values that don't depend on threshold."""

    qid: str
    weak_answer: str
    strong_answer: str
    weak_cost: float
    strong_cost: float
    weak_correct: bool
    strong_correct: bool
    # router_name -> uncertainty score for this question's weak answer
    router_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    dataset_id: str
    n: int
    routers: list[str]
    thresholds: list[float]
    points: list[dict[str, Any]]
    pareto_indices: list[int]
    recommended: dict[str, Any] | None
    weak_only: dict[str, Any]
    strong_only: dict[str, Any]
    judge_model: str
    completed_at: float = field(default_factory=time.time)


def _pareto_front(points: list[dict[str, Any]]) -> list[int]:
    """Return indices of Pareto-optimal points (low cost, high accuracy)."""
    if not points:
        return []
    order = sorted(range(len(points)), key=lambda i: (points[i]["cost_usd"], -points[i]["accuracy"]))
    front: list[int] = []
    best_acc = -1.0
    for i in order:
        if points[i]["accuracy"] > best_acc:
            front.append(i)
            best_acc = points[i]["accuracy"]
    return front


async def _precompute_one(
    *,
    row: DatasetRow,
    client: OpenAIClient,
    cache: CompletionCache,
    routers: dict[str, UncertaintyRouter],
    judge: CachedJudge,
) -> _PerQuestion:
    """Run weak + strong + each router scoring + judge, all once."""
    s = get_settings()
    messages = [{"role": "user", "content": row.question}]

    # Weak call (deterministic, with logprobs so predictive_entropy could see them
    # if we ever included that router in the sweep).
    weak_cached = cache.get(messages=messages, model=s.weak_model, temperature=0.0, n=1)
    if weak_cached is None:
        comps = await client.complete(
            model=s.weak_model,
            messages=messages,
            tier="weak",
            temperature=0.0,
            max_tokens=256,
            logprobs=True,
            top_logprobs=5,
        )
        weak_cached = comps[0]
        cache.set(messages=messages, model=s.weak_model, temperature=0.0, n=1, value=weak_cached)
    weak = weak_cached

    # Strong call.
    strong_cached = cache.get(messages=messages, model=s.strong_model, temperature=0.0, n=1)
    if strong_cached is None:
        comps = await client.complete(
            model=s.strong_model,
            messages=messages,
            tier="strong",
            temperature=0.0,
            max_tokens=256,
        )
        strong_cached = comps[0]
        cache.set(messages=messages, model=s.strong_model, temperature=0.0, n=1, value=strong_cached)
    strong = strong_cached

    # Sampler shared by routers that need it (e.g. semantic_entropy_embed).
    async def sampler(*, n: int, temperature: float):
        cached = cache.get(messages=messages, model=s.weak_model, temperature=temperature, n=n)
        if cached is not None:
            return cached
        samples = await client.complete(
            model=s.weak_model,
            messages=messages,
            tier="weak",
            temperature=temperature,
            max_tokens=256,
            n=n,
        )
        cache.set(messages=messages, model=s.weak_model, temperature=temperature, n=n, value=samples)
        return samples

    router_scores: dict[str, float] = {}
    for name, router in routers.items():
        try:
            rs = await router.score(messages=messages, weak=weak, sampler=sampler)
            router_scores[name] = rs.score
        except Exception as exc:
            # Demo robustness: if a router blows up, mark its score as 1.0 so it
            # always escalates rather than crashing the whole sweep.
            router_scores[name] = 1.0

    reference = row.ground_truth or ""
    weak_correct = await judge.judge(question=row.question, predicted=weak.text, reference=reference) if reference else False
    strong_correct = await judge.judge(question=row.question, predicted=strong.text, reference=reference) if reference else False

    return _PerQuestion(
        qid=row.qid,
        weak_answer=weak.text,
        strong_answer=strong.text,
        weak_cost=weak.cost,
        strong_cost=strong.cost,
        weak_correct=weak_correct,
        strong_correct=strong_correct,
        router_scores=router_scores,
    )


async def run_eval_sweep(
    *,
    dataset: Dataset,
    client: OpenAIClient,
    cache: CompletionCache,
    routers: dict[str, UncertaintyRouter],
    router_names: list[str],
    thresholds: list[float],
    judge_model: str,
    store: "DatasetStore | None" = None,
    judge_cache_path: Path | str | None = None,
) -> EvalReport:
    """Sweep thresholds for each requested router and return a Pareto report."""
    if not dataset.rows:
        raise ValueError("Empty dataset")
    sweep_routers = {n: routers[n] for n in router_names if n in routers}
    if not sweep_routers:
        raise ValueError(f"None of the requested routers are registered: {router_names}")

    judge = CachedJudge(client=client, model=judge_model)
    if judge_cache_path is not None:
        judge.load(judge_cache_path)
    dataset.status = "evaluating"

    # Pre-compute everything once. Run sequentially to avoid hammering the API
    # with too many concurrent requests during the demo.
    precomputed: list[_PerQuestion] = []
    for row in dataset.rows:
        pq = await _precompute_one(
            row=row, client=client, cache=cache,
            routers=sweep_routers, judge=judge,
        )
        precomputed.append(pq)

    # In-memory threshold sweep: cheap.
    points: list[dict[str, Any]] = []
    for router_name in router_names:
        if router_name not in sweep_routers:
            continue
        for t in thresholds:
            n_correct = 0
            n_escalated = 0
            total_cost = 0.0
            for pq in precomputed:
                score = pq.router_scores.get(router_name, 1.0)
                escalated = score >= t
                if escalated:
                    n_escalated += 1
                    n_correct += int(pq.strong_correct)
                    total_cost += pq.weak_cost + pq.strong_cost
                else:
                    n_correct += int(pq.weak_correct)
                    total_cost += pq.weak_cost
            n = len(precomputed)
            points.append({
                "router": router_name,
                "threshold": round(t, 3),
                "accuracy": n_correct / n if n else 0.0,
                "escalation_rate": n_escalated / n if n else 0.0,
                "cost_usd": total_cost,
                "n": n,
            })

    # Baselines: always-weak and always-strong on the same data.
    weak_only = {
        "label": "weak only (gpt-4o-mini)",
        "accuracy": sum(1 for pq in precomputed if pq.weak_correct) / len(precomputed),
        "cost_usd": sum(pq.weak_cost for pq in precomputed),
    }
    strong_only = {
        "label": "strong only (gpt-4o)",
        "accuracy": sum(1 for pq in precomputed if pq.strong_correct) / len(precomputed),
        "cost_usd": sum(pq.strong_cost for pq in precomputed),
    }

    pareto = _pareto_front(points)

    # Recommend: highest accuracy on the Pareto frontier whose cost is at most
    # 1.5x the weak-only baseline. If none qualifies, recommend the cheapest
    # frontier point that beats weak-only on accuracy.
    recommended = None
    if pareto:
        cost_ceiling = max(weak_only["cost_usd"] * 1.5, weak_only["cost_usd"] + 1e-6)
        eligible = [
            i for i in pareto
            if points[i]["cost_usd"] <= cost_ceiling and points[i]["accuracy"] >= weak_only["accuracy"]
        ]
        if eligible:
            best = max(eligible, key=lambda i: points[i]["accuracy"])
            recommended = {**points[best], "index": best, "reason": "highest accuracy under 1.5× weak-only cost"}
        else:
            # Fall back: cheapest frontier point that beats weak-only.
            beats = [i for i in pareto if points[i]["accuracy"] > weak_only["accuracy"]]
            if beats:
                cheapest = min(beats, key=lambda i: points[i]["cost_usd"])
                recommended = {**points[cheapest], "index": cheapest, "reason": "cheapest config that beats weak-only accuracy"}

    report = EvalReport(
        dataset_id=dataset.id,
        n=len(precomputed),
        routers=router_names,
        thresholds=thresholds,
        points=points,
        pareto_indices=pareto,
        recommended=recommended,
        weak_only=weak_only,
        strong_only=strong_only,
        judge_model=judge_model,
    )
    dataset.last_report = {
        "n": report.n,
        "routers": report.routers,
        "thresholds": report.thresholds,
        "points": report.points,
        "pareto_indices": report.pareto_indices,
        "recommended": report.recommended,
        "weak_only": report.weak_only,
        "strong_only": report.strong_only,
        "judge_model": report.judge_model,
        "completed_at": report.completed_at,
    }
    dataset.status = "evaluated"
    if store is not None:
        store.save(dataset)
    if judge_cache_path is not None:
        judge.save(judge_cache_path)
    return report


# ---------- Disk persistence ----------


def _dataset_to_json(ds: Dataset) -> dict[str, Any]:
    return {
        "id": ds.id,
        "name": ds.name,
        "status": ds.status,
        "has_ground_truth_on_upload": ds.has_ground_truth_on_upload,
        "generation_model": ds.generation_model,
        "created_at": ds.created_at,
        "rows": [
            {
                "qid": r.qid, "question": r.question, "ground_truth": r.ground_truth,
                "subject": r.subject, "difficulty": r.difficulty,
            }
            for r in ds.rows
        ],
        "last_report": ds.last_report,
    }


def _dataset_from_json(blob: dict[str, Any]) -> Dataset:
    rows = [
        DatasetRow(
            qid=r.get("qid", ""), question=r.get("question", ""),
            ground_truth=r.get("ground_truth"), subject=r.get("subject"),
            difficulty=r.get("difficulty"),
        )
        for r in blob.get("rows", [])
    ]
    return Dataset(
        id=blob["id"],
        name=blob.get("name", "(unnamed)"),
        rows=rows,
        status=blob.get("status", "uploaded"),
        has_ground_truth_on_upload=bool(blob.get("has_ground_truth_on_upload", False)),
        generation_model=blob.get("generation_model"),
        created_at=blob.get("created_at", time.time()),
        last_report=blob.get("last_report"),
    )


# ---------- In-memory dataset registry with disk persistence ----------


class DatasetStore:
    """In-memory dataset registry with optional auto-persistence.

    When `persist_dir` is set, every mutation (create/generate/evaluate)
    serializes the affected dataset to `<persist_dir>/<id>.json`. On startup,
    `load_from_disk()` rehydrates everything previously saved.
    """

    def __init__(self, persist_dir: Path | str | None = None) -> None:
        self._datasets: dict[str, Dataset] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, ds_id: str) -> Path | None:
        if self._persist_dir is None:
            return None
        return self._persist_dir / f"{ds_id}.json"

    def save(self, ds: Dataset) -> None:
        path = self._path_for(ds.id)
        if path is None:
            return
        try:
            path.write_text(json.dumps(_dataset_to_json(ds), ensure_ascii=False, default=str))
        except Exception:
            # Persistence failures should never break the request flow.
            pass

    def load_from_disk(self) -> int:
        """Rehydrate persisted datasets. Returns count loaded."""
        if self._persist_dir is None or not self._persist_dir.is_dir():
            return 0
        n = 0
        for fp in self._persist_dir.glob("*.json"):
            try:
                blob = json.loads(fp.read_text())
                ds = _dataset_from_json(blob)
                self._datasets[ds.id] = ds
                n += 1
            except Exception:
                # A corrupt file shouldn't kill startup; leave it on disk for inspection.
                continue
        return n

    def create_from_csv(self, *, name: str, content: bytes | str, force_new: bool = False) -> Dataset:
        rows = parse_csv(content, name_hint=name)
        # Deterministic ID = hash of (qid, question, ground_truth) content. Same CSV
        # content lands on the same dataset, which means the persisted report (and
        # any cached completions) is reused automatically — no re-eval needed.
        # When `force_new=True` we skip dedup and create a fresh dataset; useful for
        # the "generate ground truth" demo flow where we want the steps shown each time.
        if force_new:
            ds_id = uuid.uuid4().hex[:10]
        else:
            sig = json.dumps(
                [(r.qid, r.question, r.ground_truth or "") for r in rows],
                sort_keys=True, ensure_ascii=False,
            ).encode("utf-8")
            ds_id = hashlib.sha1(sig).hexdigest()[:10]
            existing = self._datasets.get(ds_id)
            if existing is not None:
                return existing

        has_gt = any(r.ground_truth for r in rows)
        ds = Dataset(
            id=ds_id, name=name, rows=rows,
            has_ground_truth_on_upload=has_gt,
            status="ready" if has_gt else "uploaded",
            generation_model="ground_truth_provided" if has_gt else None,
        )
        self._datasets[ds_id] = ds
        self.save(ds)
        return ds

    def get(self, ds_id: str) -> Dataset | None:
        return self._datasets.get(ds_id)

    def list(self) -> list[Dataset]:
        return sorted(self._datasets.values(), key=lambda d: -d.created_at)

    def delete(self, ds_id: str) -> bool:
        ds = self._datasets.pop(ds_id, None)
        if ds is None:
            return False
        path = self._path_for(ds_id)
        if path is not None and path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        return True
