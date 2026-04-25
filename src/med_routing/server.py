from __future__ import annotations

import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .audit import AuditLogger
from .cache import CompletionCache
from .cascade import CascadeController
from .config import (
    clear_runtime_threshold,
    get_settings,
    runtime_overrides,
    set_runtime_threshold,
)
from .eval.aggregator import EvalAggregator
from .feedback import recommend_for_all
from .fhir import decision_to_audit_event, to_bundle
from .llm.openai_client import OpenAIClient
from .metrics import ACCURACY, REGISTRY
from .routers.registry import KNOWN_ROUTERS, build_routers
from .store import DECISION_COLS, EVAL_COLS, Store, iter_csv


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    router: str | None = Field(default=None, description="Override router; else X-Router header.")


class AccuracyUpdate(BaseModel):
    router: str
    accuracy: float


class Observation(BaseModel):
    router: str
    score: float
    escalated: bool
    correct: bool
    subject: str | None = None


class ThresholdApply(BaseModel):
    router: str
    threshold: float
    reason: str = "manual override"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    client = OpenAIClient()
    routers = build_routers(client)
    cache = CompletionCache()
    store = Store(s.db_path)
    # Reload any persisted runtime threshold overrides so they survive restarts.
    for router_name, thr in store.get_thresholds().items():
        set_runtime_threshold(router_name, thr)
    audit = AuditLogger(root=Path(s.audit_dir), store=store)
    app.state.controller = CascadeController(client=client, routers=routers, cache=cache, audit=audit)
    app.state.routers = routers
    app.state.aggregator = EvalAggregator(store=store)
    app.state.audit = audit
    app.state.store = store
    app.state.medmcqa_pool = None  # lazily filled on first /v1/eval/sample
    yield
    audit.close()


app = FastAPI(title="med-routing", version="0.1.0", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "routers": sorted(app.state.routers.keys())}


@app.get("/v1/eval/sample")
async def sample_question(dataset: str = "medmcqa") -> dict[str, Any]:
    """Return a random eval item for the demo UI. medmcqa/medqa = MCQ;
    medquad = free-form (no options, reference_answer instead)."""
    if dataset not in ("medmcqa", "medqa", "medquad"):
        raise HTTPException(400, "dataset must be 'medmcqa', 'medqa', or 'medquad'")
    pools = app.state.medmcqa_pool or {}
    if dataset not in pools:
        if dataset == "medmcqa":
            from .eval.medmcqa import load_medmcqa
            pools["medmcqa"] = list(load_medmcqa(split="validation", limit=200))
        elif dataset == "medqa":
            from .eval.medqa import load_medqa
            pools["medqa"] = list(load_medqa(split="test", limit=200))
        else:
            from .eval.medquad import load_medquad
            pools["medquad"] = list(load_medquad(split="train", limit=200))
        app.state.medmcqa_pool = pools
    pool = pools[dataset]
    if not pool:
        raise HTTPException(503, f"{dataset} pool empty")
    item = random.choice(pool)
    out: dict[str, Any] = {
        "qid": item.qid,
        "question": item.question,
        "subject": item.subject,
        "dataset": dataset,
    }
    if dataset == "medquad":
        out["reference_answer"] = item.reference_answer
        out["format"] = "free_form"
    else:
        out["options"] = list(item.options)
        out["answer"] = item.answer
        out["format"] = "mcq"
    return out


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/eval/accuracy")
async def push_accuracy(payload: AccuracyUpdate) -> dict[str, Any]:
    """Eval runner pushes rolling accuracy here so Grafana shows it live."""
    ACCURACY.labels(router=payload.router).set(payload.accuracy)
    return {"ok": True}


@app.get("/v1/audit/recent")
async def audit_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Recent decision rows from the audit log (in-memory ring buffer)."""
    return app.state.audit.recent(limit=max(1, min(limit, 200)))


@app.get("/v1/audit/stats")
async def audit_stats() -> dict[str, Any]:
    return app.state.store.stats()


@app.get("/v1/audit/decisions")
async def query_decisions(
    limit: int = 100,
    since: str | None = None,
    router: str | None = None,
    cross_border_only: bool = False,
    format: str = "json",
):
    rows = app.state.store.query_decisions(
        limit=max(1, min(limit, 10_000)),
        since=since, router=router, cross_border_only=cross_border_only,
    )
    if format == "csv":
        return Response(
            content="".join(iter_csv(rows, DECISION_COLS)),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=decisions.csv"},
        )
    return rows


@app.get("/v1/feedback/recommendation")
async def feedback_recommendation(
    target_kept_accuracy: float = 0.85,
    min_samples: int = 30,
    limit_per_router: int = 5000,
) -> dict[str, Any]:
    """Sweep historical eval rows by router and recommend a new threshold for
    each router that has enough data. Returns recommendations only — no live
    state is changed. Use POST /v1/feedback/apply to commit a recommendation.
    """
    store = app.state.store
    rows_by_router: dict[str, list[dict[str, Any]]] = {}
    for r in app.state.routers:
        rows_by_router[r] = store.query_eval(router=r, limit=limit_per_router)

    s = get_settings()
    return {
        "target_kept_accuracy": target_kept_accuracy,
        "min_samples": min_samples,
        "current_thresholds": {r: s.threshold_for(r) for r in app.state.routers},
        "runtime_overrides": runtime_overrides(),
        "recommendations": recommend_for_all(
            rows_by_router,
            target_kept_accuracy=target_kept_accuracy,
            min_samples=min_samples,
        ),
        "n_labeled_rows": {r: len(rows) for r, rows in rows_by_router.items()},
    }


@app.post("/v1/feedback/apply")
async def feedback_apply(payload: ThresholdApply) -> dict[str, Any]:
    if payload.router not in app.state.routers:
        raise HTTPException(404, f"Unknown router {payload.router!r}")
    if not 0.0 <= payload.threshold <= 1.0:
        raise HTTPException(400, "threshold must be in [0, 1]")
    app.state.store.set_threshold(
        router=payload.router, threshold=payload.threshold, reason=payload.reason,
    )
    set_runtime_threshold(payload.router, payload.threshold)
    return {
        "ok": True,
        "router": payload.router,
        "threshold": payload.threshold,
        "effective_now": True,
    }


@app.delete("/v1/feedback/apply/{router}")
async def feedback_clear(router: str) -> dict[str, Any]:
    if router not in app.state.routers:
        raise HTTPException(404, f"Unknown router {router!r}")
    clear_runtime_threshold(router)
    # Note: history is kept; the override row stays in runtime_thresholds until
    # next apply. For a hackathon this is fine — purge logic can come later.
    return {"ok": True, "cleared": router}


@app.get("/v1/feedback/thresholds")
async def feedback_thresholds() -> dict[str, Any]:
    s = get_settings()
    return {
        "effective": {r: s.threshold_for(r) for r in app.state.routers},
        "runtime_overrides": runtime_overrides(),
        "history": app.state.store.threshold_history(limit=20),
    }


@app.get("/v1/audit/decisions.fhir")
async def query_decisions_fhir(
    limit: int = 100,
    since: str | None = None,
    router: str | None = None,
    cross_border_only: bool = False,
    format: str = "ndjson",
):
    """Cascade decisions as HL7 FHIR R4 AuditEvent resources.

    Two output shapes:
      - format=ndjson (default): one AuditEvent per line, FHIR Bulk Data style.
      - format=bundle: a single FHIR Bundle (type=collection) wrapping all events.
    """
    rows = app.state.store.query_decisions(
        limit=max(1, min(limit, 10_000)),
        since=since, router=router, cross_border_only=cross_border_only,
    )
    events = [decision_to_audit_event(r) for r in rows]

    if format == "bundle":
        import json as _json
        return Response(
            content=_json.dumps(to_bundle(events), default=str),
            media_type="application/fhir+json",
        )

    import json as _json
    body = "\n".join(_json.dumps(e, default=str) for e in events) + ("\n" if events else "")
    return Response(
        content=body,
        media_type="application/fhir+ndjson",
        headers={"Content-Disposition": "attachment; filename=audit-events.ndjson"},
    )


@app.get("/v1/audit/eval")
async def query_eval(
    limit: int = 1000,
    since: str | None = None,
    router: str | None = None,
    format: str = "json",
):
    rows = app.state.store.query_eval(
        limit=max(1, min(limit, 100_000)),
        since=since, router=router,
    )
    if format == "csv":
        return Response(
            content="".join(iter_csv(rows, EVAL_COLS)),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=eval_rows.csv"},
        )
    return rows


@app.post("/v1/eval/observe")
async def push_observation(payload: Observation, request: Request) -> dict[str, Any]:
    """Per-question evaluation observation. Updates calibration counters,
    selective accuracy gauges, per-subject accuracy, ECE."""
    summary = app.state.aggregator.observe(
        router=payload.router,
        score=payload.score,
        escalated=payload.escalated,
        correct=payload.correct,
        subject=payload.subject,
    )
    return {"ok": True, **summary}


class CompareRequest(BaseModel):
    messages: list[ChatMessage]
    routers: list[str] | None = Field(default=None, description="Subset of routers; default = all registered.")


@app.post("/v1/compare")
async def compare_routers(req: CompareRequest) -> JSONResponse:
    """Fan-out one prompt to multiple routers in parallel and return each
    router's decision side-by-side. The cascade's weak-call cache ensures we
    pay for the weak inference once even though every router uses it."""
    import asyncio
    import time as _time

    available = sorted(app.state.routers.keys())
    chosen = req.routers or available
    unknown = [r for r in chosen if r not in app.state.routers]
    if unknown:
        raise HTTPException(404, f"Unknown routers: {unknown}; available: {available}")

    controller: CascadeController = app.state.controller
    messages = [m.model_dump() for m in req.messages]

    async def one(router_name: str) -> dict[str, Any]:
        t0 = _time.perf_counter()
        try:
            res = await controller.handle(messages, router_name)
        except Exception as exc:  # surface per-router failures without aborting the whole compare
            return {"router": router_name, "error": str(exc)}
        return {
            "router": router_name,
            "score": res.score,
            "threshold": res.threshold,
            "escalated": res.escalated,
            "model_used": res.model_used,
            "text": res.text,
            "latency_ms": int((_time.perf_counter() - t0) * 1000),
            "extras": res.extras,
        }

    results = await asyncio.gather(*(one(r) for r in chosen))
    return JSONResponse(content={"prompt_sha": None, "results": results})


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> JSONResponse:
    router_name = req.router or request.headers.get("x-router")
    if not router_name:
        raise HTTPException(400, f"Specify a router via 'router' field or X-Router header. Known: {KNOWN_ROUTERS}")
    if router_name not in app.state.routers:
        raise HTTPException(
            404,
            f"Router {router_name!r} not enabled. Available: {sorted(app.state.routers)}",
        )

    controller: CascadeController = app.state.controller
    messages = [m.model_dump() for m in req.messages]
    try:
        result = await controller.handle(messages, router_name)
    except Exception as exc:  # surface OpenAI / model errors as 502
        raise HTTPException(502, f"upstream error: {exc}") from exc

    body = {
        "id": f"medr-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model_used,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "med_routing": {
            "router": result.router,
            "score": result.score,
            "threshold": result.threshold,
            "escalated": result.escalated,
            "weak_model": result.weak_completion.model,
            "strong_model": result.strong_completion.model if result.strong_completion else None,
            "extras": result.extras,
        },
    }
    headers = {
        "X-Med-Router": result.router,
        "X-Uncertainty": f"{result.score:.4f}",
        "X-Escalated": str(result.escalated).lower(),
        "X-Model-Used": result.model_used,
    }
    return JSONResponse(content=body, headers=headers)


def main() -> None:
    import uvicorn

    uvicorn.run("med_routing.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
