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

from .cache import CompletionCache
from .cascade import CascadeController
from .config import get_settings
from .eval.aggregator import EvalAggregator
from .llm.openai_client import OpenAIClient
from .metrics import ACCURACY, REGISTRY
from .routers.registry import KNOWN_ROUTERS, build_routers


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = OpenAIClient()
    routers = build_routers(client)
    cache = CompletionCache()
    app.state.controller = CascadeController(client=client, routers=routers, cache=cache)
    app.state.routers = routers
    app.state.aggregator = EvalAggregator()
    app.state.medmcqa_pool = None  # lazily filled on first /v1/eval/sample
    yield


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
async def sample_question() -> dict[str, Any]:
    """Return a random MedMCQA validation item for the demo UI."""
    pool = app.state.medmcqa_pool
    if pool is None:
        from .eval.medmcqa import load_medmcqa

        pool = list(load_medmcqa(split="validation", limit=200))
        app.state.medmcqa_pool = pool
    if not pool:
        raise HTTPException(503, "MedMCQA pool empty")
    item = random.choice(pool)
    return {
        "qid": item.qid,
        "question": item.question,
        "options": list(item.options),
        "answer": item.answer,
        "subject": item.subject,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/eval/accuracy")
async def push_accuracy(payload: AccuracyUpdate) -> dict[str, Any]:
    """Eval runner pushes rolling accuracy here so Grafana shows it live."""
    ACCURACY.labels(router=payload.router).set(payload.accuracy)
    return {"ok": True}


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
