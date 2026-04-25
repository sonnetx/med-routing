"""Collect training data for the learned router.

For each MedMCQA item, run the weak model once, sample N times, and compute all
four uncertainty signals against the *same* weak completion / sample set. Write
one JSONL row per item.

Cost per item (defaults): 1 weak call (logprobs) + 1 N=5 sampled batch +
1 self-report follow-up ≈ 3 weak calls. With --with-strong, add 1 strong call.

Usage:
    python -m med_routing.train.collect --n 5000 --split train --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..eval.medmcqa import load_medmcqa
from ..eval.scoring import is_correct, parse_letter
from ..llm.openai_client import Completion, OpenAIClient
from ..routers.predictive_entropy import PredictiveEntropyRouter
from ..routers.self_consistency import SelfConsistencyRouter
from ..routers.self_reported import SelfReportedRouter


async def _weak_with_logprobs(
    client: OpenAIClient, messages: list[dict[str, str]], *, model: str
) -> Completion:
    completions = await client.complete(
        model=model,
        messages=messages,
        tier="weak",
        temperature=0.0,
        max_tokens=64,
        logprobs=True,
        top_logprobs=5,
    )
    return completions[0]


async def _sample_n(
    client: OpenAIClient,
    messages: list[dict[str, str]],
    *,
    model: str,
    n: int,
    temperature: float,
) -> list[Completion]:
    return await client.complete(
        model=model,
        messages=messages,
        tier="weak",
        temperature=temperature,
        max_tokens=64,
        n=n,
    )


async def _process_item(
    item: Any,
    *,
    client: OpenAIClient,
    pe_router: PredictiveEntropyRouter,
    sc_router: SelfConsistencyRouter,
    sr_router: SelfReportedRouter,
    se_router: Any | None,  # SemanticEntropyRouter or None
    with_strong: bool,
) -> dict[str, Any] | None:
    s = get_settings()
    messages = item.to_messages()

    # 1. Weak call with logprobs (shared across pe + self_reported's weak base).
    try:
        weak = await _weak_with_logprobs(client, messages, model=s.weak_model)
    except Exception as exc:
        print(f"[err qid={item.qid}] weak call failed: {exc}", file=sys.stderr)
        return None

    # 2. N samples (shared across self_consistency and semantic_entropy).
    samples = await _sample_n(
        client,
        messages,
        model=s.weak_model,
        n=s.sample_n,
        temperature=s.sample_temperature,
    )

    async def cached_sampler(*, n: int, temperature: float) -> list[Completion]:
        # Routers below ask the sampler for (sample_n, sample_temperature). Since
        # we already drew that batch, hand it back without paying again.
        if n == s.sample_n and abs(temperature - s.sample_temperature) < 1e-9:
            return samples
        return await _sample_n(
            client, messages, model=s.weak_model, n=n, temperature=temperature
        )

    # 3. All four signals against the same weak/samples.
    pe = await pe_router.score(messages=messages, weak=weak, sampler=cached_sampler)
    sc = await sc_router.score(messages=messages, weak=weak, sampler=cached_sampler)
    sr = await sr_router.score(messages=messages, weak=weak, sampler=cached_sampler)
    se_score: float | None = None
    se_extras: dict[str, Any] | None = None
    if se_router is not None:
        se = await se_router.score(messages=messages, weak=weak, sampler=cached_sampler)
        se_score = se.score
        se_extras = se.extras

    weak_letter = parse_letter(weak.text)
    weak_correct = is_correct(weak.text, item.answer)

    row: dict[str, Any] = {
        "qid": item.qid,
        "subject": item.subject,
        "prompt_len": len(messages[-1]["content"]),
        "gold": item.answer,
        "weak_text": weak.text,
        "weak_letter": weak_letter,
        "weak_correct": weak_correct,
        # Features
        "self_reported": sr.score,
        "predictive_entropy": pe.score,
        "self_consistency": sc.score,
        "semantic_entropy": se_score,
        # Diagnostics — useful when fitting; cheap to keep
        "self_reported_extras": sr.extras,
        "self_consistency_extras": {
            k: v for k, v in sc.extras.items() if k != "samples"
        },
        "predictive_entropy_extras": pe.extras,
        "semantic_entropy_extras": se_extras,
    }

    if with_strong:
        try:
            strong_results = await client.complete(
                model=s.strong_model,
                messages=messages,
                tier="strong",
                temperature=0.0,
                max_tokens=64,
            )
            strong = strong_results[0]
            strong_correct = is_correct(strong.text, item.answer)
            row["strong_text"] = strong.text
            row["strong_correct"] = strong_correct
            row["escalation_helps"] = strong_correct and not weak_correct
        except Exception as exc:
            print(f"[warn qid={item.qid}] strong call failed: {exc}", file=sys.stderr)
            row["strong_text"] = None
            row["strong_correct"] = None
            row["escalation_helps"] = None

    return row


async def _run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_{int(time.time())}.jsonl"

    items = list(load_medmcqa(split=args.split, limit=args.n))
    print(f"Loaded {len(items)} MedMCQA items from split={args.split}")

    client = OpenAIClient()
    pe_router = PredictiveEntropyRouter()
    sc_router = SelfConsistencyRouter()
    sr_router = SelfReportedRouter(client)

    se_router = None
    if args.with_semantic:
        # Force-load NLI regardless of ENABLE_NLI env, since collect requires it.
        from ..nli import get_nli_scorer
        from ..routers.semantic_entropy import SemanticEntropyRouter

        s = get_settings()
        try:
            scorer = get_nli_scorer(s.nli_model)
            se_router = SemanticEntropyRouter(scorer)
        except Exception as exc:
            print(
                f"[fatal] could not load NLI model {s.nli_model!r}: {exc}\n"
                f"        Pass --no-semantic to skip semantic_entropy.",
                file=sys.stderr,
            )
            return 2

    sem = asyncio.Semaphore(args.concurrency)
    written = 0
    weak_correct_count = 0
    lock = asyncio.Lock()

    async def worker(item: Any) -> None:
        nonlocal written, weak_correct_count
        async with sem:
            row = await _process_item(
                item,
                client=client,
                pe_router=pe_router,
                sc_router=sc_router,
                sr_router=sr_router,
                se_router=se_router,
                with_strong=args.with_strong,
            )
        if row is None:
            return
        async with lock:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            weak_correct_count += int(row["weak_correct"])
            if written % args.log_every == 0:
                acc = weak_correct_count / written
                print(
                    f"[{written:>5}/{len(items)}] weak_acc={acc:.3f} "
                    f"out={out_path.name}"
                )

    with out_path.open("w", encoding="utf-8") as f:
        await asyncio.gather(*(worker(it) for it in items))

    print(
        f"\nDone. wrote {written}/{len(items)} rows to {out_path} "
        f"(weak_acc={weak_correct_count / max(written, 1):.4f})"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect MedMCQA training rows for the learned router."
    )
    p.add_argument("--n", type=int, default=5000, help="Max items to process.")
    p.add_argument("--split", default="train", help="MedMCQA split.")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--with-strong",
        action="store_true",
        help="Also call the strong model and label escalation_helps. ~doubles cost.",
    )
    p.add_argument(
        "--no-semantic",
        dest="with_semantic",
        action="store_false",
        help="Skip semantic_entropy (avoids NLI model download).",
    )
    p.set_defaults(with_semantic=True)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
