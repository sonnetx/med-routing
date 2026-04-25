from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from .medmcqa import load_medmcqa
from .medqa import load_medqa
from .medquad import load_medquad
from .scoring import is_correct, score_freeform_with_judge


_DEFAULT_SPLIT = {"medmcqa": "validation", "medqa": "test", "medquad": "train"}
_FREEFORM_DATASETS = {"medquad"}


def _load_items(dataset: str, split: str, limit: int):
    if dataset == "medmcqa":
        return list(load_medmcqa(split=split, limit=limit))
    if dataset == "medqa":
        return list(load_medqa(split=split, limit=limit))
    if dataset == "medquad":
        return list(load_medquad(split=split, limit=limit))
    raise ValueError(f"unknown dataset {dataset!r}; expected medmcqa, medqa, or medquad")


async def _run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.router}_{int(time.time())}.jsonl"

    split = args.split or _DEFAULT_SPLIT[args.dataset]
    items = _load_items(args.dataset, split, args.n)
    print(f"Loaded {len(items)} {args.dataset} items from split={split}")

    correct = 0
    total = 0
    escalations = 0
    cost_total = 0.0

    is_freeform = args.dataset in _FREEFORM_DATASETS
    judge = None
    if is_freeform:
        from ..llm.openai_client import OpenAIClient
        judge = OpenAIClient()

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                payload = {"messages": item.to_messages(), "router": args.router}
                try:
                    resp = await client.post(f"{args.base_url}/v1/chat/completions", json=payload)
                    resp.raise_for_status()
                    body = resp.json()
                except Exception as exc:
                    print(f"[err qid={item.qid}] {exc}", file=sys.stderr)
                    continue

                text = body["choices"][0]["message"]["content"]
                meta = body.get("med_routing", {})

                if is_freeform:
                    ok = await score_freeform_with_judge(
                        question=item.question,
                        predicted=text,
                        reference=item.reference_answer,
                        judge_client=judge,
                        judge_model=args.judge_model,
                    )
                    gold_for_log = item.reference_answer[:200]
                else:
                    ok = is_correct(text, item.answer)
                    gold_for_log = item.answer

                correct += int(ok)
                total += 1
                escalations += int(bool(meta.get("escalated")))

                row = {
                    "qid": item.qid,
                    "subject": item.subject,
                    "gold": gold_for_log,
                    "predicted_text": text,
                    "correct": ok,
                    "router": args.router,
                    "score": meta.get("score"),
                    "threshold": meta.get("threshold"),
                    "escalated": meta.get("escalated"),
                    "model_used": body.get("model"),
                    "dataset": args.dataset,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                try:
                    await client.post(
                        f"{args.base_url}/v1/eval/observe",
                        json={
                            "router": args.router,
                            "score": float(meta.get("score") or 0.0),
                            "escalated": bool(meta.get("escalated")),
                            "correct": ok,
                            "subject": item.subject or None,
                        },
                    )
                except Exception:
                    pass

                if total % args.push_every == 0:
                    acc = correct / total
                    print(
                        f"[{total:>4}/{len(items)}] acc={acc:.3f} "
                        f"esc={escalations / total:.3f} router={args.router}"
                    )

    if total:
        acc = correct / total
        print(f"\nDone. router={args.router} n={total} acc={acc:.4f} escalation_rate={escalations / total:.4f}")
        print(f"Per-question log: {out_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run MedMCQA through the med-routing cascade.")
    p.add_argument("--router", required=True)
    p.add_argument("--dataset", choices=["medmcqa", "medqa", "medquad"], default="medmcqa",
                   help="medmcqa easier (~75-80%); medqa USMLE-hard (~60-65%); medquad free-form, scored by LLM-as-judge.")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--split", default=None,
                   help="Default: validation for medmcqa, test for medqa, train for medquad.")
    p.add_argument("--judge-model", default="gpt-4o",
                   help="Used only for free-form datasets to grade predictions against reference answers.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--push-every", type=int, default=25)
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
