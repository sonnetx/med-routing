from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from .medmcqa import load_medmcqa
from .scoring import is_correct


async def _run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.router}_{int(time.time())}.jsonl"

    items = list(load_medmcqa(split=args.split, limit=args.n))
    print(f"Loaded {len(items)} MedMCQA items from {args.split}")

    correct = 0
    total = 0
    escalations = 0
    cost_total = 0.0

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
                ok = is_correct(text, item.answer)
                correct += int(ok)
                total += 1
                escalations += int(bool(meta.get("escalated")))

                row = {
                    "qid": item.qid,
                    "subject": item.subject,
                    "gold": item.answer,
                    "predicted_text": text,
                    "correct": ok,
                    "router": args.router,
                    "score": meta.get("score"),
                    "threshold": meta.get("threshold"),
                    "escalated": meta.get("escalated"),
                    "model_used": body.get("model"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                if total % args.push_every == 0:
                    acc = correct / total
                    try:
                        await client.post(
                            f"{args.base_url}/v1/eval/accuracy",
                            json={"router": args.router, "accuracy": acc},
                        )
                    except Exception:
                        pass
                    print(
                        f"[{total:>4}/{len(items)}] acc={acc:.3f} "
                        f"esc={escalations / total:.3f} router={args.router}"
                    )

    if total:
        acc = correct / total
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                await client.post(
                    f"{args.base_url}/v1/eval/accuracy",
                    json={"router": args.router, "accuracy": acc},
                )
            except Exception:
                pass
        print(f"\nDone. router={args.router} n={total} acc={acc:.4f} escalation_rate={escalations / total:.4f}")
        print(f"Per-question log: {out_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run MedMCQA through the med-routing cascade.")
    p.add_argument("--router", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--split", default="validation")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--push-every", type=int, default=25)
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
