from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from ..config import cost_usd, get_settings
from ..metrics import MODEL_CALLS_TOTAL, PROCESSOR_CALLS_TOTAL, record_usage
from ..processors import get_processor


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Completion:
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    logprobs: list[TokenLogprob] | None = None
    raw: dict[str, Any] | None = None

    @property
    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


def _parse_logprobs(choice: Any) -> list[TokenLogprob] | None:
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    content = getattr(lp, "content", None)
    if not content:
        return None
    out: list[TokenLogprob] = []
    for tok in content:
        top = []
        for alt in getattr(tok, "top_logprobs", None) or []:
            top.append((alt.token, alt.logprob))
        out.append(TokenLogprob(token=tok.token, logprob=tok.logprob, top=top))
    return out


class OpenAIClient:
    """Async wrapper that returns a structured Completion with usage + logprobs."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(
            api_key=api_key or s.openai_api_key or None,
            base_url=base_url or s.openai_base_url,
        )

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        tier: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        n: int = 1,
        seed: int | None = None,
    ) -> list[Completion]:
        # GPT-5 / o1 / o3 families renamed max_tokens -> max_completion_tokens.
        # Translate at this layer so callers stay model-agnostic.
        token_param = (
            "max_completion_tokens"
            if model.startswith(("gpt-5", "o1", "o3", "o4"))
            else "max_tokens"
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            token_param: max_tokens,
            "n": n,
        }
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs
        if seed is not None:
            kwargs["seed"] = seed

        resp = await self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        # Spread usage across choices proportionally; OpenAI reports a single aggregate.
        total = max(len(resp.choices), 1)
        per_choice_completion = completion_tokens // total
        cost_per_call = cost_usd(model, prompt_tokens, completion_tokens)

        MODEL_CALLS_TOTAL.labels(model=model, tier=tier).inc(total)
        proc = get_processor(model)
        PROCESSOR_CALLS_TOTAL.labels(processor=proc.name, entity=proc.entity, region=proc.region).inc(total)
        record_usage(model, prompt_tokens, completion_tokens, cost_per_call)

        out: list[Completion] = []
        for choice in resp.choices:
            text = choice.message.content or ""
            out.append(
                Completion(
                    model=model,
                    text=text,
                    prompt_tokens=prompt_tokens // total,
                    completion_tokens=per_choice_completion,
                    cost=cost_per_call / total,
                    logprobs=_parse_logprobs(choice),
                )
            )
        return out
