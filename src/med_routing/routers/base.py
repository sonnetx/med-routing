from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from ..llm.openai_client import Completion


class Sampler(Protocol):
    async def __call__(self, *, n: int, temperature: float) -> list[Completion]: ...


@dataclass
class RouterScore:
    """Output of an UncertaintyRouter.

    score: float in [0,1], higher = more uncertain (should escalate).
    extras: arbitrary diagnostics surfaced in headers/logs (kept small, JSON-safe).
    threshold_override: optional per-call threshold the router prefers the
        cascade use instead of the default for its registered name. Meta-routers
        like `auto` set this to the threshold of the sub-router they delegated
        to, so calibration stays correct across format-aware dispatch.
    """
    score: float
    extras: dict[str, Any]
    threshold_override: float | None = None


class UncertaintyRouter(ABC):
    name: ClassVar[str]

    @abstractmethod
    async def score(
        self,
        *,
        messages: list[dict[str, str]],
        weak: Completion,
        sampler: Sampler,
    ) -> RouterScore: ...
