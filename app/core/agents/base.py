"""Agent contracts for the FINO backbone and RAG extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from app.core.evidence import Evidence
from app.core.types import AgentResponse, IntentResult, RouteResult


@dataclass(frozen=True)
class AgentOutput:
    """Structured output returned by FINO agents."""

    answer: str
    evidences: List[Evidence]
    confidence: float


class Agent(Protocol):
    """Agent execution interface for the backbone and RAG output contract."""

    async def run(self, message: str, intent: IntentResult, route: RouteResult) -> AgentResponse:
        ...

    async def execute(self, message: str, intent: IntentResult, route: RouteResult) -> AgentOutput:
        ...
