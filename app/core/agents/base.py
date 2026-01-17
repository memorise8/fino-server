from __future__ import annotations

from typing import Protocol

from app.core.types import AgentResponse, IntentResult, RouteResult


class Agent(Protocol):
    """Agent execution interface."""

    async def run(self, message: str, intent: IntentResult, route: RouteResult) -> AgentResponse:
        ...
