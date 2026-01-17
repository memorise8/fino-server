from __future__ import annotations

from typing import Dict, Iterable

from app.core.agents.base import Agent
from app.core.types import AgentResponse, IntentResult, RouteResult


class AgentServiceAdapter:
    """Adapter that forwards calls to an existing AgentService instance."""

    def __init__(self, agent_service: object) -> None:
        self._agent_service = agent_service

    async def run(self, message: str, intent: IntentResult, route: RouteResult) -> AgentResponse:
        if not hasattr(self._agent_service, "run"):
            raise NotImplementedError("TODO: Wire AgentService.run into AgentServiceAdapter")
        return await self._agent_service.run(message=message, intent=intent, route=route)


class AgentRegistry:
    """Registry mapping agent names to Agent adapters."""

    def __init__(self, agents: Dict[str, Agent] | None = None) -> None:
        self._agents: Dict[str, Agent] = dict(agents or {})

    def register(self, name: str, agent: Agent) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> Agent:
        return self._agents[name]

    def list(self) -> Iterable[str]:
        return self._agents.keys()
