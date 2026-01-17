from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Iterable, Sequence

from app.core.agents.registry import AgentRegistry
from app.core.integrator import Integrator
from app.core.routing.router import Router
from app.core.types import AgentResponse, FinalResponse, IntentResult, RouteResult


class BackboneOrchestrator:
    """Backbone orchestration layer for intent → routing → agents → integration."""

    def __init__(
        self,
        intent_analyzer: Callable[[str], Awaitable[IntentResult]],
        router: Router,
        agents: AgentRegistry,
        integrator: Integrator,
    ) -> None:
        self._intent_analyzer = intent_analyzer
        self._router = router
        self._agents = agents
        self._integrator = integrator

    async def run(self, message: str, agent_names: Sequence[str]) -> FinalResponse:
        intent = await self._intent_analyzer(message)
        route = await self._router.route(intent)
        responses = await self._call_agents(message, intent, route, agent_names)
        return self._integrator.integrate(responses)

    async def _call_agents(
        self,
        message: str,
        intent: IntentResult,
        route: RouteResult,
        agent_names: Sequence[str],
    ) -> Iterable[AgentResponse]:
        if len(agent_names) == 1:
            agent = self._agents.get(agent_names[0])
            return [await agent.run(message, intent, route)]
        tasks = [self._agents.get(name).run(message, intent, route) for name in agent_names]
        return await asyncio.gather(*tasks)
