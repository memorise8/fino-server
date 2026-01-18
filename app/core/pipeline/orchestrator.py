"""Backbone orchestration layer for FINO intent → routing → agents → integration."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Iterable, List, Sequence

from app.config.settings import settings
from app.core.agents.registry import AgentRegistry
from app.core.dsl.domain_scope_adapter import DomainScopeAdapter
from app.core.dsl.types import LawScope
from app.core.integrator import Integrator
from app.core.intent.intent_types import IntentResult as StructuredIntentResult
from app.core.normalization.query_normalizer import QueryNormalizer
from app.core.routing.agent_selector import AgentSelector, SelectedAgent
from app.core.routing.router import Router
from app.core.state import ConversationState
from app.core.types import AgentResponse, FinalResponse, IntentResult, RouteResult

logger = logging.getLogger(__name__)


class BackboneOrchestrator:
    """Orchestrate intent analysis, routing, agent execution, and integration."""

    def __init__(
        self,
        intent_analyzer: Callable[[str], Awaitable[IntentResult]],
        router: Router,
        agents: AgentRegistry,
        integrator: Integrator,
        agent_selector: AgentSelector | None = None,
    ) -> None:
        self._intent_analyzer = intent_analyzer
        self._router = router
        self._agents = agents
        self._integrator = integrator
        self._agent_selector = agent_selector
        self._domain_scope_adapter = DomainScopeAdapter()
        self._env = settings.env

    async def run(self, message: str, agent_names: Sequence[str] | None = None) -> FinalResponse:
        logger.info("Orchestration start")
        intent = await self._intent_analyzer(message)
        normalized_query = QueryNormalizer.normalize(message)
        law_scope = self._domain_scope_adapter.resolve(normalized_query)
        intent = self._attach_law_scope(intent, law_scope)
        route = await self._router.route(intent)
        selections = self._select_agents(message, intent, agent_names, law_scope)
        logger.info("Selected agents=%s", [selection.name for selection in selections])
        responses = await self._call_agents(message, intent, route, selections)
        final_response = self._integrator.integrate(responses)
        logger.info("Orchestration end")
        return final_response

    def _select_agents(
        self,
        message: str,
        intent: IntentResult,
        agent_names: Sequence[str] | None,
        law_scope: LawScope | None,
    ) -> Sequence[SelectedAgent]:
        if agent_names is not None:
            return [SelectedAgent(name=name, agent=self._agents.get(name)) for name in agent_names]
        if self._agent_selector is None:
            raise ValueError("AgentSelector is required when agent_names is not provided.")
        state = self._build_state(message, intent, law_scope)
        return self._agent_selector.select(state)

    @staticmethod
    def _build_state(
        message: str,
        intent: IntentResult,
        law_scope: LawScope | None,
    ) -> ConversationState:
        structured_intent = BackboneOrchestrator._to_structured_intent(intent)
        resolved_scope = law_scope
        if resolved_scope is None and isinstance(intent.data, dict):
            resolved_scope = intent.data.get("law_scope")
        out_of_scope = resolved_scope is None
        return ConversationState(
            original_query=message,
            intent_result=structured_intent,
            law_scope=resolved_scope,
            out_of_scope=out_of_scope,
        )

    @staticmethod
    def _to_structured_intent(intent: IntentResult) -> StructuredIntentResult:
        data = intent.data if isinstance(intent.data, dict) else {}
        primary_intent = str(data.get("primary_intent", "etc"))
        raw_secondary = data.get("secondary_intents", [])
        if isinstance(raw_secondary, list):
            secondary_intents = [str(item) for item in raw_secondary]
        else:
            secondary_intents = []
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return StructuredIntentResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
        )

    async def _call_agents(
        self,
        message: str,
        intent: IntentResult,
        route: RouteResult,
        selections: Sequence[SelectedAgent],
    ) -> Iterable[AgentResponse]:
        responses: List[AgentResponse] = []
        for selection in selections:
            try:
                response = await selection.agent.run(message, intent, route)
                payload = dict(response.data)
                payload.setdefault("agent_name", selection.name)
                responses.append(AgentResponse(data=payload))
            except Exception:
                logger.exception("Agent execution failure agent=%s", selection.name)
                raise
        return responses

    @staticmethod
    def _attach_law_scope(intent: IntentResult, law_scope: LawScope | None) -> IntentResult:
        data = dict(intent.data) if isinstance(intent.data, dict) else {}
        data["law_scope"] = law_scope
        data["out_of_scope"] = law_scope is None
        return IntentResult(data=data)
