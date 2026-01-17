"""Deterministic agent selection and scheduling for FINO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.config.agents import AGENT_MAP
from app.core.agents.base import Agent
from app.core.agents.registry import AgentRegistry
from app.core.intent.intent_types import IntentResult
from app.core.state import ConversationState


@dataclass(frozen=True)
class SelectedAgent:
    """Selected agent with identity preserved for scheduling."""

    name: str
    agent: Agent


class AgentSelector:
    """Select sub-agents deterministically based on conversation intent."""

    def __init__(
        self,
        registry: AgentRegistry,
        mapping: Dict[str, Sequence[str]] | None = None,
        fallback_intent: str = "etc",
    ) -> None:
        self._registry = registry
        self._fallback_intent = fallback_intent
        self._mapping = mapping or {key: list(value) for key, value in AGENT_MAP.items()}

    def select(self, state: ConversationState) -> List[SelectedAgent]:
        """Return agents in execution order for a conversation state."""

        intent_key = self._normalize_intent(state.intent_result)
        names = list(self._mapping.get(intent_key, self._mapping[self._fallback_intent]))
        resolved_names = [self._resolve_agent_name(name) for name in names]
        return [SelectedAgent(name=name, agent=self._registry.get(name)) for name in resolved_names]

    def schedule(self, state: ConversationState) -> List[SelectedAgent]:
        """Alias for select to emphasize sequential scheduling."""

        return self.select(state)

    def _resolve_agent_name(self, name: str) -> str:
        if name in self._registry.list():
            return name
        lowered = name.replace("Agent", "").lower()
        if lowered in self._registry.list():
            return lowered
        return name

    @staticmethod
    def _normalize_intent(intent_result: IntentResult | None) -> str:
        if intent_result is None:
            return "etc"
        key = intent_result.primary_intent.strip().lower()
        return key or "etc"
