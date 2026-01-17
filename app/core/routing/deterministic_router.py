"""Deterministic router for FINO orchestration."""

from __future__ import annotations

from typing import Dict

from app.core.routing.router import Router
from app.core.types import IntentResult, RouteResult


class DeterministicRouter(Router):
    """Route intents to a stable path label without external dependencies."""

    def __init__(self, mapping: Dict[str, str] | None = None, fallback: str = "default") -> None:
        self._mapping = mapping or {
            "tax": "tax_flow",
            "accounting": "accounting_flow",
            "both": "multi_flow",
            "etc": "general_flow",
        }
        self._fallback = fallback

    async def route(self, intent: IntentResult) -> RouteResult:
        """Return a deterministic route payload for downstream agents."""

        primary_intent = str(intent.data.get("primary_intent", "etc")).lower()
        route = self._mapping.get(primary_intent, self._fallback)
        return RouteResult(
            data={
                "route": route,
                "primary_intent": primary_intent,
            }
        )
