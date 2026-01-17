from __future__ import annotations

from typing import Protocol

from app.core.types import IntentResult, RouteResult


class Router(Protocol):
    """Pure routing decision interface.

    TODO: Provide a concrete router implementation when routing logic is finalized.
    """

    async def route(self, intent: IntentResult) -> RouteResult:
        ...
