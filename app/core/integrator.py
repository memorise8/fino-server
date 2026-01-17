from __future__ import annotations

from typing import Iterable, List

from app.core.types import AgentResponse, FinalResponse


class Integrator:
    """Integrate agent responses without additional reasoning."""

    def integrate(self, responses: Iterable[AgentResponse]) -> FinalResponse:
        return FinalResponse(responses=list(responses))
