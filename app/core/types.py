from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class IntentResult:
    """Result of intent analysis.

    TODO: Replace `data` with concrete fields once the intent schema is finalized.
    """

    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteResult:
    """Routing decision for downstream agents.

    TODO: Replace `data` with concrete fields once the routing schema is finalized.
    """

    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRequest:
    """Request payload for an agent call.

    TODO: Replace `data` with concrete fields once the agent request schema is finalized.
    """

    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    """Response payload from an agent call.

    TODO: Replace `data` with concrete fields once the agent response schema is finalized.
    """

    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalResponse:
    """Final integrated response returned to the caller.

    TODO: Replace `responses` with a concrete schema once the response contract is finalized.
    """

    responses: List[AgentResponse] = field(default_factory=list)
