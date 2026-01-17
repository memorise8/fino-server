"""Intent typing primitives for FINO RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class IntentResult:
    """Structured intent summary used to steer FINO RAG routing."""

    primary_intent: str
    secondary_intents: List[str] = field(default_factory=list)
    confidence: float = 0.0
