"""Conversation state container for FINO RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.core.dsl.types import LawScope
from app.core.evidence import Evidence
from app.core.intent.intent_types import IntentResult


@dataclass
class ConversationState:
    """Holds the evolving context and artifacts for a FINO RAG conversation."""

    original_query: str
    intent_result: Optional[IntentResult] = None
    law_scope: Optional[LawScope] = None
    out_of_scope: bool = False
    selected_agent: Optional[str] = None
    evidences: List[Evidence] = field(default_factory=list)
    confidence: Optional[float] = None
