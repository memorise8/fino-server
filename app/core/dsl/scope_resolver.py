"""Deterministic law scope resolver for FINO."""

from __future__ import annotations

from app.core.dsl.dsl_loader import get_scope
from app.core.dsl.types import LawScope
from app.core.types import IntentResult


class ScopeResolver:
    """Resolve LawScope from an intent payload without fallback guessing."""

    def resolve(self, intent: IntentResult) -> LawScope | None:
        if not isinstance(intent.data, dict):
            return None
        primary_intent = str(intent.data.get("primary_intent", "")).strip().lower()
        if not primary_intent:
            return None
        return get_scope(primary_intent)
