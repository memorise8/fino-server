"""Deterministic intent analysis for FINO service routing."""

from __future__ import annotations

from typing import Iterable, List

from app.core.types import IntentResult


class DeterministicIntentAnalyzer:
    """Rule-based intent analyzer tailored to Korean tax and accounting queries."""

    def __init__(
        self,
        tax_keywords: Iterable[str] | None = None,
        accounting_keywords: Iterable[str] | None = None,
    ) -> None:
        self._tax_keywords = list(
            tax_keywords
            or [
                "세무",
                "세금",
                "부가가치세",
                "부가세",
                "법인세",
                "소득세",
                "원천세",
                "접대비",
                "세무조사",
            ]
        )
        self._accounting_keywords = list(
            accounting_keywords
            or [
                "회계",
                "분개",
                "재무제표",
                "손익계산서",
                "대차대조표",
                "감가상각",
                "원가",
                "결산",
            ]
        )

    async def analyze(self, message: str) -> IntentResult:
        """Return a structured intent payload using keyword rules."""

        normalized = message.lower()
        tax_hit = self._has_keyword(normalized, self._tax_keywords)
        accounting_hit = self._has_keyword(normalized, self._accounting_keywords)
        if tax_hit and accounting_hit:
            primary_intent = "both"
            secondary_intents = ["tax", "accounting"]
            confidence = 0.9
        elif tax_hit:
            primary_intent = "tax"
            secondary_intents = []
            confidence = 0.85
        elif accounting_hit:
            primary_intent = "accounting"
            secondary_intents = []
            confidence = 0.85
        else:
            primary_intent = "etc"
            secondary_intents = []
            confidence = 0.5
        return IntentResult(
            data={
                "primary_intent": primary_intent,
                "secondary_intents": secondary_intents,
                "confidence": confidence,
            }
        )

    @staticmethod
    def _has_keyword(text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)
