"""Deterministic query normalization for domain DSL matching in FINO."""

from __future__ import annotations

from typing import Dict, Tuple

NORMALIZATION_MAP: Dict[str, str] = {
    "부가가치세": "부가세",
    "종합소득세": "종소세",
    "법인 소득세": "법인세",
    "근로 소득": "근로소득",
}

_NORMALIZATION_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    sorted(
        (
            (key, value)
            for key, value in NORMALIZATION_MAP.items()
            if isinstance(key, str)
            and isinstance(value, str)
            and key.strip()
            and value.strip()
            and key.strip() != value.strip()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def normalize(query: str) -> str:
    """Return a normalized query string using one-way canonical replacements."""

    if not query:
        return ""
    normalized = query.strip()
    for source, target in _NORMALIZATION_PAIRS:
        if source in normalized:
            normalized = normalized.replace(source, target)
    return normalized


class QueryNormalizer:
    """Query normalization facade for orchestrator usage."""

    @staticmethod
    def normalize(query: str) -> str:
        return normalize(query)
