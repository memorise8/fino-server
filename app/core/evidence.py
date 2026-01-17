"""Evidence contract for FINO RAG answers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    """Canonical evidence pointer required by FINO agents."""

    law_code: str
    article: str
    paragraph: str
    source_id: str
