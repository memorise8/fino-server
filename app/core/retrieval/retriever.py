"""Retrieval contracts and primitives for FINO RAG agents."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, List, Protocol

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣]+")


def default_tokenizer(text: str) -> List[str]:
    """Tokenize text using Korean-aware alphanumeric segments."""

    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class RetrievalDocument:
    """Text chunk with legal metadata for FINO retrieval."""

    text: str
    law_code: str
    article: str
    paragraph: str
    source_id: str


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved chunk with score and legal metadata."""

    text: str
    law_code: str
    article: str
    paragraph: str
    source_id: str
    score: float


class Retriever(Protocol):
    """Retrieval interface shared by FINO sub-agents."""

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        ...
