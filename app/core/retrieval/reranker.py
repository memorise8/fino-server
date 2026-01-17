"""Re-ranking utilities for FINO RAG retrieval results."""

from __future__ import annotations

from typing import Callable, List, Protocol, Sequence

from app.core.retrieval.retriever import RetrievalResult, default_tokenizer


class Reranker(Protocol):
    """Re-rank retrieval results based on query relevance."""

    def rerank(
        self, query: str, results: Sequence[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        ...


class ScoreReranker:
    """Combine retriever scores with token overlap for stable ranking."""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] | None = None,
        score_weight: float = 0.7,
        overlap_weight: float = 0.3,
    ) -> None:
        total = score_weight + overlap_weight
        if total <= 0.0:
            raise ValueError("Reranker weights must sum to a positive value.")
        self._score_weight = score_weight / total
        self._overlap_weight = overlap_weight / total
        self._tokenizer = tokenizer or default_tokenizer

    def rerank(
        self, query: str, results: Sequence[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        if top_k <= 0:
            return []
        if not results:
            return []
        query_tokens = set(self._tokenizer(query))
        if not query_tokens:
            return list(results)[:top_k]
        reranked: List[RetrievalResult] = []
        for result in results:
            result_tokens = set(self._tokenizer(result.text))
            overlap = (
                len(query_tokens & result_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            score = self._score_weight * result.score + self._overlap_weight * overlap
            reranked.append(
                RetrievalResult(
                    text=result.text,
                    law_code=result.law_code,
                    article=result.article,
                    paragraph=result.paragraph,
                    source_id=result.source_id,
                    score=score,
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
