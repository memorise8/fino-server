"""Retrieval-only pipeline used inside FINO RAG agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.evidence import Evidence
from app.core.retrieval.reranker import Reranker
from app.core.retrieval.retriever import RetrievalResult, Retriever
from app.core.state import ConversationState


@dataclass(frozen=True)
class RAGResult:
    """Container for retrieved evidence and text snippets."""

    evidences: List[Evidence]
    chunks: List[str]
    confidence: float


class RAGPipeline:
    """Shared retrieval pipeline for FINO sub-agents without LLM calls."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        top_k: int = 5,
        prefetch_k: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._top_k = top_k
        self._prefetch_k = prefetch_k or max(top_k * 2, top_k)

    def run(self, query: str, state: ConversationState) -> RAGResult:
        """Retrieve and rerank evidence for a query within a conversation."""

        try:
            candidates = self._retriever.search(query, self._prefetch_k)
        except Exception:
            return RAGResult(evidences=[], chunks=[], confidence=0.0)
        if not candidates:
            return RAGResult(evidences=[], chunks=[], confidence=0.0)
        if self._reranker is None:
            reranked = list(candidates)[: self._top_k]
        else:
            reranked = self._reranker.rerank(query, candidates, self._top_k)
        evidences = [self._to_evidence(result) for result in reranked]
        chunks = [result.text for result in reranked]
        confidence = self._estimate_confidence(reranked)
        return RAGResult(evidences=evidences, chunks=chunks, confidence=confidence)

    @staticmethod
    def _to_evidence(result: RetrievalResult) -> Evidence:
        return Evidence(
            law_code=result.law_code,
            article=result.article,
            paragraph=result.paragraph,
            source_id=result.source_id,
        )

    @staticmethod
    def _estimate_confidence(results: List[RetrievalResult]) -> float:
        if not results:
            return 0.0
        max_score = max(result.score for result in results)
        if max_score <= 0.0:
            return 0.0
        normalized = [result.score / max_score for result in results]
        return min(1.0, sum(normalized) / len(normalized))
