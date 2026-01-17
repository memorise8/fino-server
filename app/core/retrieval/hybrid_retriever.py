"""Hybrid retriever combining vector and keyword search for FINO RAG."""

from __future__ import annotations

from typing import Dict, List

from app.core.retrieval.retriever import RetrievalResult, Retriever


class HybridRetriever:
    """Merge vector and keyword retrieval results with weighted scores."""

    def __init__(
        self,
        vector_retriever: Retriever,
        keyword_retriever: Retriever,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ) -> None:
        if vector_weight < 0.0 or keyword_weight < 0.0:
            raise ValueError("Retriever weights must be non-negative.")
        if vector_weight + keyword_weight == 0.0:
            raise ValueError("At least one retriever weight must be positive.")
        total = vector_weight + keyword_weight
        self._vector_weight = vector_weight / total
        self._keyword_weight = keyword_weight / total
        self._vector = vector_retriever
        self._keyword = keyword_retriever

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Return top-k merged results from both retrievers."""

        if top_k <= 0:
            return []
        vector_results = self._vector.search(query, top_k * 2)
        keyword_results = self._keyword.search(query, top_k * 2)
        if not vector_results and not keyword_results:
            return []
        vector_norm = self._normalize(vector_results)
        keyword_norm = self._normalize(keyword_results)
        combined_scores: Dict[str, float] = {}
        result_by_id: Dict[str, RetrievalResult] = {}
        for result in vector_results:
            result_by_id.setdefault(result.source_id, result)
            combined_scores[result.source_id] = (
                combined_scores.get(result.source_id, 0.0)
                + self._vector_weight * vector_norm.get(result.source_id, 0.0)
            )
        for result in keyword_results:
            result_by_id.setdefault(result.source_id, result)
            combined_scores[result.source_id] = (
                combined_scores.get(result.source_id, 0.0)
                + self._keyword_weight * keyword_norm.get(result.source_id, 0.0)
            )
        merged: List[RetrievalResult] = []
        for source_id, result in result_by_id.items():
            score = combined_scores.get(source_id, 0.0)
            if score <= 0.0:
                continue
            merged.append(
                RetrievalResult(
                    text=result.text,
                    law_code=result.law_code,
                    article=result.article,
                    paragraph=result.paragraph,
                    source_id=result.source_id,
                    score=score,
                )
            )
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    @staticmethod
    def _normalize(results: List[RetrievalResult]) -> Dict[str, float]:
        if not results:
            return {}
        max_score = max(result.score for result in results)
        if max_score <= 0.0:
            return {}
        return {result.source_id: result.score / max_score for result in results}
