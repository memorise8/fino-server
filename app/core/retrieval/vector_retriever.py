"""In-memory vector retrieval for FINO RAG agents."""

from __future__ import annotations

from collections import Counter
import math
from typing import Callable, Dict, List, Sequence

from app.core.retrieval.retriever import (
    RetrievalDocument,
    RetrievalResult,
    default_tokenizer,
)


class VectorRetriever:
    """TF-IDF vector retriever suited for Korean legal text chunks."""

    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        tokenizer: Callable[[str], List[str]] | None = None,
    ) -> None:
        self._documents = list(documents)
        self._tokenizer = tokenizer or default_tokenizer
        self._doc_tokens = [self._tokenizer(doc.text) for doc in self._documents]
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            self._doc_freq.update(set(tokens))
        self._num_docs = len(self._documents)
        self._doc_vectors = [self._build_vector(tokens) for tokens in self._doc_tokens]
        self._doc_norms = [self._norm(vec) for vec in self._doc_vectors]

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Return top-k cosine similarity matches for the query."""

        if not self._documents or top_k <= 0:
            return []
        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []
        query_vector = self._build_vector(query_tokens)
        query_norm = self._norm(query_vector)
        if query_norm == 0.0:
            return []
        results: List[RetrievalResult] = []
        for doc, doc_vector, doc_norm in zip(
            self._documents, self._doc_vectors, self._doc_norms
        ):
            if doc_norm == 0.0:
                continue
            score = self._dot(query_vector, doc_vector) / (query_norm * doc_norm)
            if score <= 0.0:
                continue
            results.append(
                RetrievalResult(
                    text=doc.text,
                    law_code=doc.law_code,
                    article=doc.article,
                    paragraph=doc.paragraph,
                    source_id=doc.source_id,
                    score=score,
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def _build_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        term_counts = Counter(tokens)
        length = len(tokens)
        return {
            term: (count / length) * self._idf(term)
            for term, count in term_counts.items()
        }

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log((1 + self._num_docs) / (1 + df)) + 1.0

    @staticmethod
    def _dot(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vec.values()))
