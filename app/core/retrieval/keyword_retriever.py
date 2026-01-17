"""Keyword-based retrieval for FINO RAG agents."""

from __future__ import annotations

from collections import Counter
import math
from typing import Callable, List, Sequence

from app.core.retrieval.retriever import (
    RetrievalDocument,
    RetrievalResult,
    default_tokenizer,
)


class KeywordRetriever:
    """BM25 keyword retriever for Korean legal documents."""

    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        tokenizer: Callable[[str], List[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._documents = list(documents)
        self._tokenizer = tokenizer or default_tokenizer
        self._k1 = k1
        self._b = b
        self._doc_tokens = [self._tokenizer(doc.text) for doc in self._documents]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_len = (sum(self._doc_lengths) / len(self._doc_lengths)) if self._doc_lengths else 0.0
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            self._doc_freq.update(set(tokens))
        self._num_docs = len(self._documents)

    def search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Return top-k BM25 matches for the query."""

        if not self._documents or top_k <= 0:
            return []
        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []
        query_counts = Counter(query_tokens)
        results: List[RetrievalResult] = []
        for doc, tokens, doc_len in zip(
            self._documents, self._doc_tokens, self._doc_lengths
        ):
            if doc_len == 0:
                continue
            score = self._score(tokens, doc_len, query_counts)
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

    def _score(self, tokens: List[str], doc_len: int, query_counts: Counter[str]) -> float:
        score = 0.0
        term_counts = Counter(tokens)
        for term, qf in query_counts.items():
            if term not in term_counts:
                continue
            idf = self._idf(term)
            tf = term_counts[term]
            denom = tf + self._k1 * (1 - self._b + self._b * (doc_len / self._avg_len))
            score += idf * ((tf * (self._k1 + 1)) / denom) * qf
        return score

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (self._num_docs - df + 0.5) / (df + 0.5))
