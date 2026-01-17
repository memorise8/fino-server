"""Retrieval configuration for FINO RAG pipelines."""

from __future__ import annotations

from typing import Dict

from app.config.settings import settings

RAG_CONFIG: Dict[str, object] = {
    "backend": settings.rag_backend,
    "top_k": settings.rag_top_k,
    "score_threshold": settings.rag_score_threshold,
}
