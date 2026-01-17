"""Tax-domain RAG agent for FINO."""

from __future__ import annotations

from app.core.agents.rag_base import BaseRAGAgent
from app.core.evidence import Evidence
from app.core.rag.rag_pipeline import RAGPipeline, RAGResult


class TaxAgent(BaseRAGAgent):
    """Agent specializing in Korean tax guidance using evidence-only RAG."""

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        super().__init__(rag_pipeline, domain_label="세무")

    def _compose_answer(self, query: str, rag_result: RAGResult, context: str) -> str:
        lines = ["세무 관련 근거를 바탕으로 안내드립니다."]
        for evidence, chunk in zip(rag_result.evidences, rag_result.chunks):
            lines.append(f"- {self._format_citation(evidence)}: {chunk}")
        return "\n".join(lines)

    @staticmethod
    def _format_citation(evidence: Evidence) -> str:
        parts = [evidence.law_code, evidence.article, evidence.paragraph]
        return " ".join(part for part in parts if part)
