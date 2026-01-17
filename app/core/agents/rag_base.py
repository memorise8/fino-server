"""Base RAG agent implementation for FINO sub-agents."""

from __future__ import annotations

from typing import List, Tuple

from app.core.agents.base import Agent, AgentOutput
from app.core.evidence import Evidence
from app.core.rag.rag_pipeline import RAGPipeline, RAGResult
from app.core.state import ConversationState
from app.core.types import AgentResponse, IntentResult, RouteResult


class BaseRAGAgent(Agent):
    """Shared RAG behavior for FINO sub-agents with evidence-first responses."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        *,
        domain_label: str = "세무/회계",
        low_confidence_threshold: float = 0.45,
    ) -> None:
        self._rag_pipeline = rag_pipeline
        self._domain_label = domain_label
        self._low_confidence_threshold = low_confidence_threshold

    async def run(self, message: str, intent: IntentResult, route: RouteResult) -> AgentResponse:
        output, context = await self._execute_with_context(message, intent, route)
        return AgentResponse(
            data={
                "answer": output.answer,
                "evidences": output.evidences,
                "confidence": output.confidence,
                "context": context,
            }
        )

    async def execute(self, message: str, intent: IntentResult, route: RouteResult) -> AgentOutput:
        output, _context = await self._execute_with_context(message, intent, route)
        return output

    async def _execute_with_context(
        self, message: str, intent: IntentResult, route: RouteResult
    ) -> Tuple[AgentOutput, str]:
        state = ConversationState(
            original_query=message,
            selected_agent=self._domain_label,
        )
        rag_result = self._rag_pipeline.run(message, state)
        if not rag_result.evidences:
            return (
                AgentOutput(
                    answer="근거 부족으로 답변 불가",
                    evidences=[],
                    confidence=0.0,
                ),
                "",
            )
        state.evidences = rag_result.evidences
        state.confidence = rag_result.confidence
        context = self._build_context(message, rag_result)
        answer = self._compose_answer(message, rag_result, context)
        if self._has_conflict(rag_result.evidences):
            answer = f"{answer}\n\n{self._conflict_message()}"
        if rag_result.confidence < self._low_confidence_threshold:
            answer = f"{answer}\n\n{self._low_confidence_message()}"
        return (
            AgentOutput(
                answer=answer,
                evidences=rag_result.evidences,
                confidence=rag_result.confidence,
            ),
            context,
        )

    def _build_context(self, query: str, rag_result: RAGResult) -> str:
        lines = [f"[질의] {query}", "[근거]"]
        for evidence, chunk in zip(rag_result.evidences, rag_result.chunks):
            citation = self._format_citation(evidence)
            lines.append(f"- {citation}: {chunk}")
        return "\n".join(lines)

    def _compose_answer(self, query: str, rag_result: RAGResult, context: str) -> str:
        lines = [f"{self._domain_label} 관련 근거를 바탕으로 안내드립니다."]
        for evidence, chunk in zip(rag_result.evidences, rag_result.chunks):
            citation = self._format_citation(evidence)
            lines.append(f"- {citation}: {chunk}")
        return "\n".join(lines)

    @staticmethod
    def _format_citation(evidence: Evidence) -> str:
        parts = [evidence.law_code, evidence.article, evidence.paragraph]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _has_conflict(evidences: List[Evidence]) -> bool:
        if len(evidences) < 2:
            return False
        unique_refs = {(evidence.law_code, evidence.article) for evidence in evidences}
        return len(unique_refs) > 1

    @staticmethod
    def _conflict_message() -> str:
        return "근거가 상충될 가능성이 있어 해석에 불확실성이 있습니다."

    @staticmethod
    def _low_confidence_message() -> str:
        return "신뢰도가 낮아 참고용으로만 활용해 주세요."
