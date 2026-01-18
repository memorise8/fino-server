"""Base RAG agent implementation for FINO sub-agents."""

from __future__ import annotations

import logging
from typing import List

from app.config.llm import LLM_CONFIG
from app.config.rag import RAG_CONFIG
from app.core.agents.base import Agent, AgentOutput
from app.core.dsl.types import LawScope, Priority
from app.core.evidence import Evidence
from app.core.llm.llm_client import LLMClient
from app.core.rag.rag_pipeline import RAGPipeline, RAGResult
from app.core.state import ConversationState
from app.core.types import AgentResponse, IntentResult, RouteResult

logger = logging.getLogger(__name__)


class BaseRAGAgent(Agent):
    """Shared RAG behavior for FINO sub-agents with evidence-first answers."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        llm_client: LLMClient,
        *,
        agent_name: str,
        domain_label: str = "세무/회계",
        low_confidence_threshold: float = RAG_CONFIG["score_threshold"],
    ) -> None:
        self._rag_pipeline = rag_pipeline
        self._llm = llm_client
        self._agent_name = agent_name
        self._domain_label = domain_label
        self._low_confidence_threshold = low_confidence_threshold

    async def run(self, message: str, intent: IntentResult, route: RouteResult) -> AgentResponse:
        output = await self.execute(message, intent, route)
        context = self._build_context(message, output.evidences)
        return AgentResponse(
            data={
                "agent_name": self._agent_name,
                "answer": output.answer,
                "evidences": output.evidences,
                "confidence": output.confidence,
                "context": context,
            }
        )

    async def execute(self, message: str, intent: IntentResult, route: RouteResult) -> AgentOutput:
        logger.info("Agent execution start agent=%s", self._agent_name)
        try:
            law_scope = self._extract_law_scope(intent)
            out_of_scope = self._extract_out_of_scope(intent, law_scope)
            state = ConversationState(
                original_query=message,
                selected_agent=self._agent_name,
                law_scope=law_scope,
                out_of_scope=out_of_scope,
            )
            if law_scope is None:
                logger.info("Agent scope empty agent=%s", self._agent_name)
                return AgentOutput(
                    answer=self._general_out_of_scope_answer(),
                    evidences=[],
                    confidence=0.0,
                )
            rag_result = self._rag_pipeline.run(message, state)
            rag_result = self._filter_by_scope(rag_result, law_scope)
            if not rag_result.evidences:
                logger.warning("Agent fallback no evidence agent=%s", self._agent_name)
                return AgentOutput(
                    answer="근거 부족으로 답변 불가",
                    evidences=[],
                    confidence=0.0,
                )
            prompt = self._build_prompt(message, rag_result, law_scope)
            model_name = str(LLM_CONFIG.get("model", "unknown"))
            logger.info("LLM call agent=%s model=%s", self._agent_name, model_name)
            answer = (await self._llm.generate(prompt)).strip()
            if not answer:
                logger.warning("Agent fallback empty LLM response agent=%s", self._agent_name)
                answer = self._fallback_answer(rag_result)
            answer = self._apply_policies(answer, rag_result.evidences, rag_result.confidence)
            logger.info(
                "Agent execution end agent=%s confidence=%.3f",
                self._agent_name,
                rag_result.confidence,
            )
            return AgentOutput(
                answer=answer,
                evidences=rag_result.evidences,
                confidence=rag_result.confidence,
            )
        except Exception:
            logger.exception("Agent execution failure agent=%s", self._agent_name)
            raise

    def _build_prompt(self, query: str, rag_result: RAGResult, law_scope: LawScope) -> str:
        lines = [
            f"당신은 FINO {self._domain_label} 에이전트입니다.",
            "아래 근거만 사용해 답변하세요. 근거에 없는 내용은 답하지 마세요.",
            "",
            "[질의]",
            query,
            "",
            "[법령 범위]",
        ]
        for code in law_scope.law_codes:
            lines.append(f"- {code}")
        lines.extend(
            [
                "",
                "[근거]",
            ]
        )
        for evidence, chunk in zip(rag_result.evidences, rag_result.chunks):
            lines.append(f"- {self._format_citation(evidence)}: {chunk}")
        lines.extend(
            [
                "",
                "[작성 지침]",
                "* 근거 문장만 사용",
                "* 모호한 경우 불확실성을 명시",
            ]
        )
        return "\n".join(lines)

    def _fallback_answer(self, rag_result: RAGResult) -> str:
        lines = [f"{self._domain_label} 관련 근거를 바탕으로 안내드립니다."]
        for evidence, chunk in zip(rag_result.evidences, rag_result.chunks):
            lines.append(f"- {self._format_citation(evidence)}: {chunk}")
        return "\n".join(lines)

    def _build_context(self, query: str, evidences: List[Evidence]) -> str:
        lines = [f"[질의] {query}", "[근거 요약]"]
        for evidence in evidences:
            lines.append(f"- {self._format_citation(evidence)}")
        return "\n".join(lines)

    def _filter_by_scope(self, rag_result: RAGResult, law_scope: LawScope) -> RAGResult:
        allowed = set(law_scope.law_codes)
        filtered = [
            (evidence, chunk)
            for evidence, chunk in zip(rag_result.evidences, rag_result.chunks)
            if evidence.law_code in allowed
        ]
        if not filtered:
            return RAGResult(evidences=[], chunks=[], confidence=0.0)
        evidences, chunks = zip(*filtered)
        return RAGResult(
            evidences=list(evidences),
            chunks=list(chunks),
            confidence=rag_result.confidence,
        )

    def _apply_policies(
        self, answer: str, evidences: List[Evidence], confidence: float
    ) -> str:
        if self._has_conflict(evidences):
            answer = f"{answer}\n\n{self._conflict_message()}"
        if confidence < self._low_confidence_threshold:
            logger.warning(
                "Low confidence agent=%s confidence=%.3f",
                self._agent_name,
                confidence,
            )
            answer = f"{answer}\n\n{self._low_confidence_message()}"
        return answer

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

    @staticmethod
    def _extract_law_scope(intent: IntentResult) -> LawScope | None:
        if not isinstance(intent.data, dict):
            return None
        law_scope = intent.data.get("law_scope")
        if isinstance(law_scope, LawScope):
            return law_scope
        if isinstance(law_scope, dict):
            intent_value = str(law_scope.get("intent", "")).strip()
            law_codes_value = law_scope.get("law_codes")
            priority_value = law_scope.get("priority")
            if not intent_value or not isinstance(law_codes_value, list) or priority_value is None:
                return None
            try:
                priority = (
                    priority_value
                    if isinstance(priority_value, Priority)
                    else Priority.from_value(str(priority_value))
                )
            except ValueError:
                return None
            law_codes = [str(code).strip() for code in law_codes_value if str(code).strip()]
            if not law_codes:
                return None
            return LawScope(intent=intent_value, law_codes=law_codes, priority=priority)
        return None

    @staticmethod
    def _extract_out_of_scope(intent: IntentResult, law_scope: LawScope | None) -> bool:
        if not isinstance(intent.data, dict):
            return law_scope is None
        out_of_scope = intent.data.get("out_of_scope")
        if isinstance(out_of_scope, bool):
            return out_of_scope
        return law_scope is None

    def _general_out_of_scope_answer(self) -> str:
        return (
            "요청하신 내용에 해당하는 법령 범위를 확인할 수 없어 일반적인 안내만 제공합니다. "
            "구체적인 법령 적용은 전문가 확인이 필요합니다."
        )
