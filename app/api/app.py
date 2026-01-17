"""FastAPI entry point for the FINO assistant service."""

from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from app.core.agents.accounting_agent import AccountingAgent
from app.core.agents.general_agent import GeneralAgent
from app.core.agents.registry import AgentRegistry
from app.core.agents.tax_agent import TaxAgent
from app.core.intent.deterministic_intent import DeterministicIntentAnalyzer
from app.core.integrator import Integrator
from app.core.llm.llm_client import StubLLMClient
from app.core.pipeline.orchestrator import BackboneOrchestrator
from app.core.rag.rag_pipeline import RAGPipeline
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.keyword_retriever import KeywordRetriever
from app.core.retrieval.retriever import RetrievalDocument
from app.core.retrieval.reranker import ScoreReranker
from app.core.retrieval.vector_retriever import VectorRetriever
from app.core.routing.agent_selector import AgentSelector
from app.core.routing.deterministic_router import DeterministicRouter

app = FastAPI(title="FINO AI Assistant")


class ChatRequest(BaseModel):
    """Request payload for FINO chat."""

    query: str


class ChatResponse(BaseModel):
    """Response payload for FINO chat."""

    answer: str
    agents: List[str]
    confidence: float


def _build_documents() -> Dict[str, List[RetrievalDocument]]:
    tax_docs = [
        RetrievalDocument(
            text="접대비는 법인세법상 손금산입 한도 내에서 인정되며 한도 초과분은 손금불산입된다.",
            law_code="법인세법",
            article="제25조",
            paragraph="1항",
            source_id="tax-001",
        ),
        RetrievalDocument(
            text="접대비 지출은 적격 증빙을 갖추어야 하며 증빙 미비 시 비용 인정이 제한될 수 있다.",
            law_code="법인세법",
            article="제116조",
            paragraph="2항",
            source_id="tax-002",
        ),
    ]
    accounting_docs = [
        RetrievalDocument(
            text="접대비는 판매비와관리비로 분류하여 기간 비용으로 인식한다.",
            law_code="기업회계기준",
            article="제5장",
            paragraph="1절",
            source_id="acc-001",
        ),
        RetrievalDocument(
            text="세무조정이 필요한 항목은 재무제표 주석과 조정명세서에 반영한다.",
            law_code="기업회계기준",
            article="제7장",
            paragraph="3절",
            source_id="acc-002",
        ),
    ]
    general_docs = [
        RetrievalDocument(
            text="세무와 회계는 목적이 달라 동일한 거래라도 인식 기준이 달라질 수 있다.",
            law_code="일반지침",
            article="서론",
            paragraph="1절",
            source_id="gen-001",
        )
    ]
    return {
        "tax": tax_docs,
        "accounting": accounting_docs,
        "general": general_docs,
    }


def _build_rag_pipeline(documents: List[RetrievalDocument]) -> RAGPipeline:
    vector_retriever = VectorRetriever(documents)
    keyword_retriever = KeywordRetriever(documents)
    hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever, vector_weight=0.6, keyword_weight=0.4)
    reranker = ScoreReranker()
    return RAGPipeline(retriever=hybrid_retriever, reranker=reranker, top_k=3)


def _build_orchestrator() -> BackboneOrchestrator:
    documents = _build_documents()
    llm_client = StubLLMClient()
    registry = AgentRegistry()
    registry.register("tax", TaxAgent(_build_rag_pipeline(documents["tax"]), llm_client))
    registry.register(
        "accounting",
        AccountingAgent(_build_rag_pipeline(documents["accounting"]), llm_client),
    )
    registry.register("general", GeneralAgent(_build_rag_pipeline(documents["general"]), llm_client))
    selector = AgentSelector(registry)
    intent_analyzer = DeterministicIntentAnalyzer()
    router = DeterministicRouter()
    integrator = Integrator()
    return BackboneOrchestrator(
        intent_analyzer=intent_analyzer.analyze,
        router=router,
        agents=registry,
        integrator=integrator,
        agent_selector=selector,
    )


_ORCHESTRATOR = _build_orchestrator()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle FINO chat requests using deterministic routing and agent execution."""

    final_response = await _ORCHESTRATOR.run(request.query)
    payload = final_response.responses[0].data if final_response.responses else {}
    answer = str(payload.get("answer") or "근거 부족으로 답변 불가")
    agents = list(payload.get("agents") or [])
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return ChatResponse(answer=answer, agents=agents, confidence=confidence)
