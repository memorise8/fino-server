"""General-domain RAG agent for FINO."""

from __future__ import annotations

from app.core.agents.rag_base import BaseRAGAgent
from app.core.llm.llm_client import LLMClient
from app.core.rag.rag_pipeline import RAGPipeline


class GeneralAgent(BaseRAGAgent):
    """Agent for general FINO guidance when intent is not specialized."""

    def __init__(self, rag_pipeline: RAGPipeline, llm_client: LLMClient) -> None:
        super().__init__(
            rag_pipeline,
            llm_client,
            agent_name="general",
            domain_label="일반",
        )
