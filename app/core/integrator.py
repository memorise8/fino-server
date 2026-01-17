"""Response integration utilities for FINO multi-agent answers."""

from __future__ import annotations

from typing import Iterable, List

from app.core.types import AgentResponse, FinalResponse


class Integrator:
    """Merge multi-agent answers while preserving agent identity."""

    def integrate(self, responses: Iterable[AgentResponse]) -> FinalResponse:
        response_list = list(responses)
        per_agent = [
            self._normalize_response(response, index)
            for index, response in enumerate(response_list)
        ]
        merged_answer = self._merge_answers(per_agent)
        agent_names = [item["agent_name"] for item in per_agent]
        final_confidence = self._merge_confidence(per_agent)
        final_payload = {
            "answer": merged_answer,
            "agents": agent_names,
            "confidence": final_confidence,
            "per_agent": per_agent,
        }
        return FinalResponse(responses=[AgentResponse(data=final_payload)])

    @staticmethod
    def _normalize_response(response: AgentResponse, index: int) -> dict:
        payload = dict(response.data)
        agent_name = str(payload.get("agent_name") or f"agent_{index + 1}")
        answer = str(payload.get("answer") or "")
        evidences = payload.get("evidences", [])
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "agent_name": agent_name,
            "answer": answer,
            "confidence": confidence,
            "evidences": evidences,
        }

    @staticmethod
    def _merge_answers(per_agent: List[dict]) -> str:
        if not per_agent:
            return "근거 부족으로 답변 불가"
        if len(per_agent) == 1:
            return per_agent[0]["answer"] or "근거 부족으로 답변 불가"
        merged_lines = []
        for item in per_agent:
            answer = item["answer"] or "근거 부족으로 답변 불가"
            merged_lines.append(f"[{item['agent_name']}] {answer}")
        return "\n\n".join(merged_lines)

    @staticmethod
    def _merge_confidence(per_agent: List[dict]) -> float:
        if not per_agent:
            return 0.0
        confidences = [item["confidence"] for item in per_agent]
        return sum(confidences) / len(confidences)
