"""LLM client contracts and stub implementation for FINO agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


class LLMClient(Protocol):
    """LLM client interface used by FINO agents."""

    async def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class StubLLMClient:
    """Deterministic LLM stub that summarizes evidence lines."""

    name: str = "stub-llm"

    async def generate(self, prompt: str) -> str:
        evidence_lines = self._extract_evidence(prompt)
        if not evidence_lines:
            return "근거가 부족하여 생성할 수 없습니다."
        summary = "\n".join(f"- {line}" for line in evidence_lines)
        return f"근거를 종합한 답변입니다.\n{summary}"

    @staticmethod
    def _extract_evidence(prompt: str) -> List[str]:
        lines = prompt.splitlines()
        evidence: List[str] = []
        in_section = False
        for line in lines:
            stripped = line.strip()
            if stripped == "[근거]":
                in_section = True
                continue
            if in_section and stripped.startswith("[") and stripped.endswith("]"):
                break
            if in_section and stripped.startswith("- "):
                evidence.append(stripped[2:])
        return evidence
