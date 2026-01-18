"""DSL types for law scope resolution in FINO."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Priority(str, Enum):
    """Priority ordering for interpreting legal scope."""

    SUBSTANCE = "substance"
    FORM = "form"

    @classmethod
    def from_value(cls, value: str) -> "Priority":
        normalized = value.strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(f"Unknown priority: {value}")


@dataclass(frozen=True)
class LawScope:
    """Resolved legal scope tied to a conversation intent."""

    intent: str
    law_codes: List[str]
    priority: Priority
