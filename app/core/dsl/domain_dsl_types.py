"""Domain-style DSL types for FINO scope adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SubDomain:
    """Keyword group bound to a single law code."""

    law_code: str
    keywords: List[str]


@dataclass(frozen=True)
class Domain:
    """Domain with optional subdomains and domain-level keywords."""

    domain: str
    description: Optional[str]
    keywords: List[str]
    subdomains: List[SubDomain]


@dataclass(frozen=True)
class DomainRule:
    """Deterministic rules for domain selection."""

    priority: List[str]
    min_keyword_match: int
    fallback: Optional[str]


@dataclass(frozen=True)
class DomainDSL:
    """Parsed domain-style DSL payload."""

    version: Optional[str]
    description: Optional[str]
    domains: List[Domain]
    rules: DomainRule
