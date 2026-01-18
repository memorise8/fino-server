"""Adapter that resolves domain-style DSL into LawScope."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List

from app.core.dsl.domain_dsl_loader import load_domain_dsl
from app.core.dsl.domain_dsl_types import Domain, DomainDSL
from app.core.dsl.types import LawScope, Priority

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DomainMatch:
    """Internal match result for a domain."""

    keyword_hits: int
    law_codes: List[str]


class DomainScopeAdapter:
    """Resolve LawScope from the domain-style DSL without LLM fallback."""

    def __init__(self, loader: Callable[[], DomainDSL] | None = None) -> None:
        self._loader = loader or load_domain_dsl

    def resolve(self, query: str) -> LawScope | None:
        """Return a normalized LawScope or None when out of scope."""

        if not query:
            return None
        dsl = self._loader()
        if not dsl.domains:
            return None
        normalized = query.lower()
        matches = self._collect_matches(normalized, dsl.domains)
        if not matches:
            return None
        for domain_key in dsl.rules.priority:
            match = matches.get(domain_key.lower())
            if match and match.keyword_hits >= dsl.rules.min_keyword_match:
                if domain_key.lower() == "none":
                    logger.info("Domain scope resolved to none")
                    return None
                if not match.law_codes:
                    logger.info("Domain scope has no matched law codes domain=%s", domain_key)
                    return None
                return LawScope(
                    intent=domain_key,
                    law_codes=match.law_codes,
                    priority=self._priority_for(domain_key),
                )
        return None

    def _collect_matches(self, query: str, domains: List[Domain]) -> Dict[str, _DomainMatch]:
        matches: Dict[str, _DomainMatch] = {}
        for domain in domains:
            keyword_set = set()
            for keyword in domain.keywords:
                if keyword and keyword in query:
                    keyword_set.add(keyword)
            law_codes: List[str] = []
            for subdomain in domain.subdomains:
                sub_hits = [kw for kw in subdomain.keywords if kw and kw in query]
                if sub_hits:
                    law_codes.append(subdomain.law_code)
                    keyword_set.update(sub_hits)
            if keyword_set:
                matches[domain.domain.lower()] = _DomainMatch(
                    keyword_hits=len(keyword_set),
                    law_codes=self._unique_preserve(law_codes),
                )
        return matches

    @staticmethod
    def _priority_for(domain: str) -> Priority:
        normalized = domain.strip().lower()
        if normalized == "accounting":
            return Priority.FORM
        return Priority.SUBSTANCE

    @staticmethod
    def _unique_preserve(items: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output
