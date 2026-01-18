"""Loader for domain-style law scope DSL used by FINO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from app.config.dsl import DSL_CONFIG
from app.core.dsl.domain_dsl_types import Domain, DomainDSL, DomainRule, SubDomain

logger = logging.getLogger(__name__)

_DSL_CACHE: DomainDSL | None = None

_DEFAULT_PRIORITY = ["none", "tax", "accounting"]
_DEFAULT_MIN_MATCH = 1


def load_domain_dsl() -> DomainDSL:
    """Load and cache the domain-style DSL definition."""

    global _DSL_CACHE
    if _DSL_CACHE is not None:
        return _DSL_CACHE
    path = Path(str(DSL_CONFIG["law_scope_path"]))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Domain DSL not found path=%s", path)
        _DSL_CACHE = _empty_dsl()
        return _DSL_CACHE
    except json.JSONDecodeError:
        logger.error("Domain DSL invalid JSON path=%s", path)
        _DSL_CACHE = _empty_dsl()
        return _DSL_CACHE
    if not isinstance(raw, dict):
        logger.error("Domain DSL must be a JSON object path=%s", path)
        _DSL_CACHE = _empty_dsl()
        return _DSL_CACHE
    domains = _parse_domains(raw.get("domains"))
    rules = _parse_rules(raw.get("rules"))
    _DSL_CACHE = DomainDSL(
        version=_coerce_str(raw.get("version")),
        description=_coerce_str(raw.get("description")),
        domains=domains,
        rules=rules,
    )
    return _DSL_CACHE


def _empty_dsl() -> DomainDSL:
    return DomainDSL(
        version=None,
        description=None,
        domains=[],
        rules=_default_rules(),
    )


def _default_rules() -> DomainRule:
    return DomainRule(
        priority=list(_DEFAULT_PRIORITY),
        min_keyword_match=_DEFAULT_MIN_MATCH,
        fallback=None,
    )


def _parse_rules(raw: object) -> DomainRule:
    if not isinstance(raw, dict):
        return _default_rules()
    decision = raw.get("domain_decision")
    if not isinstance(decision, dict):
        return _default_rules()
    priority = _coerce_str_list(decision.get("priority")) or list(_DEFAULT_PRIORITY)
    min_keyword_match = _coerce_int(decision.get("min_keyword_match"), _DEFAULT_MIN_MATCH)
    fallback = _coerce_str(decision.get("fallback"))
    return DomainRule(
        priority=priority,
        min_keyword_match=min_keyword_match,
        fallback=fallback,
    )


def _parse_domains(raw: object) -> List[Domain]:
    if not isinstance(raw, list):
        return []
    domains: List[Domain] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("Skipping invalid domain entry type=%s", type(entry).__name__)
            continue
        name = _coerce_str(entry.get("domain"))
        if not name:
            logger.warning("Skipping domain entry missing name")
            continue
        description = _coerce_str(entry.get("description"))
        keywords = _coerce_keywords(entry.get("keywords"))
        subdomains = _parse_subdomains(entry.get("subdomains"))
        domains.append(
            Domain(
                domain=name,
                description=description,
                keywords=keywords,
                subdomains=subdomains,
            )
        )
    return domains


def _parse_subdomains(raw: object) -> List[SubDomain]:
    if not isinstance(raw, list):
        return []
    subdomains: List[SubDomain] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("Skipping invalid subdomain entry type=%s", type(entry).__name__)
            continue
        law_code = _coerce_str(entry.get("law_code"))
        keywords = _coerce_keywords(entry.get("keywords"))
        if not law_code or not keywords:
            logger.warning("Skipping subdomain entry missing fields law_code=%s", law_code)
            continue
        subdomains.append(SubDomain(law_code=law_code, keywords=keywords))
    return subdomains


def _coerce_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_str_list(value: object) -> List[str] | None:
    if not isinstance(value, list):
        return None
    items = [str(item).strip().lower() for item in value if str(item).strip()]
    return items or None


def _coerce_keywords(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    keywords = [str(item).strip().lower() for item in value if str(item).strip()]
    return keywords


def _coerce_int(value: object, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default
