"""Loader for law scope DSL used by FINO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from app.config.dsl import DSL_CONFIG
from app.core.dsl.types import LawScope, Priority

logger = logging.getLogger(__name__)

_DSL_CACHE: Dict[str, LawScope] | None = None


def get_scope(intent: str) -> LawScope | None:
    """Return the LawScope for an intent, or None if not defined."""

    if not intent:
        return None
    scopes = _load_dsl()
    return scopes.get(intent.strip().lower())


def _load_dsl() -> Dict[str, LawScope]:
    global _DSL_CACHE
    if _DSL_CACHE is not None:
        return _DSL_CACHE
    path = Path(str(DSL_CONFIG["law_scope_path"]))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Law scope DSL not found path=%s", path)
        _DSL_CACHE = {}
        return _DSL_CACHE
    except json.JSONDecodeError:
        logger.error("Law scope DSL invalid JSON path=%s", path)
        _DSL_CACHE = {}
        return _DSL_CACHE
    if not isinstance(raw, dict):
        logger.error("Law scope DSL must be a JSON object path=%s", path)
        _DSL_CACHE = {}
        return _DSL_CACHE
    _DSL_CACHE = _parse_scopes(raw)
    return _DSL_CACHE


def _parse_scopes(raw: Dict[str, object]) -> Dict[str, LawScope]:
    scopes: Dict[str, LawScope] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            logger.warning("Skipping invalid scope entry key=%s", key)
            continue
        law_codes = _coerce_law_codes(value.get("law_codes"))
        priority_value = value.get("priority")
        if law_codes is None or priority_value is None:
            logger.warning("Skipping scope entry missing fields intent=%s", key)
            continue
        try:
            priority = Priority.from_value(str(priority_value))
        except ValueError:
            logger.warning("Skipping scope entry invalid priority intent=%s", key)
            continue
        intent_key = key.strip().lower()
        scopes[intent_key] = LawScope(
            intent=key.strip(),
            law_codes=law_codes,
            priority=priority,
        )
    return scopes


def _coerce_law_codes(value: object) -> List[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    codes = [str(item).strip() for item in value if str(item).strip()]
    return codes or None
