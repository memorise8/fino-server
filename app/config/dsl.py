"""DSL configuration for FINO law scope and guide assets."""

from __future__ import annotations

from typing import Dict

from app.config.settings import settings

DSL_CONFIG: Dict[str, str] = {
    "law_scope_path": settings.law_scope_dsl_path,
    "guide_index_path": settings.guide_index_path,
}
