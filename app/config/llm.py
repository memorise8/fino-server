"""LLM configuration values for FINO agents."""

from __future__ import annotations

from typing import Dict

from app.config.settings import settings

LLM_CONFIG: Dict[str, object] = {
    "api_key": settings.openai_api_key,
    "model": settings.openai_model,
    "temperature": settings.openai_temperature,
    "timeout": settings.openai_timeout,
}
