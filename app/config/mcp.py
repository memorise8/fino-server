"""MCP integration configuration for FINO."""

from __future__ import annotations

from typing import Dict

from app.config.settings import settings

MCP_CONFIG: Dict[str, object] = {
    "enabled": settings.mcp_enabled,
    "endpoint": settings.mcp_endpoint,
}
