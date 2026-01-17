from __future__ import annotations

from typing import Awaitable, Callable, Sequence

from app.config.settings import settings
from app.core.pipeline.orchestrator import BackboneOrchestrator

BACKBONE_ENABLED = settings.backbone_enabled


async def chat(
    message: str,
    *,
    backbone_orchestrator: BackboneOrchestrator | None = None,
    agent_names: Sequence[str] | None = None,
    legacy_handler: Callable[[str], Awaitable[object]] | None = None,
) -> object:
    if BACKBONE_ENABLED:
        if backbone_orchestrator is None:
            raise NotImplementedError("TODO: Provide BackboneOrchestrator instance")
        if agent_names is None:
            raise NotImplementedError("TODO: Provide agent names for BackboneOrchestrator")
        return await backbone_orchestrator.run(message, agent_names)
    if legacy_handler is None:
        raise NotImplementedError("TODO: Provide legacy chat handler")
    return await legacy_handler(message)
