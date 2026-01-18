"""Debug entry point to run the FINO FastAPI service."""

from __future__ import annotations

import sys

from app.api.app import app
from app.config.settings import settings


def main() -> int:
    """Launch the FINO API server for local debugging."""

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required to run the debug server.", file=sys.stderr)
        return 1
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=str(settings.log_level).lower(),
        reload=settings.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
