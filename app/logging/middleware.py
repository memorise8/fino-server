"""Request ID middleware for FINO FastAPI services."""

from __future__ import annotations

import logging
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging.logger import bind_request_id, reset_request_id

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach request_id to each HTTP request and response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid4())
        token = bind_request_id(request_id)
        request.state.request_id = request_id
        logger.info("Incoming request method=%s path=%s", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Request failed")
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            logger.info("Response sent status=%s", response.status_code)
            return response
        finally:
            reset_request_id(token)
