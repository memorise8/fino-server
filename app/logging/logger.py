"""Central logging configuration and request_id context for FINO."""

from __future__ import annotations

import logging
import logging.config
from contextvars import ContextVar, Token

from app.config.settings import settings

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Inject request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def bind_request_id(request_id: str) -> Token:
    """Bind request_id to the current context."""

    return _request_id.set(request_id)


def reset_request_id(token: Token) -> None:
    """Reset request_id to the previous context value."""

    _request_id.reset(token)


def setup_logging() -> None:
    """Configure structured logging for FINO."""

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_id": {"()": RequestIdFilter},
            },
            "formatters": {
                "standard": {
                    "format": "[%(asctime)s][%(levelname)s][%(name)s][request_id=%(request_id)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "filters": ["request_id"],
                }
            },
            "root": {
                "handlers": ["console"],
                "level": settings.log_level,
            },
        }
    )
