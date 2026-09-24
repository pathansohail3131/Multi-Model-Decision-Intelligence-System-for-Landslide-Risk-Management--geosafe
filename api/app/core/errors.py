"""Error types and the API's single error-response shape.

Every failure reaches the client as ``{"error": {"code", "message", ...}}`` so
the frontend can branch on ``code`` instead of parsing prose.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.utils.logging_setup import get_logger

__all__ = [
    "ServiceError",
    "ServiceNotReady",
    "Utf8JSONResponse",
    "error_response",
    "register_exception_handlers",
]

logger = get_logger("api.errors")


class ServiceError(RuntimeError):
    """A problem with the model artefacts that the operator can fix."""


class ServiceNotReady(RuntimeError):
    """Raised when a request arrives before the models have finished loading."""


class Utf8JSONResponse(JSONResponse):
    """JSON with an explicit charset.

    The bodies are UTF-8 either way, but several HTTP clients fall back to
    Latin-1 when the header omits the charset, which mangles the em dashes and
    degree signs in the model's own wording.
    """

    media_type = "application/json; charset=utf-8"


def error_response(status: int, code: str, message: str, **extra) -> JSONResponse:
    """Build the API's error body."""
    return Utf8JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message, **extra}}
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the handlers that map exceptions onto that shape."""

    @app.exception_handler(ServiceNotReady)
    async def _not_ready(_: Request, exc: ServiceNotReady):
        return error_response(503, "models_loading", str(exc))

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError):
        return error_response(503, "models_unavailable", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        fields = []
        for item in exc.errors():
            location = [
                str(part) for part in item.get("loc", []) if part not in ("body", "factors")
            ]
            fields.append({"field": ".".join(location), "message": item.get("msg", "invalid value")})
        return error_response(422, "invalid_input", "Some scenario values are invalid.", fields=fields)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        logger.exception("Unhandled API error")
        return error_response(
            500, "internal_error", "The prediction engine hit an unexpected error. See the API log."
        )
