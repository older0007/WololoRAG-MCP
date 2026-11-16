"""
Optional Authentication Middleware for the MCP HTTP Microservice (src/mcp_server/auth.py).

- When `mcp_server.auth.enabled` is `false` (default): all requests pass through.
- When `mcp_server.auth.enabled` is `true`: requires valid `X-MCP-Key` or
  `Authorization: Bearer <secret_token>`, except on `/health`.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import McpAuthConfig


class OptionalTokenAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing token auth only when `auth.enabled` is True."""

    def __init__(self, app, auth_config: McpAuthConfig) -> None:
        super().__init__(app)
        self.auth_config = auth_config

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.auth_config.enabled:
            return await call_next(request)

        if request.url.path == "/health":
            return await call_next(request)

        expected = self.auth_config.secret_token
        provided_header = request.headers.get(self.auth_config.api_key_header, "").strip()

        auth_header = request.headers.get("Authorization", "").strip()
        bearer_token = ""
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header[7:].strip()

        candidate = provided_header or bearer_token
        if not candidate or not hmac.compare_digest(candidate, expected):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": (
                        f"Provide a valid token via '{self.auth_config.api_key_header}' "
                        "or 'Authorization: Bearer <token>'."
                    ),
                },
            )

        return await call_next(request)
