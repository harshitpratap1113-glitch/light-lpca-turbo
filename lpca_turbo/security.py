"""
Security Hardening Module for LIGHT-LPCA Turbo v2.0.
- Sliding-window per-IP rate limiter (protects server-side GROQ_API_KEY from abuse)
- Strict HTTP security headers middleware
- Secret redaction utility for safe error/log output
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


SECRET_PATTERNS = re.compile(
    r"(gsk_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{16,}|AIzaSy[A-Za-z0-9_-]{20,}|vcp_[A-Za-z0-9]{16,})"
)


def redact_secrets(text: str) -> str:
    """Masks accidental API tokens before logging or returning error strings."""
    return SECRET_PATTERNS.sub("[REDACTED_KEY]", text or "")


class SlidingWindowRateLimiter:
    """In-memory per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int = 25, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, client_ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        q = self._hits[client_ip]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True

    def remaining(self, client_ip: str) -> int:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        q = self._hits[client_ip]
        while q and q[0] < cutoff:
            q.popleft()
        return max(0, self.max_requests - len(q))


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds defensive security headers to every HTTP response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        return response
