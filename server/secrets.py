"""Reject likely secrets before they are stored. Never echo the match."""

from __future__ import annotations

import re

# Patterns are intentionally specific: a bare word like "password" or "token"
# in normal prose does not match. A value has to be present.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(password|passwd|pwd)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(password|passwd|pwd)\s+is\s+\S+"),
    re.compile(
        r"(?i)\b(api[_-]?key|apikey|secret|access[_-]?token|auth[_-]?token|token)\b\s*[:=]\s*\S+"
    ),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]{20,}={0,2}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

REJECTION_MESSAGE = (
    "rejected: content looks like it contains a secret "
    "(password, token, API key, or JWT). Remove it and try again."
)


def find_secret(text: str) -> bool:
    """Return True when text looks like it contains a credential."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def redact_secrets(text: str) -> str:
    """Replace secret-shaped substrings. Used for previews, never for logging raw text."""
    redacted = text or ""
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted
