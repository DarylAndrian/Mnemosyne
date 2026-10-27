"""Small text helpers that must not log episode bodies or secrets."""

from __future__ import annotations

import base64
import json
import re

from .secrets import redact_secrets


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token)."""
    if not text:
        return 0
    return (len(text) + 3) // 4


def safe_error(exc: BaseException, content: str = "") -> str:
    """Exception summary safe to store. Episode text and secrets are removed."""
    text = f"{type(exc).__name__}: {exc}"
    if content and len(content) >= 12:
        text = text.replace(content, "[redacted]")
    text = redact_secrets(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 300:
        text = text[:300]
    return text or type(exc).__name__


def snippet_around(content: str, query: str, window: int = 200) -> str:
    """Excerpt of about `window` characters around the best query match."""
    if not content:
        return ""
    terms = [term for term in re.findall(r"\w+", query.lower()) if len(term) > 2]
    hay = content.lower()
    best = -1
    best_len = 0
    for term in terms:
        index = hay.find(term)
        if index != -1 and (best == -1 or index < best):
            best = index
            best_len = len(term)
    if best < 0:
        excerpt = content[:window]
    else:
        half = max(window // 2, 1)
        start = max(0, best - half)
        excerpt = content[start : start + window]
        if start > 0:
            excerpt = "..." + excerpt[3:] if len(excerpt) > 3 else excerpt
        # Keep the match inside the window even after the ellipsis.
        if best_len and excerpt.lower().find(content[best : best + best_len].lower()) < 0:
            excerpt = content[best : best + window]
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if len(excerpt) > window:
        excerpt = excerpt[:window].rstrip()
    return excerpt


def encode_cursor(created_at: str, episode_id: str) -> str:
    raw = json.dumps({"t": created_at, "id": episode_id}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, str]:
    if not cursor:
        return "", ""
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid cursor") from exc
    if not isinstance(data, dict) or "t" not in data or "id" not in data:
        raise ValueError("invalid cursor")
    return str(data["t"]), str(data["id"])
