"""Entity grounding, generic-name filtering, and deterministic extraction.

Extraction is limited to the episode's own text. Names that do not appear in
that text, generic labels, and entities already attached to more than 30% of
episodes are not auto-attached.
"""

from __future__ import annotations

import re
from collections import Counter

from .models import EntityOut

COMMON_ENTITY_FRACTION = 0.30

GENERIC_ENTITY_NAMES = {
    "ai",
    "agent",
    "assistant",
    "bot",
    "chat",
    "concept",
    "content",
    "context",
    "conversation",
    "data",
    "episode",
    "fact",
    "hermes",
    "human",
    "information",
    "item",
    "memory",
    "message",
    "mnemosyne",
    "n/a",
    "none",
    "note",
    "other",
    "person",
    "project",
    "session",
    "summary",
    "system",
    "task",
    "thing",
    "today",
    "unknown",
    "update",
    "user",
}

# Explicit "Kind: Name" labels, plus a few proper-name / id shapes.
_LABEL_RE = re.compile(
    r"(?im)\b(Person|Product|Board|Project|Identifier|Topic)[ \t]*:[ \t]*"
    r"([A-Za-z0-9][A-Za-z0-9_+-]{0,40}(?:[ \t]+[A-Za-z0-9][A-Za-z0-9_+-]{0,40}){0,3})"
)
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]+\b")
_ALLCAPS_PHRASE_RE = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z][A-Za-z0-9]+){1,4}\b")
_EP_ID_RE = re.compile(r"\bep_[0-9a-fA-F]{8,}\b")
_LONG_ID_RE = re.compile(r"\b\d{8,}\b")

REPAIR_TARGETS: tuple[dict[str, str], ...] = (
    {"name": "HOKI Rewards", "type": "Product"},
    {"name": "VelaTavern", "type": "Project"},
)

_ALLOWED_TYPES = {
    "Person",
    "Project",
    "Service",
    "API",
    "Tool",
    "Decision",
    "Concept",
    "Task",
    "Product",
    "Board",
    "Identifier",
    "Topic",
}


def compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def is_generic(name: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (name or "").strip().lower())
    if not cleaned or cleaned in GENERIC_ENTITY_NAMES:
        return True
    if len(cleaned) < 3 and not any(ch.isdigit() for ch in cleaned):
        return True
    return False


def grounded_in_text(name: str, content: str) -> bool:
    """True when the entity name actually occurs in this episode."""
    if not name or not content:
        return False
    return name.strip().lower() in content.lower()


def too_common(linked: int, total: int, threshold: float = COMMON_ENTITY_FRACTION) -> bool:
    """True when an entity is already on more than `threshold` of episodes."""
    if total <= 0 or linked <= 0:
        return False
    return (linked / total) > threshold


def acceptable_alias(name: str, alias: str) -> bool:
    """Reject aliases that would collapse a shorter distinct name into a longer one."""
    if is_generic(alias):
        return False
    if compact_name(alias) == compact_name(name):
        return False
    short, long = sorted((compact_name(alias), compact_name(name)), key=len)
    if short and long.startswith(short) and long != short:
        return False
    return True


def is_absorbed_name(long_name: str, short_name: str) -> bool:
    long_c, short_c = compact_name(long_name), compact_name(short_name)
    return bool(short_c) and long_c != short_c and long_c.startswith(short_c)


def mentions_standalone(content: str, short: str, blockers: list[str]) -> bool:
    """True when `short` appears and is not only a prefix of a longer known name."""
    if not content or not short:
        return False
    hay = content.lower()
    needle = short.lower()
    blocked_by = [b.lower() for b in blockers if b and b.lower() != needle]
    start = 0
    while True:
        index = hay.find(needle, start)
        if index < 0:
            return False
        if not any(hay.startswith(blocker, index) for blocker in blocked_by):
            return True
        start = index + max(len(needle), 1)


def _clean_label(value: str) -> str:
    return re.sub(r"[\s,;:]+$", "", value).strip()[:200]


def mine_entities(content: str) -> list[EntityOut]:
    """Pull people, products, boards, projects, and ids from this text only."""
    if not content:
        return []
    found: list[EntityOut] = []
    seen: set[str] = set()

    def add(name: str, type_name: str) -> None:
        cleaned = _clean_label(name)
        if not cleaned or not grounded_in_text(cleaned, content) or is_generic(cleaned):
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        type_name = type_name if type_name in _ALLOWED_TYPES else "Concept"
        found.append(EntityOut(name=cleaned, type=type_name, description="", aliases=[]))

    for match in _LABEL_RE.finditer(content):
        add(match.group(2), match.group(1).capitalize())
    for match in _CAMEL_RE.finditer(content):
        add(match.group(0), "Product")
    for match in _ALLCAPS_PHRASE_RE.finditer(content):
        add(match.group(0), "Product")
    for match in _EP_ID_RE.finditer(content):
        add(match.group(0), "Identifier")
    for match in _LONG_ID_RE.finditer(content):
        add(match.group(0), "Identifier")
    return found


def assign_entities(episodes: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Choose per-episode entity names, dropping generic and >30% entities.

    `episodes` is a list of (episode_id, text). The result is specific names
    only -- no episode text is returned.
    """
    mined: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    for episode_id, text in episodes:
        names = []
        seen: set[str] = set()
        for entity in mine_entities(text):
            key = entity.name.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(entity.name)
        mined[episode_id] = names
        counts.update(names)
    total = len(episodes)
    assigned: dict[str, list[str]] = {}
    for episode_id, names in mined.items():
        assigned[episode_id] = [
            name for name in names if not too_common(counts[name], total)
        ]
    return assigned
