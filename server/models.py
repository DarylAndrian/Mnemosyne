"""Pydantic models for the memory pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EntityOut(BaseModel):
    """Entity as extracted by the LLM."""
    name: str
    type: str = "Concept"
    description: str = ""
    aliases: list[str] = Field(default_factory=list)


class RelationshipOut(BaseModel):
    """Relationship between two entities as extracted by the LLM."""
    from_name: str
    to_name: str
    type: str = "RELATED_TO"
    properties: dict = Field(default_factory=dict)


class FactOut(BaseModel):
    """Standalone factual claim as extracted by the LLM."""
    subject_name: str
    content: str
    confidence: float = 0.8


class ExtractionResult(BaseModel):
    """Full structured extraction from one episode."""
    entities: list[EntityOut] = Field(default_factory=list)
    relationships: list[RelationshipOut] = Field(default_factory=list)
    facts: list[FactOut] = Field(default_factory=list)


class ConflictInfo(BaseModel):
    entity: str
    old_fact_id: str
    old_content: str
    new_content: str
    action: str = "superseded"
    reason: str = ""


class RecallHit(BaseModel):
    content: str
    score: float
    kind: str  # episode | fact
    source_agent_id: Optional[str] = None
    source_session_id: Optional[str] = None
    # Who wrote the memory (same value as source_agent_id). Always present.
    source: Optional[str] = None
    entities: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    still_valid: bool = True
    # Episode id for forget(). Null on fact hits, which are not episodes.
    episode_id: Optional[str] = None


class RememberResponse(BaseModel):
    episode_id: str
    entities_created: list[str] = Field(default_factory=list)
    entities_merged: list[str] = Field(default_factory=list)
    relationships_created: int = 0
    facts_stored: int = 0
    conflicts_detected: list[ConflictInfo] = Field(default_factory=list)
    embedding_status: str = "ok"
    # True while LLM extraction + embedding run in the background
    processing: bool = False
    # True when an existing episode was returned instead of creating a new one
    deduplicated: bool = False
    # Episode processing status: pending | processed | failed | rejected
    status: str = "processed"
    # Set when an existing keyed episode was replaced in place.
    updated: bool = False
    key: str = ""
    # Set when remember refuses to store the text.
    rejected: bool = False
    error: str = ""
