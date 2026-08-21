"""Settings loaded from environment variables (.env file)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency)."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


@dataclass
class Settings:
    neo4j_uri: str = field(default_factory=lambda: os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.environ.get("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "neo4j"))

    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    extract_model: str = field(default_factory=lambda: os.environ.get("EXTRACT_MODEL", "qwen2.5:3b"))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "nomic-embed-text"))

    mcp_host: str = field(default_factory=lambda: os.environ.get("MCP_HOST", "0.0.0.0"))
    mcp_port: int = field(default_factory=lambda: int(os.environ.get("MCP_PORT", "8080")))
    mcp_api_key: str = field(default_factory=lambda: os.environ.get("MCP_API_KEY", ""))
    allowed_agents: list = field(default_factory=lambda: [
        a.strip() for a in os.environ.get("ALLOWED_AGENTS", "").split(",") if a.strip()
    ])

    embed_dim: int = 768  # nomic-embed-text output dimension

    def validate(self) -> list[str]:
        problems = []
        if not self.mcp_api_key:
            problems.append("MCP_API_KEY is empty -- generate one: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        if self.neo4j_password in ("neo4j", "change-me-a-strong-password"):
            problems.append("NEO4J_PASSWORD is still the default -- set a real password in .env")
        return problems


def get_settings() -> Settings:
    return Settings()


def new_api_key() -> str:
    return secrets.token_urlsafe(32)
