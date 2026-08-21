"""Mnemosyne MCP server entry point.

FastMCP (official mcp package) with Streamable HTTP transport and a
shared-API-key auth middleware.

Run:  python -m server.main
"""

from __future__ import annotations

import contextvars
import logging
import sys

from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import get_settings
from .engine import MemoryEngine
from .graph.schema import ensure_schema

# Context variable for per-agent token attribution
# Set by auth middleware, read by tool handlers
current_agent_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_agent_id", default=""
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("mnemosyne")

settings = get_settings()
engine = MemoryEngine(settings)

mcp = FastMCP(
    "mnemosyne",
    host=settings.mcp_host,
    port=settings.mcp_port,
    streamable_http_path="/mcp",
)


# ---------------------------------------------------------------------------
# Auth middleware: shared API key OR per-agent token via Authorization: Bearer
# ---------------------------------------------------------------------------

class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Auth middleware that supports:
    1. The original shared MCP_API_KEY (legacy, any agent_id)
    2. Per-agent tokens from agent-tokens.json (auto-sets agent_id)
    """

    def __init__(self, app):
        super().__init__(app)
        self._agent_tokens: dict = {}

    def _reload_tokens(self) -> None:
        from .agent_tokens import load_tokens
        self._agent_tokens = load_tokens()

    async def dispatch(self, request, call_next):
        # Health endpoint stays open for monitoring
        if request.url.path in ("/health", "/"):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""

        if not token:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        # 1. Check shared API key (legacy -- any agent_id)
        if settings.mcp_api_key and token == settings.mcp_api_key:
            return await call_next(request)

        # 2. Check per-agent token (auto-sets agent_id via context var)
        self._reload_tokens()
        agent_id = self._agent_tokens.get(token, {}).get("agent_id")
        if agent_id:
            current_agent_id.set(agent_id)
            return await call_next(request)

        return JSONResponse({"error": "unauthorized"}, status_code=401)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def remember(content: str, agent_id: str = "", session_id: str = "", tags: list[str] | None = None) -> dict:
    """Store a memory. The server extracts entities/relationships/facts into
    the knowledge graph, detects conflicts with existing facts, and embeds
    the episode for semantic search.

    Args:
        content: The raw memory content (conversation excerpt, note, event).
        agent_id: Identifier of the agent storing this memory. Auto-set from token if not provided.
        session_id: Optional session identifier for provenance.
        tags: Optional list of tags.

    Returns:
        Episode id, created/merged entities, relationship count, and any
        conflicts detected. Report conflicts to the user.
    """
    # Auto-set agent_id from per-agent token if not explicitly provided
    if not agent_id:
        agent_id = current_agent_id.get() or "unknown"
    return (await engine.remember(content, agent_id, session_id, tags)).model_dump()


@mcp.tool()
async def recall(query: str, top_k: int = 5, agent_id: str = "") -> dict:
    """Hybrid RAG search across the shared memory: vector similarity +
    keyword matching + graph neighborhood expansion, fused with recency
    and access boosts. Use before answering questions that may need
    cross-agent context.

    Args:
        query: Natural language question or search phrase.
        top_k: Max results to return (1-20, default 5).
        agent_id: Optional agent identifier (for future filtering).

    Returns:
        Ranked results with provenance, plus graph context (entities and paths).
    """
    return await engine.recall(query, top_k=top_k, agent_id=agent_id)


@mcp.tool()
async def context(entity_name: str, depth_limit: int = 100) -> dict:
    """Get the graph neighborhood around an entity: everything connected
    to it within 1-3 hops. Use when exploring a topic's relationships.

    Args:
        entity_name: Entity name (fuzzy-matched if exact name not found).
        depth_limit: Max edges to return (default 100).

    Returns:
        The entity and all its edges (from -[TYPE]-> to).
    """
    return await engine.context(entity_name, depth_limit)


@mcp.tool()
async def resolve(entity_a: str, entity_b: str) -> dict:
    """Merge two duplicate entities. Keeps entity_a as primary, moves all
    of entity_b's edges and facts to it, adds entity_b's name as an alias,
    then deletes entity_b.

    Args:
        entity_a: Name of the entity to keep.
        entity_b: Name of the entity to merge in and delete.

    Returns:
        Merge summary with new alias list and reconnected edge count.
    """
    return await engine.resolve(entity_a, entity_b)


@mcp.tool()
async def forget(memory_id: str) -> dict:
    """Soft-delete an episode. Marks it inactive and invalidates its facts,
    but preserves everything for provenance.

    Args:
        memory_id: Episode id to forget (starts with 'ep_').

    Returns:
        Confirmation or error if the episode was not found.
    """
    return await engine.forget(memory_id)


# ---------------------------------------------------------------------------
# Plain-HTTP health endpoint (not part of MCP)
# ---------------------------------------------------------------------------

async def _mount_health(app):
    from starlette.routing import Route

    async def health(request):
        h = await engine.health()
        code = 200 if h["ok"] else 503
        return JSONResponse(h, status_code=code)

    app.routes.insert(0, Route("/health", health, methods=["GET"]))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

async def _startup_checks() -> None:
    """Verify backends and create schema before serving."""
    problems = settings.validate()
    for p in problems:
        log.warning("CONFIG: %s", p)

    if not await engine.graph.ping():
        log.error("Neo4j unreachable at %s -- is docker compose up?", settings.neo4j_uri)
    else:
        summary = await ensure_schema(engine.graph)
        log.info("Neo4j schema ensured: %s", summary)

    if not await engine.embedder.health():
        log.error("Ollama unreachable at %s -- is ollama running?", settings.ollama_url)

    models_needed = {settings.embed_model, settings.extract_model}
    log.info("Required Ollama models: %s", ", ".join(sorted(models_needed)))


def main() -> None:
    problems = settings.validate()
    if problems:
        for p in problems:
            print(f"[config warning] {p}", file=sys.stderr)

    import asyncio
    import uvicorn

    async def amain() -> None:
        # Everything in ONE event loop: startup checks, clients, and serving.
        await _startup_checks()

        app = mcp.streamable_http_app()
        app.add_middleware(ApiKeyMiddleware)
        await _mount_health(app)

        log.info(
            "Mnemosyne MCP serving on http://%s:%s/mcp (health: /health)",
            settings.mcp_host, settings.mcp_port,
        )
        config = uvicorn.Config(app, host=settings.mcp_host, port=settings.mcp_port, log_level="info")
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            await engine.close()

    asyncio.run(amain())


if __name__ == "__main__":
    main()
