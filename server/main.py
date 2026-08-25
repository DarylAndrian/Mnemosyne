"""Mnemosyne MCP server entry point.

FastMCP (official mcp package) with Streamable HTTP transport and a
shared-API-key auth middleware.

Run:  python -m server.main
"""

from __future__ import annotations

import contextvars
import logging
import sys

from pathlib import Path

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
        path = request.url.path

        # Public assets: health, frontend, static files.  Only /mcp and /api/* are protected.
        # This lets the Obsidian-style frontend load without a token, but API calls still require it.
        if path in ("/health", "/", "/index.html"):
            return await call_next(request)
        # Deploy webhook authenticates via HMAC signature (X-Hub-Signature-256),
        # not a Bearer token -- verified inside the handler.
        if path == "/api/deploy/webhook":
            return await call_next(request)
        # Deploy status is read-only monitoring info -- public like /health.
        if path == "/api/deploy/status":
            return await call_next(request)
        if path.startswith("/assets/") or path.startswith("/static/"):
            return await call_next(request)
        # Frontend bundle files (no auth needed to load the SPA shell)
        if path.endswith((".js", ".css", ".html", ".svg", ".png", ".ico", ".woff2")) and not path.startswith("/api/"):
            return await call_next(request)
        if path.startswith("/api/") or path.startswith("/mcp"):
            pass  # fall through to auth check
        elif not path.startswith("/api/") and not path.startswith("/mcp"):
            # Any other non-API path (e.g. "/" SPA fallback) is public
            # Keep MKS: return early for truly public paths, but we already handled static above
            # For safety, treat unknown non-API as public to not break frontend routing
            if not path.startswith("/api"):
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
async def remember(
    content: str,
    agent_id: str = "",
    session_id: str = "",
    tags: list[str] | None = None,
    request_id: str = "",
) -> dict:
    """Store a memory. Returns immediately with episode_id and
    processing=true; entity/relationship/fact extraction and embedding
    run in the background (10-20s on CPU-only hosts). Check completion
    with episode_status if needed.

    Idempotent: pass a unique request_id per logical store intent (e.g.
    a UUID) and retries after a client timeout return the original
    episode instead of creating duplicates. Identical or near-identical
    content within 5 minutes is also deduplicated automatically.

    Args:
        content: The raw memory content (conversation excerpt, note, event).
        agent_id: Identifier of the agent storing this memory. Auto-set from token if not provided.
        session_id: Optional session identifier for provenance.
        tags: Optional list of tags.
        request_id: Optional client-generated idempotency key (UUID recommended).

    Returns:
        episode_id, processing flag, and status. If deduplicated=true,
        the episode already existed and nothing new was stored.
    """
    # Auto-set agent_id from per-agent token if not explicitly provided
    if not agent_id:
        agent_id = current_agent_id.get() or "unknown"
    return (await engine.remember(content, agent_id, session_id, tags, request_id)).model_dump()


@mcp.tool()
async def episode_status(episode_id: str) -> dict:
    """Check the background processing status of an episode stored via
    remember. Use after remember returns processing=true if you need to
    know when extraction/embedding finished.

    Args:
        episode_id: Episode id returned by remember (starts with 'ep_').

    Returns:
        Status: pending (still processing), processed (complete),
        failed (extraction error), or not_found.
    """
    return await engine.episode_status(episode_id)


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
# REST API for the Obsidian-style frontend
# ---------------------------------------------------------------------------

async def _mount_api(app):
    from starlette.requests import Request
    from starlette.routing import Route

    def _cors(resp: JSONResponse) -> JSONResponse:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    async def _options(request: Request):
        return _cors(JSONResponse({}, status_code=204))

    async def api_stats(request: Request):
        h = await engine.health()
        # Return same stats shape as /health but wrapped for frontend
        return _cors(JSONResponse(h))

    async def api_graph(request: Request):
        """Return nodes/edges for vis-network.
        Query params: limit (default 400), include_episodes (bool), type filter.
        """
        try:
            limit = int(request.query_params.get("limit", "400"))
        except ValueError:
            limit = 400
        limit = max(10, min(limit, 2000))
        q_type = request.query_params.get("type", "").strip()
        include_episodes = request.query_params.get("include_episodes", "false").lower() in ("1", "true", "yes")

        nodes: list[dict] = []
        edges: list[dict] = []

        if not await engine.graph.ping():
            return _cors(JSONResponse({"nodes": [], "edges": [], "error": "neo4j unreachable"}, status_code=503))

        # Entities
        try:
            if q_type:
                rows = await engine.graph.run(
                    "MATCH (e:Entity {type: $t}) RETURN e.id AS id, e.name AS name, e.type AS type, "
                    "e.description AS description, e.aliases AS aliases, coalesce(e.access_count,0) AS ac LIMIT $limit",
                    t=q_type, limit=limit,
                )
            else:
                rows = await engine.graph.run(
                    "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type, "
                    "e.description AS description, e.aliases AS aliases, coalesce(e.access_count,0) AS ac LIMIT $limit",
                    limit=limit,
                )
        except Exception as exc:
            return _cors(JSONResponse({"nodes": [], "edges": [], "error": str(exc)}, status_code=500))

        id_to_name: dict[str, str] = {}
        for r in rows:
            nid = r.get("id")
            name = r.get("name") or nid
            id_to_name[nid] = name
            nodes.append({
                "id": nid,
                "label": name,
                "type": r.get("type") or "Concept",
                "description": r.get("description") or "",
                "aliases": r.get("aliases") or [],
                "access_count": r.get("ac") or 0,
                "group": r.get("type") or "Concept",
            })

        # Relationships between fetched entities + optionally to outside entities if limit not hit
        if id_to_name:
            try:
                # Only edges where both ends are in the fetched set for clean initial view
                edge_rows = await engine.graph.run(
                    "MATCH (a:Entity)-[r]->(b:Entity) "
                    "WHERE a.id IN $ids AND b.id IN $ids "
                    "RETURN a.id AS from_id, b.id AS to_id, type(r) AS rel_type LIMIT $limit",
                    ids=list(id_to_name.keys()), limit=limit * 2,
                )
                # Fallback: if sparse, allow edges to outside to show connectivity (capped)
                if len(edge_rows) < 5 and len(id_to_name) < limit:
                    extra = await engine.graph.run(
                        "MATCH (a:Entity)-[r]->(b:Entity) "
                        "WHERE a.id IN $ids "
                        "RETURN a.id AS from_id, b.id AS to_id, type(r) AS rel_type, "
                        "b.name AS to_name, b.type AS to_type, b.id AS to_id2 LIMIT $limit",
                        ids=list(id_to_name.keys()), limit=limit,
                    )
                    # add missing outside nodes
                    for er in extra:
                        tid = er.get("to_id")
                        if tid not in id_to_name:
                            nodes.append({
                                "id": tid,
                                "label": er.get("to_name") or tid,
                                "type": er.get("to_type") or "Concept",
                                "description": "",
                                "aliases": [],
                                "access_count": 0,
                                "group": er.get("to_type") or "Concept",
                            })
                            id_to_name[tid] = er.get("to_name") or tid
                    edge_rows = extra
            except Exception:
                edge_rows = []

            seen = set()
            for er in edge_rows:
                fid, tid, typ = er.get("from_id"), er.get("to_id"), er.get("rel_type") or "RELATED_TO"
                key = (fid, tid, typ)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"id": f"{fid}->{typ}->{tid}", "from": fid, "to": tid, "label": typ})

        # Optionally add Episode nodes (lightweight)
        if include_episodes:
            try:
                ep_rows = await engine.graph.run(
                    "MATCH (ep:Episode) WHERE ep.active = true "
                    "RETURN ep.id AS id, ep.summary AS summary, ep.content AS content, ep.tags AS tags "
                    "ORDER BY ep.created_at DESC LIMIT $limit",
                    limit=min(limit, 100),
                )
                for er in ep_rows:
                    eid = er.get("id")
                    nodes.append({
                        "id": eid,
                        "label": (er.get("summary") or er.get("content") or eid)[:40],
                        "type": "Episode",
                        "description": (er.get("content") or "")[:300],
                        "aliases": er.get("tags") or [],
                        "access_count": 0,
                        "group": "Episode",
                    })
                # Episode-entity links
                if ep_rows:
                    link_rows = await engine.graph.run(
                        "MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode) "
                        "WHERE e.id IN $ids RETURN e.id AS from_id, ep.id AS to_id LIMIT $limit",
                        ids=list(id_to_name.keys()), limit=limit,
                    )
                    for lr in link_rows:
                        edges.append({"id": f"{lr['from_id']}->MENTIONED_IN->{lr['to_id']}", "from": lr["from_id"], "to": lr["to_id"], "label": "MENTIONED_IN"})
            except Exception:
                pass

        return _cors(JSONResponse({"nodes": nodes, "edges": edges, "counts": {"nodes": len(nodes), "edges": len(edges)}}))

    async def api_entity(request: Request):
        name = request.query_params.get("name", "").strip()
        if not name:
            return _cors(JSONResponse({"error": "missing ?name="}, status_code=400))
        # reuse engine.context for neighborhood
        data = await engine.context(name, depth_limit=100)
        # Also fetch entity details + facts
        details = {}
        try:
            ent = await engine.graph.run_one(
                "MATCH (e:Entity) WHERE e.name = $name OR $name IN e.aliases RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description, e.aliases AS aliases, coalesce(e.access_count,0) AS ac",
                name=name,
            )
            if ent:
                details = ent
                # facts
                facts = await engine.graph.run(
                    "MATCH (f:Fact) WHERE f.subject = $sid AND f.valid_until IS NULL RETURN f.id AS id, f.content AS content, f.confidence AS confidence LIMIT 20",
                    sid=ent["id"],
                )
                details["facts"] = facts
                # episodes mentioning it
                eps = await engine.graph.run(
                    "MATCH (e:Entity {id: $sid})-[:MENTIONED_IN]->(ep:Episode) RETURN ep.id AS id, ep.summary AS summary, ep.content AS content LIMIT 10",
                    sid=ent["id"],
                )
                details["episodes"] = eps
        except Exception:
            pass
        return _cors(JSONResponse({"context": data, "entity": details}))

    async def api_recall(request: Request):
        q = request.query_params.get("q") or request.query_params.get("query") or ""
        if not q:
            return _cors(JSONResponse({"error": "missing ?q="}, status_code=400))
        try:
            top_k = int(request.query_params.get("top_k", "5"))
        except ValueError:
            top_k = 5
        data = await engine.recall(q, top_k=top_k)
        return _cors(JSONResponse(data))

    async def api_episodes(request: Request):
        try:
            limit = int(request.query_params.get("limit", "20"))
        except ValueError:
            limit = 20
        limit = max(1, min(limit, 100))
        if not await engine.graph.ping():
            return _cors(JSONResponse({"episodes": [], "error": "neo4j unreachable"}, status_code=503))
        rows = await engine.graph.run(
            "MATCH (ep:Episode) RETURN ep.id AS id, ep.content AS content, ep.summary AS summary, ep.tags AS tags, ep.source_agent_id AS source_agent_id, toString(ep.created_at) AS created_at, ep.active AS active ORDER BY ep.created_at DESC LIMIT $limit",
            limit=limit,
        )
        return _cors(JSONResponse({"episodes": rows}))

    # Register routes - insert at front so they take precedence
    routes = [
        Route("/api/stats", api_stats, methods=["GET", "OPTIONS"]),
        Route("/api/graph", api_graph, methods=["GET", "OPTIONS"]),
        Route("/api/entity", api_entity, methods=["GET", "OPTIONS"]),
        Route("/api/recall", api_recall, methods=["GET", "OPTIONS"]),
        Route("/api/episodes", api_episodes, methods=["GET", "OPTIONS"]),
        Route("/api/graph", _options, methods=["OPTIONS"]),
        Route("/api/stats", _options, methods=["OPTIONS"]),
        Route("/api/entity", _options, methods=["OPTIONS"]),
        Route("/api/recall", _options, methods=["OPTIONS"]),
        Route("/api/episodes", _options, methods=["OPTIONS"]),
    ]
    for r in reversed(routes):
        app.routes.insert(0, r)


async def _mount_frontend(app):
    """Serve the Obsidian-style SPA from /frontend. Falls back to index.html."""
    from starlette.responses import FileResponse, HTMLResponse
    from starlette.routing import Route

    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if not frontend_dir.exists():
        return

    async def serve_index(request):
        idx = frontend_dir / "index.html"
        if idx.exists():
            return FileResponse(str(idx), media_type="text/html")
        return HTMLResponse("<h1>Frontend not built</h1><p>Missing frontend/index.html</p>", status_code=404)

    async def serve_file(request):
        # Starlette will handle /static via this catch-all
        fname = request.path_params.get("path", "")
        fpath = frontend_dir / fname
        # Security: prevent path traversal
        try:
            fpath = fpath.resolve()
            if not str(fpath).startswith(str(frontend_dir.resolve())):
                return JSONResponse({"error": "forbidden"}, status_code=403)
        except Exception:
            return JSONResponse({"error": "not found"}, status_code=404)
        if fpath.exists() and fpath.is_file():
            return FileResponse(str(fpath))
        # SPA fallback
        idx = frontend_dir / "index.html"
        if idx.exists():
            return FileResponse(str(idx), media_type="text/html")
        return JSONResponse({"error": "not found"}, status_code=404)

    # Only mount if not already mounted
    app.routes.insert(0, Route("/", serve_index, methods=["GET"]))
    app.routes.insert(0, Route("/index.html", serve_index, methods=["GET"]))
    # Catch-all for frontend assets - must be after /api and /mcp but before 404
    # Use a path param route
    from starlette.routing import Route as R
    app.routes.append(R("/{path:path}", serve_file, methods=["GET"]))


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
        await _mount_api(app)
        from .deploy import mount_deploy

        await mount_deploy(app)
        await _mount_frontend(app)

        log.info(
            "Mnemosyne MCP serving on http://%s:%s/mcp (health: /health, graph: /)",
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
