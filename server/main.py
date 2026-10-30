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
    key: str = "",
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
        key: Optional stable key. When an active episode already has this key,
            its content is replaced and the previous body is kept as history.

    Returns:
        episode_id, processing flag, and status. If deduplicated=true,
        the episode already existed and nothing new was stored.
        Passwords, tokens, API keys, and JWTs are rejected and not stored.
    """
    # Auto-set agent_id from per-agent token if not explicitly provided
    if not agent_id:
        agent_id = current_agent_id.get() or "unknown"
    return (await engine.remember(content, agent_id, session_id, tags, request_id, key)).model_dump()


@mcp.tool()
async def episode_status(episode_id: str) -> dict:
    """Check the background processing status of an episode stored via
    remember. Use after remember returns processing=true if you need to
    know when extraction/embedding finished.

    Args:
        episode_id: Episode id returned by remember (starts with 'ep_').

    Returns:
        status (pending, processed, failed, or not_found), error_message,
        attempts, and last_attempt_at.
    """
    return await engine.episode_status(episode_id)


@mcp.tool()
async def recall(query: str, top_k: int = 5, agent_id: str = "", mode: str = "full") -> dict:
    """Hybrid RAG search across the shared memory: vector similarity +
    keyword matching + graph neighborhood expansion, fused with recency
    and access boosts. Use before answering questions that may need
    cross-agent context.

    Args:
        query: Natural language question or search phrase.
        top_k: Max results to return (1-20, default 5).
        agent_id: Optional agent identifier (for future filtering).
        mode: "full" (default) or "snippet". Snippet returns id, score, and
            a short excerpt around the match.

    Returns:
        Ranked results with episode_id, created_at, source, and score.
        Full mode also includes content and graph context.
    """
    return await engine.recall(query, top_k=top_k, agent_id=agent_id, mode=mode)


@mcp.tool()
async def context(entity_name: str = "", depth_limit: int = 100) -> dict:
    """Get the graph neighborhood around an entity: everything connected
    to it within 1-3 hops. Use when exploring a topic's relationships.

    Args:
        entity_name: Entity name (fuzzy-matched if exact name not found).
            Omit it to load the INDEX episode and TOPIC CARD titles.
        depth_limit: Max edges to return (default 100).

    Returns:
        The entity and all its edges, or the INDEX payload when no name is given.
    """
    return await engine.context(entity_name, depth_limit)


@mcp.tool()
async def resolve(
    entity_a: str,
    entity_b: str,
    fuzzy: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Merge two duplicate entities. Keeps entity_a as primary, moves all
    of entity_b's edges and facts to it, adds entity_b's name as an alias,
    then deletes entity_b.

    Args:
        entity_a: Name of the entity to keep. Exact name match unless fuzzy is set.
        entity_b: Name of the entity to merge in and delete.
        fuzzy: When true, allow a high-confidence fuzzy name match. Default is exact.
        dry_run: When true, report what would merge and change nothing.
        force: When true, merge even if the two names look unrelated. By default
            a merge is refused (and dry_run reports would_merge=false) when the
            names share no tokens and have low similarity, since entity_b is
            deleted. Set fuzzy or force to override.

    Returns:
        Merge summary with merge_id (for unresolve), alias list, and edge count.
    """
    return await engine.resolve(
        entity_a, entity_b, fuzzy=fuzzy, dry_run=dry_run, force=force
    )


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


@mcp.tool()
async def list_episodes(
    text_contains: str = "",
    source: str = "",
    created_before: str = "",
    created_after: str = "",
    entity: str = "",
    status: str = "",
    limit: int = 50,
    cursor: str = "",
) -> dict:
    """List active episodes. Every filter is optional.

    Args:
        text_contains: Case-insensitive substring of the episode text.
        source: Agent or device id that wrote the episode.
        created_before: ISO timestamp, inclusive upper bound.
        created_after: ISO timestamp, inclusive lower bound.
        entity: Exact entity name linked to the episode.
        status: pending, processed, or failed.
        limit: Page size, default 50, max 200.
        cursor: Pass next_cursor from the previous page.

    Returns:
        id, created_at, source, status, and the first 160 characters.
    """
    return await engine.list_episodes(
        text_contains=text_contains,
        source=source,
        created_before=created_before,
        created_after=created_after,
        entity=entity,
        status=status,
        limit=limit,
        cursor=cursor,
    )


@mcp.tool()
async def list_orphans(
    include_single_edge: bool = False,
    limit: int = 5000,
    include_inactive: bool = False,
) -> dict:
    """List graph nodes that have no edges.

    Args:
        include_single_edge: When true, also include nodes with exactly one edge.
        limit: Max rows to return (the count field is the full total).
        include_inactive: When true, also include forgotten (active=false)
            episodes and entities backed only by them. Default false, so
            soft-deleted material does not show as orphans.

    Returns:
        count plus id, name, type, source_episode_id, and degree for each node.
    """
    return await engine.list_orphans(
        include_single_edge=include_single_edge,
        limit=limit,
        include_inactive=include_inactive,
    )


@mcp.tool()
async def reprocess_failed(episode_ids: list[str] | None = None) -> dict:
    """Re-queue every failed episode, or the given ids, for extraction.

    Safe to run more than once. Does not delete anything. Returns ids only.
    """
    return await engine.reprocess_failed(episode_ids=episode_ids or None, wait=False)


@mcp.tool()
async def unresolve(merge_id: str) -> dict:
    """Undo a resolve() merge. Restores both entities and their links.

    Args:
        merge_id: Id returned by resolve (starts with 'merge_').
    """
    return await engine.unresolve(merge_id)


@mcp.tool()
async def get_by_key(key: str) -> dict:
    """Fetch the current episode for a stable key, plus older versions.

    Args:
        key: Key passed to remember, for example "topic-card:slingshot".
    """
    return await engine.get_by_key(key)


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
        include_inactive = request.query_params.get("include_inactive", "false").lower() in ("1", "true", "yes")

        nodes: list[dict] = []
        edges: list[dict] = []

        if not await engine.graph.ping():
            return _cors(JSONResponse({"nodes": [], "edges": [], "error": "neo4j unreachable"}, status_code=503))

        # Skip entities that are only backed by forgotten (active=false)
        # episodes: soft-deleted material should not appear in the graph.
        # Entities with no episode at all are kept - they are real orphans.
        entity_visibility = (
            "$include_inactive = true "
            "OR NOT (e)-[:MENTIONED_IN]->(:Episode) "
            "OR EXISTS { MATCH (e)-[:MENTIONED_IN]->(ep:Episode) "
            "            WHERE coalesce(ep.active, true) = true }"
        )

        # Entities
        try:
            if q_type:
                rows = await engine.graph.run(
                    "MATCH (e:Entity {type: $t}) "
                    f"WHERE {entity_visibility} "
                    "RETURN e.id AS id, e.name AS name, e.type AS type, "
                    "e.description AS description, e.aliases AS aliases, coalesce(e.access_count,0) AS ac LIMIT $limit",
                    t=q_type, limit=limit, include_inactive=include_inactive,
                )
            else:
                rows = await engine.graph.run(
                    "MATCH (e:Entity) "
                    f"WHERE {entity_visibility} "
                    "RETURN e.id AS id, e.name AS name, e.type AS type, "
                    "e.description AS description, e.aliases AS aliases, coalesce(e.access_count,0) AS ac LIMIT $limit",
                    limit=limit, include_inactive=include_inactive,
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
                target = er.get("target") or tid
                if not fid or not target:
                    continue
                key = (fid, target, typ)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({
                    "id": f"{fid}->{typ}->{target}",
                    "from": fid,
                    "to": target,
                    "source": fid,
                    "target": target,
                    "label": typ,
                })

        # Optionally add Episode nodes (lightweight)
        if include_episodes:
            try:
                ep_rows = await engine.graph.run(
                    "MATCH (ep:Episode) WHERE $include_inactive = true OR coalesce(ep.active, true) = true "
                    "RETURN ep.id AS id, ep.summary AS summary, ep.content AS content, ep.tags AS tags "
                    "ORDER BY ep.created_at DESC LIMIT $limit",
                    include_inactive=include_inactive,
                    limit=min(limit, 100),
                )
                ep_ids = []
                for er in ep_rows:
                    eid = er.get("id")
                    if not eid:
                        continue
                    ep_ids.append(eid)
                    nodes.append({
                        "id": eid,
                        "label": (er.get("summary") or er.get("content") or eid)[:40],
                        "type": "Episode",
                        "description": (er.get("content") or "")[:300],
                        "aliases": er.get("tags") or [],
                        "access_count": 0,
                        "group": "Episode",
                    })
                # Episode-entity links. Pull in mentioned entities that fell
                # outside the first page so an episode node is not left dangling.
                if ep_ids:
                    extra_ents = await engine.graph.run(
                        "MATCH (e:Entity)-[r:MENTIONED_IN]->(ep:Episode) "
                        "WHERE ep.id IN $ep_ids AND e.id IS NOT NULL "
                        "RETURN DISTINCT e.id AS id, e.name AS name, e.type AS type, "
                        "e.description AS description, e.aliases AS aliases LIMIT $limit",
                        ep_ids=ep_ids, limit=limit,
                    )
                    for r in extra_ents:
                        nid = r.get("id")
                        if not nid or nid in id_to_name:
                            continue
                        name = r.get("name") or nid
                        id_to_name[nid] = name
                        nodes.append({
                            "id": nid,
                            "label": name,
                            "type": r.get("type") or "Concept",
                            "description": r.get("description") or "",
                            "aliases": r.get("aliases") or [],
                            "access_count": 0,
                            "group": r.get("type") or "Concept",
                        })
                    link_rows = await engine.graph.run(
                        "MATCH (e:Entity)-[r:MENTIONED_IN]->(ep:Episode) "
                        "WHERE ep.id IN $ep_ids AND e.id IS NOT NULL AND ep.id IS NOT NULL "
                        "RETURN e.id AS from_id, ep.id AS to_id, coalesce(r.target, ep.id) AS target "
                        "LIMIT $limit",
                        ep_ids=ep_ids, limit=max(limit, 500),
                    )
                    for lr in link_rows:
                        fid = lr.get("from_id")
                        tid = lr.get("target") or lr.get("to_id")
                        if not fid or not tid:
                            continue
                        edges.append({
                            "id": f"{fid}->MENTIONED_IN->{tid}",
                            "from": fid,
                            "to": tid,
                            "source": fid,
                            "target": tid,
                            "label": "MENTIONED_IN",
                        })
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

    async def api_orphans(request: Request):
        include_single = request.query_params.get("include_single_edge", "false").lower() in ("1", "true", "yes")
        include_inactive = request.query_params.get("include_inactive", "false").lower() in ("1", "true", "yes")
        try:
            limit = int(request.query_params.get("limit", "5000"))
        except ValueError:
            limit = 5000
        if not await engine.graph.ping():
            return _cors(JSONResponse({"orphans": [], "count": 0, "error": "neo4j unreachable"}, status_code=503))
        data = await engine.list_orphans(
            include_single_edge=include_single, limit=limit, include_inactive=include_inactive
        )
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
        Route("/api/orphans", api_orphans, methods=["GET", "OPTIONS"]),
        Route("/api/graph", _options, methods=["OPTIONS"]),
        Route("/api/stats", _options, methods=["OPTIONS"]),
        Route("/api/entity", _options, methods=["OPTIONS"]),
        Route("/api/recall", _options, methods=["OPTIONS"]),
        Route("/api/episodes", _options, methods=["OPTIONS"]),
        Route("/api/orphans", _options, methods=["OPTIONS"]),
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
        try:
            pending = await engine.resume_pending()
            log.info("Requeued pending episodes: %s", len(pending.get("queued") or []))
        except Exception as exc:  # noqa: BLE001
            log.error("Could not resume pending episodes: %s", type(exc).__name__)

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
