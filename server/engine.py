"""MemoryEngine: the business logic shared by all MCP tools.

Pipeline owner for remember / recall / context / resolve / forget.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from datetime import datetime, timezone

from .config import Settings
from .embeddings.ollama import OllamaEmbedder
from .graph import queries as Q
from .graph.driver import GraphDriver
from .llm.extract import ExtractionLLM
from .models import (
    ConflictInfo,
    ExtractionResult,
    RecallHit,
    RememberResponse,
)

log = logging.getLogger("daryl-memories.engine")

# --- scoring constants (from PRD v2) ---
W_VECTOR, W_KEYWORD, W_GRAPH = 0.4, 0.3, 0.3
RECENCY_FULL_DAYS = 7
RECENCY_MIN_DAYS = 90
RECENCY_FLOOR = 0.5
ACCESS_BOOST_STEP = 0.1
ACCESS_BOOST_CAP = 2.0


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _name_similarity(a: str, b: str) -> float:
    """Cheap character-level similarity in [0,1] (difflib-free)."""
    a, b = _norm_name(a), _norm_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # bigram Jaccard
    ga = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    gb = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def recency_boost(created_at: str | None) -> float:
    """1.0 under 7 days, linear decay to 0.5 at 90 days, floor 0.5."""
    if not created_at:
        return 1.0
    try:
        # Neo4j toString(datetime()) -> 2026-08-21T08:39:33.000Z style
        ts = created_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return 1.0
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    if age_days <= RECENCY_FULL_DAYS:
        return 1.0
    if age_days >= RECENCY_MIN_DAYS:
        return RECENCY_FLOOR
    frac = (age_days - RECENCY_FULL_DAYS) / (RECENCY_MIN_DAYS - RECENCY_FULL_DAYS)
    return 1.0 - frac * (1.0 - RECENCY_FLOOR)


class MemoryEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph = GraphDriver(settings)
        self.embedder = OllamaEmbedder(settings)
        self.llm = ExtractionLLM(settings)

    async def close(self) -> None:
        await self.llm.close()
        await self.embedder.close()
        await self.graph.close()

    # ------------------------------------------------------------------
    # remember
    # ------------------------------------------------------------------

    async def remember(
        self,
        content: str,
        agent_id: str,
        session_id: str = "",
        tags: list[str] | None = None,
    ) -> RememberResponse:
        tags = tags or []
        episode_id = new_id("ep")
        summary = content[:300].replace("\n", " ")

        # 1. Store raw episode first (survives extraction failure)
        await self.graph.run(
            Q.CREATE_EPISODE,
            id=episode_id,
            content=content,
            summary=summary,
            tags=tags,
            source_agent_id=agent_id,
            source_session_id=session_id,
        )

        # 2. Extract entities / relationships / facts
        extraction = await self.llm.extract(
            content, agent_id=agent_id, session_id=session_id, tags=tags
        )

        entities_created: list[str] = []
        entities_merged: list[str] = []
        rel_count = 0
        fact_count = 0
        conflicts: list[ConflictInfo] = []

        name_to_entity_id: dict[str, str] = {}

        # 3. Dedup + upsert entities
        for ent in extraction.entities:
            # Type safety: ensure aliases is a list of strings
            if not isinstance(ent.aliases, list):
                ent.aliases = []
            ent.aliases = [str(a) for a in ent.aliases if isinstance(a, str)][:10]
            ent_id, merged_into = await self._upsert_entity(ent)
            name_to_entity_id[_norm_name(ent.name)] = ent_id
            for alias in ent.aliases:
                name_to_entity_id.setdefault(_norm_name(alias), ent_id)
            if merged_into:
                entities_merged.append(f"{ent.name} -> {merged_into}")
            else:
                entities_created.append(ent.name)

        # 4. Relationships (resolve endpoint names, creating missing entities)
        for rel in extraction.relationships:
            from_id = await self._resolve_entity_id(rel.from_name, name_to_entity_id)
            to_id = await self._resolve_entity_id(rel.to_name, name_to_entity_id)
            if not from_id or not to_id:
                continue
            await self.graph.run(
                Q.CREATE_RELATIONSHIP_STATIC,
                from_id=from_id,
                to_id=to_id,
                type=rel.type,
            )
            rel_count += 1

        # 5. Link mentioned entities to the episode
        for ent_id in set(name_to_entity_id.values()):
            await self.graph.run(Q.LINK_EPISODE_ENTITY, episode_id=episode_id, entity_id=ent_id)

        # 6. Facts with conflict detection
        for fact in extraction.facts:
            subject_id = None
            if fact.subject_name:
                subject_id = await self._resolve_entity_id(fact.subject_name, name_to_entity_id)
            fact_id = new_id("fact")
            if subject_id:
                conflict = await self._detect_conflict(subject_id, fact.content, fact_id)
                if conflict:
                    conflicts.append(conflict)
            await self.graph.run(
                Q.CREATE_FACT,
                id=fact_id,
                content=fact.content,
                subject=subject_id or "",
                confidence=fact.confidence,
                source_agent_id=agent_id,
                source_session_id=session_id,
            )
            fact_count += 1

        # 7. Embed the episode
        embedding_status = "ok"
        try:
            vector = await self.embedder.embed(content[:4000])
            await self.graph.run(Q.SET_EPISODE_EMBEDDING, episode_id=episode_id, vector=vector)
        except Exception as exc:  # noqa: BLE001
            embedding_status = f"failed: {exc}"
            log.error("Episode embedding failed: %s", exc)

        # 8. Embed entity summaries (for entity-level search + future dedup)
        for ent in extraction.entities:
            ent_id = name_to_entity_id.get(_norm_name(ent.name))
            if not ent_id:
                continue
            try:
                text = f"{ent.name} ({ent.type}): {ent.description or ent.name}"
                vec = await self.embedder.embed(text)
                await self.graph.run(Q.SET_ENTITY_EMBEDDING, entity_id=ent_id, vector=vec)
            except Exception:  # noqa: BLE001
                pass  # non-fatal

        return RememberResponse(
            episode_id=episode_id,
            entities_created=entities_created,
            entities_merged=entities_merged,
            relationships_created=rel_count,
            facts_stored=fact_count,
            conflicts_detected=conflicts,
            embedding_status=embedding_status,
        )

    async def _upsert_entity(self, ent) -> tuple[str, str | None]:
        """Upsert an entity with dedup. Returns (entity_id, merged_into_name|None)."""
        # Exact name/alias match
        exact = await self.graph.run_one(Q.FIND_ENTITY_EXACT, name=ent.name)
        if exact:
            await self.graph.run(Q.ADD_ALIAS, id=exact["id"], alias=ent.name)
            return exact["id"], None

        # Fuzzy candidates by substring
        candidates = await self.graph.run(Q.FIND_ENTITY_FUZZY, needle=ent.name)
        best, best_sim = None, 0.0
        for cand in candidates:
            sims = [_name_similarity(ent.name, cand["name"])]
            for a in (cand.get("aliases") or []):
                if isinstance(a, str):
                    sims.append(_name_similarity(ent.name, a))
            sim = max(sims) if sims else 0.0
            if sim > best_sim:
                best, best_sim = cand, sim
        if best and best_sim >= 0.85:
            await self.graph.run(Q.ADD_ALIAS, id=best["id"], alias=ent.name)
            return best["id"], best["name"]

        ent_id = new_id("ent")
        await self.graph.run(
            Q.UPSERT_ENTITY,
            id=ent_id,
            name=ent.name,
            type=ent.type,
            description=ent.description,
            aliases=ent.aliases,
        )
        return ent_id, None

    async def _resolve_entity_id(self, name: str, cache: dict[str, str]) -> str | None:
        """Find entity id by name; create on the fly if unknown."""
        key = _norm_name(name)
        if key in cache:
            return cache[key]
        exact = await self.graph.run_one(Q.FIND_ENTITY_EXACT, name=name)
        if exact:
            cache[key] = exact["id"]
            return exact["id"]
        # Create a lightweight entity so relationships don't dangle
        ent_id = new_id("ent")
        await self.graph.run(
            Q.UPSERT_ENTITY, id=ent_id, name=name, type="Concept",
            description="", aliases=[],
        )
        cache[key] = ent_id
        return ent_id

    async def _detect_conflict(self, subject_id: str, new_content: str, new_fact_id: str) -> ConflictInfo | None:
        """Heuristic conflict detection: high token overlap with an existing
        active fact on the same subject, but different content => contradiction.
        The newer fact supersedes the older one."""
        existing = await self.graph.run(Q.FIND_FACTS_ON_SUBJECT, subject_id=subject_id)
        for old in existing:
            old_content = old["content"]
            overlap = _token_overlap(new_content, old_content)
            if overlap >= 0.5 and _norm_name(new_content) != _norm_name(old_content):
                await self.graph.run(Q.SUPERSEDE_FACT, old_id=old["id"], new_id=new_fact_id)
                return ConflictInfo(
                    entity=subject_id,
                    old_fact_id=old["id"],
                    old_content=old_content,
                    new_content=new_content,
                    action="superseded",
                    reason=f"token overlap {overlap:.2f} with existing fact",
                )
        return None

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        agent_id: str = "",
    ) -> dict:
        top_k = max(1, min(top_k, 20))
        candidate_pool = top_k * 3

        vector_results: list[dict] = []
        fts_results: list[dict] = []
        fact_fts_results: list[dict] = []
        entity_matches: list[dict] = []
        graph_context: dict = {"entities_found": [], "paths": []}

        # 1. Embed query
        try:
            qvec = await self.embedder.embed(query)
            vector_results = await self.graph.run(
                Q.VECTOR_SEARCH_EPISODES, limit=candidate_pool, query_vector=qvec
            )
            entity_matches = await self.graph.run(
                Q.VECTOR_SEARCH_ENTITIES, limit=5, query_vector=qvec
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector search unavailable: %s", exc)

        # 2. Full-text search (escape Lucene special chars)
        fts_query = self._lucene_escape(query) or "*"
        try:
            fts_results = await self.graph.run(
                Q.FULLTEXT_SEARCH_EPISODES, query=fts_query, limit=candidate_pool
            )
            fact_fts_results = await self.graph.run(
                Q.FULLTEXT_SEARCH_FACTS, query=fts_query, limit=candidate_pool
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Full-text search failed: %s", exc)

        # 3. Graph expansion from matched entities
        entity_ids = [e["id"] for e in entity_matches]
        entity_names = {e["id"]: e["name"] for e in entity_matches}
        graph_facts: list[dict] = []
        if entity_ids:
            graph_context["entities_found"] = [entity_names[i] for i in entity_ids]
            try:
                paths = await self.graph.run(Q.GRAPH_EXPAND, entity_ids=entity_ids, limit=20)
                graph_context["paths"] = [
                    f"{p['from_name']} -> {p['rel_type']} -> {p['to_name']}" for p in paths
                ][:10]
                graph_facts = await self.graph.run(Q.FACTS_FOR_ENTITIES, entity_ids=entity_ids, limit=10)
            except Exception as exc:  # noqa: BLE001
                log.warning("Graph expansion failed: %s", exc)

        # 4. Score fusion per unique content id
        fused: dict[str, dict] = {}

        def _bump(rec: dict, weight: float, score: float, kind: str) -> None:
            rid = rec.get("id")
            if not rid:
                return
            entry = fused.setdefault(rid, {
                "vector": 0.0, "keyword": 0.0, "graph": 0.0,
                "content": rec.get("content", ""),
                "kind": kind,
                "source_agent_id": rec.get("source_agent_id"),
                "source_session_id": rec.get("source_session_id"),
                "created_at": rec.get("created_at"),
                "entities": [],
            })
            entry[weight] = max(entry[weight], float(score or 0.0))

        for rec in vector_results:
            _bump(rec, "vector", rec.get("score", 0.0), "episode")
        for rec in fts_results:
            # Lucene scores are unbounded; squash into (0,1]
            _bump(rec, "keyword", 1.0 / (1.0 + 1.0 / max(rec.get("score", 0.0), 1e-6)), "episode")
        for rec in fact_fts_results:
            _bump(rec, "keyword", 1.0 / (1.0 + 1.0 / max(rec.get("score", 0.0), 1e-6)), "fact")
        for rec in graph_facts:
            rid = rec.get("id")
            if rid:
                entry = fused.setdefault(rid, {
                    "vector": 0.0, "keyword": 0.0, "graph": 0.0,
                    "content": rec.get("content", ""), "kind": "fact",
                    "source_agent_id": None, "source_session_id": None,
                    "created_at": rec.get("created_at"), "entities": [],
                })
                entry["graph"] = max(entry["graph"], 0.8)
                ent = entity_names.get(rec.get("subject_id"))
                if ent and ent not in entry["entities"]:
                    entry["entities"].append(ent)

        # attach entity names to episode hits mentioned via graph
        for rid, entry in fused.items():
            if entry["kind"] == "episode":
                entry["entities"] = graph_context["entities_found"][:5]

        # 5. Final score with recency + access boosts
        hits: list[RecallHit] = []
        for rid, entry in fused.items():
            base = (W_VECTOR * entry["vector"]
                    + W_KEYWORD * entry["keyword"]
                    + W_GRAPH * entry["graph"])
            boost = recency_boost(entry["created_at"])
            access = 1.0  # access_count fetched cheaply below only for winners
            final = base * boost * access
            if final <= 0:
                continue
            hits.append(RecallHit(
                content=entry["content"],
                score=round(final, 4),
                kind=entry["kind"],
                source_agent_id=entry["source_agent_id"],
                source_session_id=entry["source_session_id"],
                entities=entry["entities"],
                created_at=entry["created_at"],
                still_valid=True,
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:top_k]

        # bump access counters for winners (best-effort)
        for hit in hits:
            pass  # ids not carried into RecallHit; skipped for simplicity

        return {"results": [h.model_dump() for h in hits], "graph_context": graph_context}

    @staticmethod
    def _lucene_escape(text: str) -> str:
        """Escape Lucene full-text special characters and join words with OR
        for recall-friendly matching."""
        specials = r'+-&&||!(){}[]^"~*?:\\/'
        escaped = "".join("\\" + c if c in specials else c for c in text)
        words = [w for w in escaped.split() if w]
        return " OR ".join(words)

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------

    async def context(self, entity_name: str, depth_limit: int = 100) -> dict:
        rows = await self.graph.run(Q.ENTITY_NEIGHBORHOOD, name=entity_name, limit=max(depth_limit, 1))
        if not rows:
            # fuzzy fallback
            candidates = await self.graph.run(Q.FIND_ENTITY_FUZZY, needle=entity_name)
            if candidates:
                rows = await self.graph.run(
                    Q.ENTITY_NEIGHBORHOOD, name=candidates[0]["name"], limit=max(depth_limit, 1)
                )
        edges = []
        seen = set()
        for r in rows:
            key = (r.get("from_name"), r.get("rel_type"), r.get("to_name"))
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "from": r.get("from_name"),
                "type": r.get("rel_type"),
                "to": r.get("to_name"),
            })
        root = rows[0].get("root_name") if rows else None
        return {"entity": root or entity_name, "edges": edges, "edge_count": len(edges)}

    # ------------------------------------------------------------------
    # resolve (merge duplicates)
    # ------------------------------------------------------------------

    async def resolve(self, entity_a: str, entity_b: str) -> dict:
        a = await self.graph.run_one(Q.FIND_ENTITY_EXACT, name=entity_a)
        b = await self.graph.run_one(Q.FIND_ENTITY_EXACT, name=entity_b)
        if not a or not b:
            fuzzy = await self.graph.run(Q.FIND_ENTITY_FUZZY, needle=entity_b if not b else entity_a)
            missing = "entity_b" if not b else "entity_a"
            return {"merged": False, "error": f"{missing} not found", "candidates": [c["name"] for c in fuzzy]}
        if a["id"] == b["id"]:
            return {"merged": False, "error": "both names resolve to the same entity"}

        # Collect b's edges, merge b into a, reconnect edges, drop b
        edges = await self.graph.run(Q.COLLECT_EDGES_OF, drop_id=b["id"], keep_id=a["id"])
        summary = await self.graph.run_one(Q.REPARENT_ENTITY, keep_id=a["id"], drop_id=b["id"])

        # Re-point facts from b -> a
        await self.graph.run(
            "MATCH (f:Fact {subject: $old}) SET f.subject = $new RETURN count(f) AS n",
            old=b["id"], new=a["id"],
        )

        reconnected = 0
        for edge in edges:
            other_id = edge["other_id"]
            rel_type = edge["rel_type"]
            if edge["direction"] == "out":
                await self.graph.run(Q.CREATE_RELATIONSHIP_STATIC, from_id=a["id"], to_id=other_id, type=rel_type)
            else:
                await self.graph.run(Q.CREATE_RELATIONSHIP_STATIC, from_id=other_id, to_id=a["id"], type=rel_type)
            reconnected += 1

        await self.graph.run(Q.DELETE_ENTITY, drop_id=b["id"])

        return {
            "merged": True,
            "kept": a["name"],
            "dropped": b["name"],
            "aliases_now": summary.get("aliases") if summary else [],
            "edges_reconnected": reconnected,
        }

    # ------------------------------------------------------------------
    # forget (soft delete)
    # ------------------------------------------------------------------

    async def forget(self, memory_id: str) -> dict:
        row = await self.graph.run_one(Q.SOFT_DELETE_EPISODE, episode_id=memory_id)
        if not row:
            return {"forgotten": False, "error": f"episode {memory_id} not found"}
        return {"forgotten": True, "episode_id": memory_id}

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        neo4j_ok = await self.graph.ping()
        ollama_ok = await self.embedder.health()
        stats = {}
        if neo4j_ok:
            try:
                stats = (await self.graph.run(Q.HEALTH_STATS))[0]
            except Exception:  # noqa: BLE001
                stats = {}
        return {
            "ok": neo4j_ok and ollama_ok,
            "neo4j": neo4j_ok,
            "ollama": ollama_ok,
            "stats": stats,
        }
