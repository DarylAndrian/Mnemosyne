"""MemoryEngine: the business logic shared by all MCP tools.

Pipeline owner for remember / recall / context / resolve / forget.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone

from .config import Settings
from .embeddings.ollama import OllamaEmbedder
from .entity_rules import (
    REPAIR_TARGETS,
    acceptable_alias,
    grounded_in_text,
    is_absorbed_name,
    is_generic,
    mentions_standalone,
    mine_entities,
    too_common,
)
from .graph import queries as Q
from .graph.driver import GraphDriver
from .llm.extract import ExtractionLLM
from .models import (
    ConflictInfo,
    ExtractionResult,
    RecallHit,
    RememberResponse,
)
from .secrets import REJECTION_MESSAGE, find_secret, redact_secrets
from .textutil import (
    decode_cursor,
    encode_cursor,
    estimate_tokens,
    safe_error,
    snippet_around,
)

log = logging.getLogger("mnemosyne.engine")

# --- scoring constants (from PRD v2) ---
W_VECTOR, W_KEYWORD, W_GRAPH = 0.4, 0.3, 0.3
RECENCY_FULL_DAYS = 7
RECENCY_MIN_DAYS = 90
RECENCY_FLOOR = 0.5
ACCESS_BOOST_STEP = 0.1
ACCESS_BOOST_CAP = 2.0

# Near-duplicate detection: token overlap at/above this within the 5-min
# window (see RECENT_EPISODES_IN_WINDOW / FIND_EPISODE_BY_HASH_RECENT)
NEAR_DUP_TOKEN_OVERLAP = 0.97

MAX_PROCESS_ATTEMPTS = 3
# resolve() refuses to merge names below this surface similarity unless
# fuzzy or force is set. Bigram-Jaccard over normalised names: "Slingshot
# Results Hub" vs "Renaldi Tjahaya" scores ~0.03, real variants score 0.5+.
MIN_RESOLVE_SIMILARITY = 0.35
INDEX_TOKEN_BUDGET = 2000
SNIPPET_CHARS = 200


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _content_hash(content: str) -> str:
    """Stable hash of whitespace-normalized, lowercased content."""
    normalized = re.sub(r"\s+", " ", content.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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


def _names_look_related(a: str, b: str) -> bool:
    """Surface-similarity gate for resolve(). Same name, any shared token, or
    enough bigram overlap. Unrelated names (different people/places) return
    False so resolve() refuses unless fuzzy or force is set."""
    if _norm_name(a) == _norm_name(b):
        return True
    if _token_overlap(a, b) > 0:
        return True
    return _name_similarity(a, b) >= MIN_RESOLVE_SIMILARITY


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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
        self._bg_tasks: set[asyncio.Task] = set()
        # One extraction at a time. Parallel Ollama calls were timing out and
        # leaving episodes failed before an embedding was written.
        #
        # Two lanes so a burst of retries can never starve fresh saves:
        #   _process_sem -> new episodes (attempt 1)
        #   _retry_sem   -> retries (attempt 2+) and reprocess_failed requeues
        # asyncio.Semaphore is FIFO, so sharing one lane meant a single
        # reprocess_failed(28 episodes) pushed new work to the back of a
        # 28-deep line; fresh episodes sat at "pending / attempts 0".
        self._process_sem = asyncio.Semaphore(1)
        self._retry_sem = asyncio.Semaphore(1)
        self._retry_backoff = (2.0, 8.0, 30.0)
        self._sleep = asyncio.sleep

    async def close(self) -> None:
        await self.llm.close()
        await self.embedder.close()
        await self.graph.close()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    # ------------------------------------------------------------------
    # remember
    # ------------------------------------------------------------------

    async def remember(
        self,
        content: str,
        agent_id: str,
        session_id: str = "",
        tags: list[str] | None = None,
        request_id: str = "",
        key: str = "",
    ) -> RememberResponse:
        tags = tags or []
        key = (key or "").strip()

        # Reject before any read or write so secrets are never stored or logged.
        if find_secret(content):
            return RememberResponse(
                episode_id="",
                processing=False,
                status="rejected",
                rejected=True,
                error=REJECTION_MESSAGE,
                key=key,
            )

        # --- Dedup gate (before any write) ---------------------------------

        # 1. Idempotency key: same request_id always returns the original
        if request_id:
            existing = await self.graph.run_one(Q.FIND_EPISODE_BY_REQUEST_ID, rid=request_id)
            if existing:
                status = existing.get("status") or "processed"
                return RememberResponse(
                    episode_id=existing["id"],
                    deduplicated=True,
                    processing=(status == "pending"),
                    status=status,
                    key=key,
                )

        # 2. Stable key: replace the current episode and keep the old text.
        if key:
            existing = await self.graph.run_one(Q.FIND_EPISODE_BY_KEY, key=key)
            if existing:
                return await self._replace_keyed_episode(existing, content, tags, key)

        # 3. Exact content hash within the 5-min window
        content_hash = _content_hash(content)
        existing = await self.graph.run_one(Q.FIND_EPISODE_BY_HASH_RECENT, content_hash=content_hash)
        if existing:
            status = existing.get("status") or "processed"
            return RememberResponse(
                episode_id=existing["id"],
                deduplicated=True,
                processing=(status == "pending"),
                status=status,
            )

        # 4. Near-duplicate: high token overlap with a very recent episode
        recent = await self.graph.run(Q.RECENT_EPISODES_IN_WINDOW)
        for old in recent:
            if _token_overlap(content, old.get("content") or "") >= NEAR_DUP_TOKEN_OVERLAP:
                status = old.get("status") or "processed"
                return RememberResponse(
                    episode_id=old["id"],
                    deduplicated=True,
                    processing=(status == "pending"),
                    status=status,
                )

        # --- Store fast, process in background ------------------------------

        episode_id = new_id("ep")
        summary = content[:300].replace("\n", " ")

        # Store raw episode first (survives extraction failure)
        await self.graph.run(
            Q.CREATE_EPISODE,
            id=episode_id,
            content=content,
            summary=summary,
            tags=tags,
            source_agent_id=agent_id,
            source_session_id=session_id,
            request_id=request_id,
            content_hash=content_hash,
            key=key,
        )

        self._spawn(self._process_episode(episode_id, content, agent_id, session_id, tags))

        return RememberResponse(
            episode_id=episode_id,
            processing=True,
            status="pending",
            key=key,
        )

    async def _replace_keyed_episode(
        self,
        existing: dict,
        content: str,
        tags: list[str],
        key: str,
    ) -> RememberResponse:
        """Overwrite one keyed episode and archive the previous body."""
        previous = existing.get("content") or ""
        if _content_hash(content) == _content_hash(previous):
            status = existing.get("status") or "processed"
            return RememberResponse(
                episode_id=existing["id"],
                deduplicated=True,
                processing=(status == "pending"),
                status=status,
                key=key,
                updated=False,
            )
        version = int(existing.get("version") or 1)
        summary = content[:300].replace("\n", " ")
        await self.graph.run(
            Q.INSERT_EPISODE_VERSION,
            id=new_id("ver"),
            episode_id=existing["id"],
            content=previous,
            summary=existing.get("summary") or previous[:300],
            version=version,
        )
        await self.graph.run(
            Q.UPDATE_EPISODE_CONTENT,
            episode_id=existing["id"],
            content=content,
            summary=summary,
            content_hash=_content_hash(content),
            tags=tags,
            version=version + 1,
        )
        self._spawn(
            self._process_episode(
                existing["id"],
                content,
                existing.get("source_agent_id") or "",
                existing.get("source_session_id") or "",
                tags,
            )
        )
        return RememberResponse(
            episode_id=existing["id"],
            processing=True,
            status="pending",
            updated=True,
            key=key,
        )

    async def _process_episode(
        self,
        episode_id: str,
        content: str,
        agent_id: str,
        session_id: str,
        tags: list[str],
        lane: str = "new",
    ) -> None:
        """Background extraction + embedding. Retries up to 3 times with backoff.

        lane="new"  -> first attempt uses the fresh-episode lane
        lane="retry"-> every attempt uses the retry lane (reprocess requeues)

        Attempts after the first always move to the retry lane, so a failing
        episode releases the fresh lane immediately instead of re-queuing in
        front of brand new saves.
        """
        fresh_sem = self._process_sem if lane == "new" else self._retry_sem
        # Embed first so a later extraction error still leaves the episode searchable.
        await self._embed_episode(episode_id, content, fresh_sem)
        for attempt in range(1, MAX_PROCESS_ATTEMPTS + 1):
            sem = fresh_sem if attempt == 1 else self._retry_sem
            try:
                async with sem:
                    await self._extract_and_link(episode_id, content, agent_id, session_id, tags)
                    await self.graph.run(
                        Q.MARK_EPISODE_PROCESSED,
                        episode_id=episode_id,
                        attempts=attempt,
                    )
                return
            except Exception as exc:  # noqa: BLE001
                message = safe_error(exc, content)
                log.error(
                    "Background processing failed for %s (attempt %s): %s",
                    episode_id,
                    attempt,
                    message,
                )
                status = "failed" if attempt >= MAX_PROCESS_ATTEMPTS else "pending"
                try:
                    await self.graph.run(
                        Q.SET_EPISODE_FAILURE,
                        episode_id=episode_id,
                        status=status,
                        error_message=message,
                        attempts=attempt,
                    )
                except Exception as persist_exc:  # noqa: BLE001
                    log.error(
                        "Could not store failure for %s: %s",
                        episode_id,
                        type(persist_exc).__name__,
                    )
                if attempt >= MAX_PROCESS_ATTEMPTS:
                    return
                delay = self._retry_backoff[min(attempt - 1, len(self._retry_backoff) - 1)]
                await self._sleep(delay)

    async def _embed_episode(
        self, episode_id: str, content: str, sem: asyncio.Semaphore | None = None
    ) -> None:
        try:
            async with (sem or self._process_sem):
                vector = await self.embedder.embed((content or "")[:4000])
                await self.graph.run(Q.SET_EPISODE_EMBEDDING, episode_id=episode_id, vector=vector)
        except Exception as exc:  # noqa: BLE001
            log.error("Episode embedding failed for %s: %s", episode_id, safe_error(exc, content))

    def _select_entities(self, content: str, extraction: ExtractionResult) -> list:
        """LLM entities plus names mined from this episode, grounded and de-duplicated."""
        merged = []
        seen: set[str] = set()
        for ent in list(extraction.entities) + mine_entities(content):
            name = (ent.name or "").strip()
            if not name or is_generic(name) or not grounded_in_text(name, content):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            ent.aliases = [a for a in (ent.aliases or []) if acceptable_alias(name, a)]
            merged.append(ent)
        return merged

    async def _extract_and_link(
        self,
        episode_id: str,
        content: str,
        agent_id: str,
        session_id: str,
        tags: list[str],
    ) -> None:
        await self.graph.run(Q.CLEAR_EPISODE_LINKS, episode_id=episode_id)
        await self.graph.run(Q.DELETE_EPISODE_FACTS, episode_id=episode_id)

        extraction = await self.llm.extract(
            content, agent_id=agent_id, session_id=session_id, tags=tags
        )
        entities = self._select_entities(content, extraction)
        name_to_entity_id: dict[str, str] = {}
        linked_ids: set[str] = set()

        for ent in entities:
            ent_id, _merged_into = await self._upsert_entity(ent)
            name_to_entity_id[_norm_name(ent.name)] = ent_id
            if await self._is_overly_common(ent_id):
                continue
            await self.graph.run(Q.LINK_EPISODE_ENTITY, episode_id=episode_id, entity_id=ent_id)
            linked_ids.add(ent_id)

        for rel in extraction.relationships:
            if not grounded_in_text(rel.from_name, content) or not grounded_in_text(rel.to_name, content):
                continue
            if is_generic(rel.from_name) or is_generic(rel.to_name):
                continue
            from_id = name_to_entity_id.get(_norm_name(rel.from_name))
            to_id = name_to_entity_id.get(_norm_name(rel.to_name))
            if not from_id or not to_id or from_id not in linked_ids or to_id not in linked_ids:
                continue
            await self.graph.run(
                Q.CREATE_RELATIONSHIP_STATIC,
                from_id=from_id,
                to_id=to_id,
                type=rel.type,
            )

        for fact in extraction.facts:
            subject_id = ""
            if fact.subject_name:
                subject_id = name_to_entity_id.get(_norm_name(fact.subject_name), "")
                if subject_id not in linked_ids:
                    subject_id = ""
            fact_id = new_id("fact")
            await self.graph.run(
                Q.CREATE_FACT,
                id=fact_id,
                content=fact.content,
                subject=subject_id,
                confidence=fact.confidence,
                source_agent_id=agent_id,
                source_session_id=session_id,
                episode_id=episode_id,
            )
            if subject_id:
                await self._detect_conflict(subject_id, fact.content, fact_id)

        for ent in entities:
            ent_id = name_to_entity_id.get(_norm_name(ent.name))
            if not ent_id or ent_id not in linked_ids:
                continue
            try:
                text = f"{ent.name} ({ent.type}): {ent.description or ent.name}"
                vec = await self.embedder.embed(text)
                await self.graph.run(Q.SET_ENTITY_EMBEDDING, entity_id=ent_id, vector=vec)
            except Exception:  # noqa: BLE001
                pass

    async def _is_overly_common(self, entity_id: str) -> bool:
        row = await self.graph.run_one(Q.ENTITY_LINK_STATS, id=entity_id)
        if not row:
            return False
        return too_common(int(row.get("linked") or 0), int(row.get("total") or 0))

    async def episode_status(self, episode_id: str) -> dict:
        row = await self.graph.run_one(Q.GET_EPISODE_STATUS, episode_id=episode_id)
        if not row:
            return {
                "episode_id": episode_id,
                "status": "not_found",
                "error_message": "",
                "attempts": 0,
                "last_attempt_at": None,
            }
        return {
            "episode_id": row["id"],
            "status": row.get("status") or "processed",
            "error_message": row.get("error_message") or "",
            "attempts": int(row.get("attempts") or 0),
            "last_attempt_at": row.get("last_attempt_at"),
            "request_id": row.get("request_id") or "",
        }

    async def reprocess_failed(
        self,
        episode_ids: list[str] | None = None,
        wait: bool = False,
    ) -> dict:
        """Re-queue failed episodes, or the given ids whatever their status."""
        if episode_ids:
            rows = await self.graph.run(Q.GET_EPISODES_BY_IDS, ids=list(episode_ids))
            found = {row["id"] for row in rows}
            missing = [eid for eid in episode_ids if eid not in found]
        else:
            rows = await self.graph.run(Q.LIST_FAILED_EPISODES, limit=1000)
            missing = []
        queued: list[str] = []
        skipped: list[str] = []
        reset: list[str] = []
        tasks = []
        for row in rows:
            if row.get("active") is False:
                skipped.append(row["id"])
                continue
            # Reset BEFORE enqueueing: without this a requeued episode kept
            # status="failed" / attempts=N / its old error_message, so callers
            # saw "queued" but list_episodes(status="failed") never changed.
            try:
                await self.graph.run(Q.RESET_EPISODE_FOR_REPROCESS, episode_id=row["id"])
                reset.append(row["id"])
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Could not reset %s for reprocess: %s",
                    row["id"],
                    type(exc).__name__,
                )
            queued.append(row["id"])
            tags = row.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            tasks.append(
                self._spawn(
                    self._process_episode(
                        row["id"],
                        row.get("content") or "",
                        row.get("source_agent_id") or "",
                        row.get("source_session_id") or "",
                        tags,
                        lane="retry",
                    )
                )
            )
        if wait and tasks:
            await asyncio.gather(*tasks)
        result: dict = {
            "queued": queued,
            "reset": reset,
            "missing": missing,
            "skipped": skipped,
        }
        if wait:
            result["results"] = [await self.episode_status(eid) for eid in queued]
        return result

    async def _upsert_entity(self, ent) -> tuple[str, str | None]:
        """Upsert by exact name only. Fuzzy collapse is opt-in via resolve(fuzzy=True)."""
        exact = await self.graph.run_one(Q.FIND_ENTITY_BY_NAME_CI, name=ent.name)
        if exact:
            return exact["id"], None

        ent_id = new_id("ent")
        aliases = [a for a in (ent.aliases or []) if acceptable_alias(ent.name, a)]
        await self.graph.run(
            Q.UPSERT_ENTITY,
            id=ent_id,
            name=ent.name,
            type=ent.type,
            description=ent.description,
            aliases=aliases,
        )
        return ent_id, None

    async def _detect_conflict(self, subject_id: str, new_content: str, new_fact_id: str) -> ConflictInfo | None:
        """Heuristic conflict detection: high token overlap with an existing
        active fact on the same subject, but different content => contradiction.
        The newer fact supersedes the older one."""
        existing = await self.graph.run(Q.FIND_FACTS_ON_SUBJECT, subject_id=subject_id)
        for old in existing:
            if old["id"] == new_fact_id:
                continue
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
        mode: str = "full",
    ) -> dict:
        mode = (mode or "full").strip().lower()
        if mode not in ("full", "snippet"):
            return {"results": [], "error": "mode must be 'full' or 'snippet'"}
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
            log.warning("Vector search unavailable: %s", safe_error(exc))

        # 2. Full-text search (escape Lucene special chars)
        fts_query = self._lucene_escape(query) or "*"
        try:
            fts_results = await self.graph.run(
                Q.FULLTEXT_SEARCH_EPISODES, q=fts_query, limit=candidate_pool
            )
            fact_fts_results = await self.graph.run(
                Q.FULLTEXT_SEARCH_FACTS, q=fts_query, limit=candidate_pool
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Full-text search failed: %s", safe_error(exc))

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
                log.warning("Graph expansion failed: %s", safe_error(exc))

        # 4. Score fusion per unique content id
        fused: dict[str, dict] = {}

        def _bump(rec: dict, weight: str, score: float, kind: str) -> None:
            rid = rec.get("id")
            if not rid:
                return
            entry = fused.setdefault(rid, {
                "vector": 0.0, "keyword": 0.0, "graph": 0.0,
                "content": rec.get("content", ""),
                "kind": kind,
                "source_agent_id": rec.get("source_agent_id"),
                "source_session_id": rec.get("source_session_id"),
                "source": rec.get("source_agent_id"),
                "created_at": rec.get("created_at"),
                "entities": [],
                "episode_id": rid if kind == "episode" else None,
            })
            entry[weight] = max(entry[weight], float(score or 0.0))
            if kind == "episode":
                entry["episode_id"] = rid
                entry["kind"] = "episode"

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
                    "source_agent_id": rec.get("source_agent_id"),
                    "source_session_id": None,
                    "source": rec.get("source_agent_id"),
                    "created_at": rec.get("created_at"), "entities": [],
                    "episode_id": None,
                })
                entry["graph"] = max(entry["graph"], 0.8)
                ent = entity_names.get(rec.get("subject_id"))
                if ent and ent not in entry["entities"]:
                    entry["entities"].append(ent)

        episode_ids = [rid for rid, entry in fused.items() if entry["kind"] == "episode"]
        if episode_ids:
            try:
                linked = await self.graph.run(Q.EPISODE_ENTITY_NAMES, ids=episode_ids)
                by_episode = {row["episode_id"]: row.get("names") or [] for row in linked}
            except Exception as exc:  # noqa: BLE001
                log.warning("Episode entity lookup failed: %s", safe_error(exc))
                by_episode = {}
            for rid, entry in fused.items():
                if entry["kind"] == "episode":
                    entry["entities"] = list(by_episode.get(rid) or [])

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
                source=entry.get("source") or entry.get("source_agent_id"),
                entities=entry["entities"],
                created_at=entry["created_at"],
                still_valid=True,
                episode_id=entry.get("episode_id"),
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        if mode == "snippet":
            hits = [hit for hit in hits if hit.kind == "episode" and hit.episode_id][:top_k]
            return {
                "mode": "snippet",
                "results": [
                    {
                        "id": hit.episode_id,
                        "episode_id": hit.episode_id,
                        "score": hit.score,
                        "excerpt": snippet_around(hit.content, query, SNIPPET_CHARS),
                    }
                    for hit in hits
                ],
            }

        hits = hits[:top_k]
        return {
            "mode": "full",
            "results": [h.model_dump() for h in hits],
            "graph_context": graph_context,
        }

    @staticmethod
    def _lucene_escape(text: str) -> str:
        """Escape Lucene full-text special characters and join words with OR
        for recall-friendly matching."""
        specials = r'+-&&||!(){}[]^"~*?:\\/'
        escaped = "".join("\\" + c if c in specials else c for c in text)
        words = [w for w in escaped.split() if w]
        return " OR ".join(words)

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    async def list_episodes(
        self,
        text_contains: str = "",
        source: str = "",
        created_before: str = "",
        created_after: str = "",
        entity: str = "",
        status: str = "",
        limit: int = 50,
        cursor: str = "",
    ) -> dict:
        limit = max(1, min(int(limit or 50), 200))
        try:
            cursor_time, cursor_id = decode_cursor(cursor) if cursor else ("", "")
        except ValueError:
            return {"episodes": [], "next_cursor": None, "error": "invalid cursor"}
        rows = await self.graph.run(
            Q.LIST_EPISODES,
            text=text_contains or "",
            source=source or "",
            status=status or "",
            created_after=created_after or "",
            created_before=created_before or "",
            cursor_time=cursor_time,
            cursor_id=cursor_id,
            entity=entity or "",
            limit=limit + 1,
        )
        extra = len(rows) > limit
        rows = rows[:limit]
        episodes = []
        for row in rows:
            preview = redact_secrets(row.get("preview") or "")[:160]
            episodes.append({
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "source": row.get("source") or "",
                "status": row.get("status") or "processed",
                "preview": preview,
            })
        next_cursor = None
        if extra and episodes:
            last = episodes[-1]
            next_cursor = encode_cursor(last.get("created_at") or "", last.get("id") or "")
        return {"episodes": episodes, "next_cursor": next_cursor}

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------

    async def context(self, entity_name: str = "", depth_limit: int = 100) -> dict:
        if not (entity_name or "").strip():
            return await self._index_context()
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

    async def _index_context(self) -> dict:
        """INDEX body plus TOPIC CARD titles, trimmed to stay under 2,000 tokens."""
        index_row = await self.graph.run_one(Q.INDEX_EPISODE)
        card_rows = await self.graph.run(Q.TOPIC_CARD_EPISODES)
        cards = []
        seen_titles: set[str] = set()
        for row in card_rows:
            title = _topic_card_title(row)
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            cards.append({"episode_id": row.get("id"), "title": title})
        total_cards = len(cards)
        body = (index_row or {}).get("content") or ""
        episode_id = (index_row or {}).get("id")
        truncated_index = False
        truncated_cards = False
        while True:
            payload = {
                "mode": "index",
                "index": {
                    "episode_id": episode_id,
                    "title": "INDEX",
                    "content": body,
                },
                "topic_cards": cards,
                "topic_card_count": total_cards,
                "truncated": truncated_index or truncated_cards,
            }
            tokens = estimate_tokens(json.dumps(payload))
            if tokens <= INDEX_TOKEN_BUDGET:
                payload["approx_tokens"] = tokens
                return payload
            if len(body) > 400:
                body = body[: max(400, int(len(body) * 0.75))]
                truncated_index = True
                continue
            if len(cards) > 1:
                cards = cards[:-1]
                truncated_cards = True
                continue
            body = body[:800]
            truncated_index = True
            payload = {
                "mode": "index",
                "index": {"episode_id": episode_id, "title": "INDEX", "content": body},
                "topic_cards": cards[:1],
                "topic_card_count": total_cards,
                "truncated": True,
                "approx_tokens": estimate_tokens(body) + 50,
            }
            return payload

    async def get_by_key(self, key: str) -> dict:
        row = await self.graph.run_one(Q.FIND_EPISODE_BY_KEY, key=key)
        if not row:
            return {"found": False, "key": key}
        history = await self.graph.run(Q.LIST_EPISODE_VERSIONS, episode_id=row["id"])
        return {
            "found": True,
            "episode_id": row["id"],
            "key": row.get("key") or key,
            "content": row.get("content") or "",
            "summary": row.get("summary") or "",
            "version": int(row.get("version") or 1),
            "status": row.get("status") or "processed",
            "source": row.get("source_agent_id") or "",
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "history": [
                {
                    "version": item.get("version"),
                    "content": item.get("content") or "",
                    "created_at": item.get("created_at"),
                }
                for item in history
            ],
        }

    # ------------------------------------------------------------------
    # resolve (merge duplicates) / unresolve
    # ------------------------------------------------------------------

    async def _find_for_resolve(self, name: str, fuzzy: bool) -> tuple[dict | None, list[str]]:
        exact = await self.graph.run_one(Q.FIND_ENTITY_BY_NAME_CI, name=name)
        if exact:
            return exact, []
        candidates = await self.graph.run(Q.FIND_ENTITY_FUZZY, needle=name)
        names = [c.get("name") for c in candidates if c.get("name")]
        if not fuzzy:
            return None, names
        best, best_sim, second = None, 0.0, 0.0
        for cand in candidates:
            sims = [_name_similarity(name, cand.get("name") or "")]
            for alias in cand.get("aliases") or []:
                if isinstance(alias, str):
                    sims.append(_name_similarity(name, alias))
            sim = max(sims) if sims else 0.0
            if sim > best_sim:
                second = best_sim
                best, best_sim = cand, sim
            elif sim > second:
                second = sim
        if best and best_sim >= 0.85 and (best_sim - second) >= 0.02:
            return best, names
        return None, names

    async def _load_entity_state(self, entity: dict) -> dict:
        edges = await self.graph.run(Q.LIST_ENTITY_EDGES, id=entity["id"])
        episodes = await self.graph.run(Q.LIST_ENTITY_EPISODES, id=entity["id"])
        facts = await self.graph.run(Q.FIND_FACTS_ON_SUBJECT, subject_id=entity["id"])
        return {
            "id": entity["id"],
            "name": entity.get("name") or "",
            "type": entity.get("type") or "Concept",
            "description": entity.get("description") or "",
            "aliases": [a for a in (entity.get("aliases") or []) if isinstance(a, str)],
            "edges": [
                {
                    "rel_type": edge.get("rel_type"),
                    "other_id": edge.get("other_id"),
                    "direction": edge.get("direction"),
                }
                for edge in edges
            ],
            "episode_ids": [ep.get("id") for ep in episodes if ep.get("id")],
            "fact_ids": [fact.get("id") for fact in facts if fact.get("id")],
        }

    async def resolve(
        self,
        entity_a: str,
        entity_b: str,
        fuzzy: bool = False,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict:
        a, cands_a = await self._find_for_resolve(entity_a, fuzzy)
        b, cands_b = await self._find_for_resolve(entity_b, fuzzy)
        if not a or not b:
            missing = "entity_a" if not a else "entity_b"
            return {
                "merged": False,
                "dry_run": dry_run,
                "error": f"{missing} not found",
                "candidates": cands_a if not a else cands_b,
            }
        if a["id"] == b["id"]:
            return {"merged": False, "dry_run": dry_run, "error": "both names resolve to the same entity"}

        # Guard: refuse clearly-unrelated names unless fuzzy or force is set.
        # resolve() deletes entity_b, so a bad call is destructive - dry_run
        # previously reported would_merge=true for e.g. "Slingshot Results Hub"
        # vs "Renaldi Tjahaya".
        if not (fuzzy or force) and not _names_look_related(a["name"], b["name"]):
            sim = _name_similarity(a["name"], b["name"])
            tok = _token_overlap(a["name"], b["name"])
            warning = (
                f"'{a['name']}' and '{b['name']}' look unrelated "
                f"(similarity {sim:.2f}, token overlap {tok:.2f}) - refusing to merge. "
                "Pass force=true to merge anyway, or fuzzy=true if this is a loose match."
            )
            if dry_run:
                return {
                    "merged": False,
                    "dry_run": True,
                    "would_merge": False,
                    "warning": warning,
                    "kept": a["name"],
                    "dropped": b["name"],
                }
            return {"merged": False, "dry_run": False, "error": warning}

        state_a = await self._load_entity_state(a)
        state_b = await self._load_entity_state(b)
        aliases = _unique(state_a["aliases"] + [state_b["name"]])
        preview = {
            "kept": state_a["name"],
            "dropped": state_b["name"],
            "kept_id": state_a["id"],
            "dropped_id": state_b["id"],
            "edges": state_b["edges"],
            "episodes": state_b["episode_ids"],
            "facts": state_b["fact_ids"],
            "aliases_after": aliases,
        }
        if dry_run:
            return {"merged": False, "dry_run": True, "would_merge": True, "merge_preview": preview}

        merge_id = new_id("merge")
        before = json.dumps({"kept": state_a, "dropped": state_b})
        await self.graph.run(
            Q.CREATE_MERGE_LOG,
            id=merge_id,
            before_state=before,
            kept_name=state_a["name"],
            dropped_name=state_b["name"],
        )
        await self.graph.run(Q.SET_ENTITY_ALIASES, id=state_a["id"], aliases=aliases)
        reconnected = 0
        for edge in state_b["edges"]:
            other = edge.get("other_id")
            if not other or other == state_a["id"]:
                continue
            if edge.get("direction") == "out":
                await self.graph.run(
                    Q.CREATE_RELATIONSHIP_STATIC,
                    from_id=state_a["id"],
                    to_id=other,
                    type=edge.get("rel_type"),
                )
            else:
                await self.graph.run(
                    Q.CREATE_RELATIONSHIP_STATIC,
                    from_id=other,
                    to_id=state_a["id"],
                    type=edge.get("rel_type"),
                )
            reconnected += 1
        for episode_id in state_b["episode_ids"]:
            await self.graph.run(Q.LINK_EPISODE_ENTITY, episode_id=episode_id, entity_id=state_a["id"])
        if state_b["fact_ids"]:
            await self.graph.run(
                Q.REPOINT_FACTS,
                old=state_b["id"],
                new=state_a["id"],
                ids=state_b["fact_ids"],
            )
        await self.graph.run(Q.DELETE_ENTITY, drop_id=state_b["id"])
        return {
            "merged": True,
            "dry_run": False,
            "merge_id": merge_id,
            "kept": state_a["name"],
            "dropped": state_b["name"],
            "aliases_now": aliases,
            "edges_reconnected": reconnected,
        }

    async def unresolve(self, merge_id: str) -> dict:
        row = await self.graph.run_one(Q.GET_MERGE_LOG, id=merge_id)
        if not row:
            return {"undone": False, "error": f"merge {merge_id} not found"}
        if row.get("undone"):
            return {"undone": False, "error": "merge already undone", "merge_id": merge_id}
        try:
            state = json.loads(row.get("before_state") or "")
        except (TypeError, json.JSONDecodeError):
            return {"undone": False, "error": "merge log is unreadable", "merge_id": merge_id}
        kept = state.get("kept") or {}
        dropped = state.get("dropped") or {}
        if not kept.get("id") or not dropped.get("id"):
            return {"undone": False, "error": "merge log is missing entity ids", "merge_id": merge_id}
        if await self.graph.run_one(Q.GET_ENTITY_BY_ID, id=dropped["id"]):
            return {"undone": False, "error": "dropped entity already exists", "merge_id": merge_id}
        if not await self.graph.run_one(Q.GET_ENTITY_BY_ID, id=kept["id"]):
            return {"undone": False, "error": "kept entity is missing", "merge_id": merge_id}

        await self.graph.run(
            Q.RECREATE_ENTITY,
            id=dropped["id"],
            name=dropped.get("name") or "",
            type=dropped.get("type") or "Concept",
            description=dropped.get("description") or "",
            aliases=dropped.get("aliases") or [],
        )
        await self.graph.run(Q.SET_ENTITY_ALIASES, id=kept["id"], aliases=kept.get("aliases") or [])

        kept_edges = {
            (edge.get("rel_type"), edge.get("other_id"), edge.get("direction"))
            for edge in kept.get("edges") or []
        }
        for edge in dropped.get("edges") or []:
            other = edge.get("other_id")
            rel_type = edge.get("rel_type")
            direction = edge.get("direction")
            if not other or not rel_type:
                continue
            if other == kept["id"]:
                if direction == "out":
                    await self.graph.run(
                        Q.CREATE_RELATIONSHIP_STATIC,
                        from_id=dropped["id"],
                        to_id=kept["id"],
                        type=rel_type,
                    )
                else:
                    await self.graph.run(
                        Q.CREATE_RELATIONSHIP_STATIC,
                        from_id=kept["id"],
                        to_id=dropped["id"],
                        type=rel_type,
                    )
                continue
            if (rel_type, other, direction) not in kept_edges:
                if direction == "out":
                    await self.graph.run(
                        Q.DELETE_RELATIONSHIP,
                        from_id=kept["id"],
                        to_id=other,
                        rel_type=rel_type,
                    )
                else:
                    await self.graph.run(
                        Q.DELETE_RELATIONSHIP,
                        from_id=other,
                        to_id=kept["id"],
                        rel_type=rel_type,
                    )
            if direction == "out":
                await self.graph.run(
                    Q.CREATE_RELATIONSHIP_STATIC,
                    from_id=dropped["id"],
                    to_id=other,
                    type=rel_type,
                )
            else:
                await self.graph.run(
                    Q.CREATE_RELATIONSHIP_STATIC,
                    from_id=other,
                    to_id=dropped["id"],
                    type=rel_type,
                )

        kept_episodes = set(kept.get("episode_ids") or [])
        for episode_id in dropped.get("episode_ids") or []:
            await self.graph.run(Q.LINK_EPISODE_ENTITY, episode_id=episode_id, entity_id=dropped["id"])
            if episode_id not in kept_episodes:
                await self.graph.run(
                    Q.UNLINK_EPISODE_ENTITY,
                    episode_id=episode_id,
                    entity_id=kept["id"],
                )
        if dropped.get("fact_ids"):
            await self.graph.run(
                Q.REPOINT_FACTS,
                old="__no_match__",
                new=dropped["id"],
                ids=dropped["fact_ids"],
            )
        await self.graph.run(Q.MARK_MERGE_UNDONE, id=merge_id)
        return {
            "undone": True,
            "merge_id": merge_id,
            "restored": dropped.get("name"),
            "kept": kept.get("name"),
        }

    async def repair_known_merges(self) -> dict:
        """Undo logged merges of the known-bad names, then split alias collapses."""
        undone = []
        targets = {target["name"].lower() for target in REPAIR_TARGETS}
        logs = await self.graph.run(Q.LIST_MERGE_LOGS)
        for logrow in logs:
            if logrow.get("undone"):
                continue
            try:
                state = json.loads(logrow.get("before_state") or "{}")
            except json.JSONDecodeError:
                state = {}
            dropped_name = ((state.get("dropped") or {}).get("name")) or logrow.get("dropped_name") or ""
            if dropped_name.lower() not in targets:
                continue
            result = await self.unresolve(logrow["id"])
            undone.append({
                "merge_id": logrow["id"],
                "name": dropped_name,
                "undone": bool(result.get("undone")),
            })
        restored = []
        for target in REPAIR_TARGETS:
            restored.append(await self._restore_distinct_entity(target["name"], target["type"]))
        return {"unresolved_merges": undone, "restored": restored}

    async def _restore_distinct_entity(self, name: str, type_name: str) -> dict:
        entities = await self.graph.run(Q.LIST_ALL_ENTITIES, limit=5000)
        exact = next((ent for ent in entities if (ent.get("name") or "").lower() == name.lower()), None)
        absorbed = []
        for ent in entities:
            if exact and ent.get("id") == exact.get("id"):
                continue
            aliases = [a for a in (ent.get("aliases") or []) if isinstance(a, str)]
            alias_hit = any(alias.lower() == name.lower() for alias in aliases)
            if alias_hit or is_absorbed_name(ent.get("name") or "", name):
                absorbed.append(ent)
        if exact is None:
            ent_id = new_id("ent")
            await self.graph.run(
                Q.UPSERT_ENTITY,
                id=ent_id,
                name=name,
                type=type_name,
                description="Restored as its own entity.",
                aliases=[],
            )
        else:
            ent_id = exact["id"]
        blockers = [ent.get("name") or "" for ent in absorbed if ent.get("name")]
        episodes: dict[str, str] = {}
        for ent in absorbed:
            for row in await self.graph.run(Q.LIST_ENTITY_EPISODES, id=ent["id"]):
                episodes[row["id"]] = row.get("content") or ""
            cleaned = [
                alias for alias in (ent.get("aliases") or [])
                if isinstance(alias, str) and alias.lower() != name.lower()
            ]
            await self.graph.run(Q.SET_ENTITY_ALIASES, id=ent["id"], aliases=cleaned)
        for row in await self.graph.run(Q.EPISODES_CONTAINING, phrase=name):
            episodes.setdefault(row["id"], row.get("content") or "")
        relinked: list[str] = []
        detached: list[dict] = []
        for episode_id, content in episodes.items():
            if not mentions_standalone(content, name, blockers):
                continue
            await self.graph.run(Q.LINK_EPISODE_ENTITY, episode_id=episode_id, entity_id=ent_id)
            relinked.append(episode_id)
            for ent in absorbed:
                long_name = ent.get("name") or ""
                if long_name.lower() not in (content or "").lower():
                    await self.graph.run(
                        Q.UNLINK_EPISODE_ENTITY,
                        episode_id=episode_id,
                        entity_id=ent["id"],
                    )
                    detached.append({"episode_id": episode_id, "from_entity": ent["id"]})
        return {
            "name": name,
            "entity_id": ent_id,
            "type": type_name,
            "relinked": relinked,
            "detached": detached,
        }

    async def backfill_entities(self, limit: int | None = None, use_llm: bool = True) -> dict:
        """Re-extract entities per episode and drop ones on more than 30% of episodes."""
        rows = await self.graph.run(Q.ALL_ACTIVE_EPISODES, limit=limit or 10000)
        mined: dict[str, list] = {}
        failures = 0
        for row in rows:
            content = row.get("content") or ""
            extraction = ExtractionResult()
            if use_llm:
                try:
                    # Retry lane: a long backfill must never block fresh saves.
                    async with self._retry_sem:
                        extraction = await self.llm.extract(content, agent_id="backfill")
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    log.error("Backfill extraction failed for %s: %s", row["id"], safe_error(exc, content))
            mined[row["id"]] = self._select_entities(content, extraction)
        counts: dict[str, int] = {}
        for entities in mined.values():
            for name in {ent.name for ent in entities}:
                counts[name] = counts.get(name, 0) + 1
        total = len(rows)
        skipped = sorted(name for name, count in counts.items() if too_common(count, total))
        links = 0
        sample = []
        for row in rows:
            await self.graph.run(Q.CLEAR_EPISODE_LINKS, episode_id=row["id"])
            kept: list[str] = []
            for ent in mined[row["id"]]:
                if too_common(counts.get(ent.name, 0), total):
                    continue
                ent_id, _ = await self._upsert_entity(ent)
                await self.graph.run(Q.LINK_EPISODE_ENTITY, episode_id=row["id"], entity_id=ent_id)
                kept.append(ent.name)
                links += 1
            if len(sample) < 20:
                sample.append({"episode_id": row["id"], "entities": kept})
        return {
            "episodes": total,
            "links": links,
            "skipped_common": skipped,
            "failures": failures,
            "sample": sample,
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


def _topic_card_title(row: dict) -> str:
    key = row.get("key") or ""
    if key.lower().startswith("topic-card:"):
        return "TOPIC CARD: " + key.split(":", 1)[1].strip()
    for field in ("content_head", "summary_head"):
        head = (row.get(field) or "").splitlines()[0].strip()
        if head.lower().startswith("topic card:"):
            return head[:160]
    return ""
