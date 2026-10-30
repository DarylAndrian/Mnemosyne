"""In-memory graph, embedder, and extractor for engine tests."""

from __future__ import annotations

from server.config import Settings
from server.engine import MemoryEngine
from server.graph import queries as Q
from server.models import ExtractionResult, EntityOut


class FakeLLM:
    def __init__(self, fail_times: int = 0, entities: list[EntityOut] | None = None):
        self.fail_times = fail_times
        self.entities = entities
        self.calls = 0

    async def extract(self, content, agent_id="", session_id="", tags=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("ollama timeout " + (content or ""))
        if self.entities is not None:
            return ExtractionResult(entities=list(self.entities))
        return ExtractionResult()

    async def close(self) -> None:
        return None


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2, 0.3]

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class FakeGraph:
    def __init__(self):
        self.episodes: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.facts: dict[str, dict] = {}
        self.rels: list[tuple[str, str, str]] = []
        self.rel_sources: dict[tuple[str, str, str], set[str]] = {}
        self.rel_target: dict[tuple[str, str, str], str | None] = {}
        self.mentions: set[tuple[str, str]] = set()
        self.mention_target: dict[tuple[str, str], str | None] = {}
        self.versions: list[dict] = []
        self.merges: dict[str, dict] = {}
        self.embeddings: dict[str, list] = {}
        self._clock = 0

    def _now(self) -> str:
        self._clock += 1
        minute, second = divmod(self._clock, 60)
        hour = 12 + minute // 60
        return f"2026-10-27T{hour:02d}:{minute % 60:02d}:{second % 60:02d}+00:00"

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def run_one(self, query: str, **params):
        rows = await self.run(query, **params)
        return rows[0] if rows else None

    def _episode_public(self, ep: dict, *, include_content: bool = True) -> dict:
        row = {
            "id": ep["id"],
            "status": ep.get("status") or "processed",
            "request_id": ep.get("request_id") or "",
            "error_message": ep.get("error_message") or "",
            "attempts": int(ep.get("attempts") or 0),
            "last_attempt_at": ep.get("last_attempt_at"),
            "source_agent_id": ep.get("source_agent_id") or "",
            "source_session_id": ep.get("source_session_id") or "",
            "tags": list(ep.get("tags") or []),
            "active": ep.get("active", True),
            "summary": ep.get("summary") or "",
            "version": int(ep.get("version") or 1),
            "key": ep.get("key") or "",
            "created_at": ep.get("created_at"),
            "updated_at": ep.get("updated_at"),
        }
        if include_content:
            row["content"] = ep.get("content") or ""
        return row

    def _entity_row(self, ent: dict) -> dict:
        return {
            "id": ent["id"],
            "name": ent["name"],
            "type": ent.get("type") or "Concept",
            "aliases": list(ent.get("aliases") or []),
            "description": ent.get("description") or "",
        }

    async def run(self, query: str, **params) -> list[dict]:
        if query == Q.CREATE_EPISODE:
            now = self._now()
            self.episodes[params["id"]] = {
                "id": params["id"],
                "content": params["content"],
                "summary": params["summary"],
                "tags": list(params.get("tags") or []),
                "source_agent_id": params.get("source_agent_id") or "",
                "source_session_id": params.get("source_session_id") or "",
                "request_id": params.get("request_id") or "",
                "content_hash": params.get("content_hash") or "",
                "key": params.get("key") or "",
                "version": 1,
                "status": "pending",
                "error_message": "",
                "attempts": 0,
                "last_attempt_at": None,
                "created_at": now,
                "updated_at": now,
                "active": True,
            }
            return [{"id": params["id"]}]
        if query == Q.FIND_EPISODE_BY_REQUEST_ID:
            for ep in self.episodes.values():
                if ep.get("request_id") and ep["request_id"] == params["rid"]:
                    return [{"id": ep["id"], "status": ep["status"]}]
            return []
        if query == Q.FIND_EPISODE_BY_HASH_RECENT:
            matches = [
                ep for ep in self.episodes.values()
                if ep.get("active") and ep.get("content_hash") == params["content_hash"]
            ]
            matches.sort(key=lambda ep: ep["created_at"], reverse=True)
            return [{"id": matches[0]["id"], "status": matches[0]["status"]}] if matches else []
        if query == Q.RECENT_EPISODES_IN_WINDOW:
            active = [ep for ep in self.episodes.values() if ep.get("active")]
            active.sort(key=lambda ep: ep["created_at"], reverse=True)
            return [
                {"id": ep["id"], "content": ep["content"], "status": ep["status"]}
                for ep in active[:20]
            ]
        if query == Q.FIND_EPISODE_BY_KEY:
            matches = [
                ep for ep in self.episodes.values()
                if ep.get("active") and ep.get("key") and ep["key"] == params["key"]
            ]
            matches.sort(key=lambda ep: ep["updated_at"], reverse=True)
            return [self._episode_public(matches[0])] if matches else []
        if query == Q.UPDATE_EPISODE_CONTENT:
            ep = self.episodes[params["episode_id"]]
            ep.update({
                "content": params["content"],
                "summary": params["summary"],
                "content_hash": params["content_hash"],
                "tags": list(params.get("tags") or []),
                "status": "pending",
                "error_message": "",
                "version": params["version"],
                "updated_at": self._now(),
            })
            return [{"id": ep["id"]}]
        if query == Q.INSERT_EPISODE_VERSION:
            now = self._now()
            self.versions.append({
                "id": params["id"],
                "episode_id": params["episode_id"],
                "content": params["content"],
                "summary": params.get("summary") or "",
                "version": params["version"],
                "created_at": now,
            })
            return [{"id": params["id"]}]
        if query == Q.LIST_EPISODE_VERSIONS:
            rows = [v for v in self.versions if v["episode_id"] == params["episode_id"]]
            rows.sort(key=lambda v: v["version"])
            return [
                {"version": v["version"], "content": v["content"], "created_at": v["created_at"]}
                for v in rows
            ]
        if query == Q.SET_EPISODE_EMBEDDING:
            self.embeddings[params["episode_id"]] = list(params["vector"])
            return [{"id": params["episode_id"]}]
        if query == Q.SET_ENTITY_EMBEDDING:
            self.embeddings[params["entity_id"]] = list(params["vector"])
            return [{"id": params["entity_id"]}]
        if query == Q.MARK_EPISODE_PROCESSED:
            ep = self.episodes[params["episode_id"]]
            ep["status"] = "processed"
            ep["error_message"] = ""
            ep["attempts"] = params["attempts"]
            ep["last_attempt_at"] = self._now()
            ep["updated_at"] = ep["last_attempt_at"]
            return [{"id": ep["id"]}]
        if query == Q.SET_EPISODE_FAILURE:
            ep = self.episodes[params["episode_id"]]
            ep["status"] = params["status"]
            ep["error_message"] = params["error_message"]
            ep["attempts"] = params["attempts"]
            ep["last_attempt_at"] = self._now()
            return [{"id": ep["id"], "attempts": ep["attempts"]}]
        if query == Q.GET_EPISODE_STATUS:
            ep = self.episodes.get(params["episode_id"])
            if not ep:
                return []
            return [self._episode_public(ep, include_content=False)]
        if query == Q.GET_EPISODES_BY_IDS:
            return [
                self._episode_public(self.episodes[eid])
                for eid in params["ids"]
                if eid in self.episodes
            ]
        if query == Q.LIST_FAILED_EPISODES:
            rows = [
                self._episode_public(ep)
                for ep in self.episodes.values()
                if ep.get("active") and ep.get("status") == "failed"
            ]
            return rows[: params.get("limit") or 1000]
        if query == Q.CLEAR_EPISODE_LINKS:
            removed = {pair for pair in self.mentions if pair[1] == params["episode_id"]}
            self.mentions -= removed
            for pair in removed:
                self.mention_target.pop(pair, None)
            return [{"deleted": len(removed)}]
        if query == Q.DELETE_EPISODE_FACTS:
            self.facts = {
                fid: fact for fid, fact in self.facts.items()
                if fact.get("source_episode_id") != params["episode_id"]
            }
            return [{"deleted": 0}]
        if query == Q.FIND_ENTITY_BY_NAME_CI:
            needle = (params.get("name") or "").lower()
            for ent in self.entities.values():
                if ent["name"].lower() == needle:
                    return [self._entity_row(ent)]
            return []
        if query == Q.GET_ENTITY_BY_ID:
            ent = self.entities.get(params["id"])
            return [self._entity_row(ent)] if ent else []
        if query == Q.UPSERT_ENTITY:
            self.entities[params["id"]] = {
                "id": params["id"],
                "name": params["name"],
                "type": params.get("type") or "Concept",
                "description": params.get("description") or "",
                "aliases": list(params.get("aliases") or []),
            }
            return [{"id": params["id"], "name": params["name"]}]
        if query == Q.RECREATE_ENTITY:
            self.entities[params["id"]] = {
                "id": params["id"],
                "name": params["name"],
                "type": params.get("type") or "Concept",
                "description": params.get("description") or "",
                "aliases": list(params.get("aliases") or []),
            }
            return [{"id": params["id"]}]
        if query == Q.SET_ENTITY_ALIASES:
            self.entities[params["id"]]["aliases"] = list(params.get("aliases") or [])
            return [{"id": params["id"]}]
        if query == Q.LIST_ALL_ENTITIES:
            return [self._entity_row(ent) for ent in list(self.entities.values())[: params.get("limit") or 5000]]
        if query == Q.FIND_ENTITY_FUZZY:
            needle = (params.get("needle") or "").lower()
            rows = []
            for ent in self.entities.values():
                names = [ent["name"], *(ent.get("aliases") or [])]
                if any(needle and needle in (name or "").lower() for name in names):
                    rows.append(self._entity_row(ent))
            return rows[:5]
        if query == Q.ENTITY_LINK_STATS:
            linked = sum(
                1 for ent_id, ep_id in self.mentions
                if ent_id == params["id"] and self.episodes.get(ep_id, {}).get("active", True)
            )
            total = sum(1 for ep in self.episodes.values() if ep.get("active", True))
            if params["id"] not in self.entities and linked == 0 and total == 0:
                return []
            return [{"linked": linked, "total": total}]
        if query == Q.LINK_EPISODE_ENTITY:
            pair = (params["entity_id"], params["episode_id"])
            if params["entity_id"] in self.entities and params["episode_id"] in self.episodes:
                self.mentions.add(pair)
                self.mention_target[pair] = params["episode_id"]
                ent = self.entities[params["entity_id"]]
                if not ent.get("source_episode_id"):
                    ent["source_episode_id"] = params["episode_id"]
            return [{"id": params["entity_id"], "target": params["episode_id"]}]
        if query == Q.UNLINK_EPISODE_ENTITY:
            pair = (params["entity_id"], params["episode_id"])
            self.mentions.discard(pair)
            self.mention_target.pop(pair, None)
            return [{"deleted": 1}]
        if query == Q.CREATE_RELATIONSHIP_STATIC:
            if params["from_id"] in self.entities and params["to_id"] in self.entities:
                rel = (params["from_id"], params["type"], params["to_id"])
                if rel not in self.rels:
                    self.rels.append(rel)
                if params["type"] != "MENTIONED_IN":
                    self.rel_target[rel] = params["to_id"]
            return [{"done": 1}]
        if query == Q.STAMP_RELATIONSHIP:
            rel = (params["from_id"], params["rel_type"], params["to_id"])
            if rel in self.rels and params.get("episode_id"):
                self.rel_sources.setdefault(rel, set()).add(params["episode_id"])
                self.rel_target[rel] = params["to_id"]
            return [{"stamped": 1 if rel in self.rels else 0}]
        if query == Q.DELETE_RELATIONSHIP:
            before = len(self.rels)
            self.rels = [
                rel for rel in self.rels
                if not (
                    rel[0] == params["from_id"]
                    and rel[1] == params["rel_type"]
                    and rel[2] == params["to_id"]
                )
            ]
            return [{"deleted": before - len(self.rels)}]
        if query == Q.LIST_ENTITY_EDGES:
            rows = []
            for src, rel_type, dst in self.rels:
                if src == params["id"]:
                    rows.append({"rel_type": rel_type, "other_id": dst, "direction": "out"})
                elif dst == params["id"]:
                    rows.append({"rel_type": rel_type, "other_id": src, "direction": "in"})
            return rows
        if query == Q.LIST_ENTITY_EPISODES:
            return [
                {"id": ep_id, "content": self.episodes[ep_id]["content"]}
                for ent_id, ep_id in self.mentions
                if ent_id == params["id"] and ep_id in self.episodes
            ]
        if query == Q.EPISODES_CONTAINING:
            phrase = params.get("phrase") or ""
            return [
                {"id": ep["id"], "content": ep["content"]}
                for ep in self.episodes.values()
                if ep.get("active") and phrase and phrase in (ep.get("content") or "")
            ]
        if query == Q.CREATE_FACT:
            self.facts[params["id"]] = {
                "id": params["id"],
                "content": params["content"],
                "subject": params.get("subject") or "",
                "confidence": params.get("confidence"),
                "source_agent_id": params.get("source_agent_id") or "",
                "source_session_id": params.get("source_session_id") or "",
                "source_episode_id": params.get("episode_id") or "",
                "valid_until": None,
                "created_at": self._now(),
            }
            return [{"id": params["id"]}]
        if query == Q.FIND_FACTS_ON_SUBJECT:
            return [
                {
                    "id": fact["id"],
                    "content": fact["content"],
                    "confidence": fact.get("confidence"),
                    "source_agent_id": fact.get("source_agent_id"),
                    "created_at": fact.get("created_at"),
                }
                for fact in self.facts.values()
                if fact.get("subject") == params["subject_id"] and fact.get("valid_until") is None
            ]
        if query == Q.SUPERSEDE_FACT:
            old = self.facts.get(params["old_id"])
            if old and params["new_id"] in self.facts:
                old["valid_until"] = self._now()
                old["superseded_by"] = params["new_id"]
            return []
        if query == Q.REPOINT_FACTS:
            ids = set(params.get("ids") or [])
            count = 0
            for fact in self.facts.values():
                if fact.get("subject") == params.get("old") or fact["id"] in ids:
                    fact["subject"] = params["new"]
                    count += 1
            return [{"n": count}]
        if query == Q.DELETE_ENTITY:
            ent_id = params["drop_id"]
            self.entities.pop(ent_id, None)
            self.rels = [rel for rel in self.rels if ent_id not in (rel[0], rel[2])]
            self.mentions = {pair for pair in self.mentions if pair[0] != ent_id}
            return [{"deleted": ent_id}]
        if query == Q.CREATE_MERGE_LOG:
            self.merges[params["id"]] = {
                "id": params["id"],
                "before_state": params["before_state"],
                "kept_name": params["kept_name"],
                "dropped_name": params["dropped_name"],
                "undone": False,
            }
            return [{"id": params["id"]}]
        if query == Q.GET_MERGE_LOG:
            row = self.merges.get(params["id"])
            return [dict(row)] if row else []
        if query == Q.LIST_MERGE_LOGS:
            return [dict(row) for row in self.merges.values()]
        if query == Q.MARK_MERGE_UNDONE:
            if params["id"] in self.merges:
                self.merges[params["id"]]["undone"] = True
            return [{"id": params["id"]}]
        if query == Q.EPISODE_ENTITY_NAMES:
            names: dict[str, list[str]] = {}
            for ent_id, ep_id in self.mentions:
                if ep_id in (params.get("ids") or []) and ent_id in self.entities:
                    names.setdefault(ep_id, [])
                    name = self.entities[ent_id]["name"]
                    if name not in names[ep_id]:
                        names[ep_id].append(name)
            return [{"episode_id": ep_id, "names": values} for ep_id, values in names.items()]
        if query == Q.VECTOR_SEARCH_EPISODES:
            rows = []
            for ep in self.episodes.values():
                if ep.get("active") and ep["id"] in self.embeddings:
                    rows.append({
                        "id": ep["id"],
                        "content": ep["content"],
                        "score": 0.2,
                        "source_agent_id": ep.get("source_agent_id"),
                        "source_session_id": ep.get("source_session_id"),
                        "created_at": ep.get("created_at"),
                        "tags": ep.get("tags"),
                    })
            return rows[: params.get("limit") or 10]
        if query == Q.VECTOR_SEARCH_ENTITIES:
            return []
        if query == Q.FULLTEXT_SEARCH_EPISODES:
            terms = [
                part.strip("\\ ").lower()
                for part in (params.get("q") or "").split("OR")
                if part.strip("\\ ") and part.strip() != "*"
            ]
            rows = []
            for ep in self.episodes.values():
                if not ep.get("active"):
                    continue
                hay = (ep.get("content") or "").lower()
                if terms and not any(term in hay for term in terms):
                    continue
                if not terms:
                    continue
                rows.append({
                    "id": ep["id"],
                    "content": ep["content"],
                    "score": 3.0,
                    "source_agent_id": ep.get("source_agent_id"),
                    "source_session_id": ep.get("source_session_id"),
                    "created_at": ep.get("created_at"),
                    "tags": ep.get("tags"),
                })
            rows.sort(key=lambda row: row["score"], reverse=True)
            return rows[: params.get("limit") or 10]
        if query == Q.FULLTEXT_SEARCH_FACTS:
            return []
        if query == Q.GRAPH_EXPAND:
            return []
        if query == Q.FACTS_FOR_ENTITIES:
            return []
        if query == Q.LIST_EPISODES:
            rows = [ep for ep in self.episodes.values() if ep.get("active")]
            text = (params.get("text") or "").lower()
            if text:
                rows = [
                    ep for ep in rows
                    if text in (ep.get("content") or "").lower() or text in (ep.get("summary") or "").lower()
                ]
            if params.get("source"):
                rows = [ep for ep in rows if ep.get("source_agent_id") == params["source"]]
            if params.get("status"):
                rows = [ep for ep in rows if (ep.get("status") or "processed") == params["status"]]
            if params.get("created_after"):
                rows = [ep for ep in rows if (ep.get("created_at") or "") >= params["created_after"]]
            if params.get("created_before"):
                rows = [ep for ep in rows if (ep.get("created_at") or "") <= params["created_before"]]
            if params.get("entity"):
                needle = params["entity"].lower()
                wanted = set()
                for ent in self.entities.values():
                    names = [ent["name"], *(ent.get("aliases") or [])]
                    if any((name or "").lower() == needle for name in names):
                        wanted.add(ent["id"])
                rows = [
                    ep for ep in rows
                    if any((ent_id, ep["id"]) in self.mentions for ent_id in wanted)
                ]
            if params.get("cursor_time"):
                cursor_time = params["cursor_time"]
                cursor_id = params.get("cursor_id") or ""
                rows = [
                    ep for ep in rows
                    if ep["created_at"] < cursor_time
                    or (ep["created_at"] == cursor_time and ep["id"] < cursor_id)
                ]
            rows.sort(key=lambda ep: (ep["created_at"], ep["id"]), reverse=True)
            limit = params.get("limit") or 50
            return [
                {
                    "id": ep["id"],
                    "created_at": ep["created_at"],
                    "source": ep.get("source_agent_id") or "",
                    "status": ep.get("status") or "processed",
                    "preview": (ep.get("content") or "")[:160],
                }
                for ep in rows[:limit]
            ]
        if query == Q.INDEX_EPISODE:
            matches = []
            for ep in self.episodes.values():
                if not ep.get("active"):
                    continue
                content = (ep.get("content") or "")[:5].lower()
                summary = (ep.get("summary") or "")[:5].lower()
                key = (ep.get("key") or "").lower()
                if key == "index" or content == "index" or summary == "index":
                    matches.append(ep)
            matches.sort(key=lambda ep: ep.get("updated_at") or "", reverse=True)
            if not matches:
                return []
            ep = matches[0]
            return [{
                "id": ep["id"],
                "content": ep["content"],
                "summary": ep.get("summary") or "",
                "updated_at": ep.get("updated_at"),
            }]
        if query == Q.TOPIC_CARD_EPISODES:
            rows = []
            for ep in self.episodes.values():
                if not ep.get("active"):
                    continue
                content = (ep.get("content") or "")[:11].lower()
                summary = (ep.get("summary") or "")[:11].lower()
                key = (ep.get("key") or "").lower()
                if content == "topic card:" or summary == "topic card:" or key.startswith("topic-card:"):
                    rows.append({
                        "id": ep["id"],
                        "key": ep.get("key") or "",
                        "content_head": (ep.get("content") or "")[:160],
                        "summary_head": (ep.get("summary") or "")[:160],
                    })
            rows.sort(key=lambda row: (row["key"], row["id"]))
            return rows
        if query == Q.ALL_ACTIVE_EPISODES:
            rows = [ep for ep in self.episodes.values() if ep.get("active")]
            rows.sort(key=lambda ep: ep["created_at"])
            limit = params.get("limit") or 10000
            return [{"id": ep["id"], "content": ep["content"]} for ep in rows[:limit]]
        if query == Q.SOFT_DELETE_EPISODE:
            ep = self.episodes.get(params["episode_id"])
            if not ep:
                return []
            ep["active"] = False
            return [{"id": ep["id"]}]
        handled = self._run_connectivity(query, params)
        if handled is not None:
            return handled
        if query == Q.ENTITY_NEIGHBORHOOD:
            return []
        raise AssertionError("unexpected query: " + " ".join(query.split())[:80])

    def _degree(self, node_id: str) -> int:
        degree = sum(1 for ent_id, ep_id in self.mentions if node_id in (ent_id, ep_id))
        degree += sum(1 for src, _rel, dst in self.rels if node_id in (src, dst))
        return degree

    def _active_mentions(self, entity_id: str) -> set[str]:
        return {
            ep_id
            for ent_id, ep_id in self.mentions
            if ent_id == entity_id and self.episodes.get(ep_id, {}).get("active", True)
        }

    def _orphan_rows(self, max_degree: int, include_inactive: bool = False) -> list[dict]:
        rows = []
        nodes: list[tuple[str, str]] = [("entity", ent_id) for ent_id in self.entities]
        nodes += [("episode", ep_id) for ep_id in self.episodes]
        for kind, node_id in nodes:
            if kind == "episode" and not include_inactive:
                if self.episodes[node_id].get("active") is False:
                    continue
            if kind == "entity" and not include_inactive:
                # Hide entities whose every mention points at a forgotten
                # episode; keep entities with no episode at all.
                mentioned = [ep_id for ent_id, ep_id in self.mentions if ent_id == node_id]
                if mentioned and not any(
                    self.episodes.get(e, {}).get("active") is not False for e in mentioned
                ):
                    continue
            degree = self._degree(node_id)
            if degree > max_degree:
                continue
            if kind == "entity":
                ent = self.entities[node_id]
                mentioned_eps = [ep_id for ent_id, ep_id in self.mentions if ent_id == node_id]
                rows.append({
                    "id": node_id,
                    "name": ent.get("name") or node_id,
                    "type": ent.get("type") or "Entity",
                    "source_episode_id": ent.get("source_episode_id") or (mentioned_eps[0] if mentioned_eps else ""),
                    "degree": degree,
                })
            else:
                ep = self.episodes[node_id]
                rows.append({
                    "id": node_id,
                    "name": ep.get("key") or (ep.get("summary") or ep.get("content") or node_id)[:80],
                    "type": "Episode",
                    "source_episode_id": node_id,
                    "degree": degree,
                })
        rows.sort(key=lambda row: (row["name"], row["id"]))
        return rows

    def _run_connectivity(self, query: str, params: dict) -> list[dict] | None:
        if query == Q.GET_EPISODE_BRIEF:
            ep = self.episodes.get(params["episode_id"])
            if not ep:
                return []
            return [{
                "id": ep["id"],
                "key": ep.get("key") or "",
                "content_head": (ep.get("content") or "")[:80],
                "active": ep.get("active", True),
            }]
        if query == Q.LIST_PENDING_IDS:
            ids = [
                {"id": ep["id"]}
                for ep in self.episodes.values()
                if ep.get("active", True) and (ep.get("status") or "") == "pending"
            ]
            return ids[: params.get("limit") or 500]
        if query == Q.INVALIDATE_EPISODE_FACTS:
            n = 0
            for fact in self.facts.values():
                if fact.get("source_episode_id") == params["episode_id"] and fact.get("valid_until") is None:
                    fact["valid_until"] = self._now()
                    fact["superseded_by"] = "forgotten"
                    n += 1
            return [{"invalidated": n}]
        if query == Q.ENTITY_IDS_ONLY_ON_EPISODE:
            episode_id = params["episode_id"]
            rows = []
            for ent_id in self.entities:
                eps = [ep_id for linked, ep_id in self.mentions if linked == ent_id]
                if episode_id not in eps:
                    continue
                other = [
                    ep_id for ep_id in eps
                    if ep_id != episode_id and self.episodes.get(ep_id, {}).get("active", True)
                ]
                if not other:
                    rows.append({"id": ent_id})
            return rows
        if query == Q.DROP_PROVENANCE_EDGES:
            episode_id = params["episode_id"]
            kept = []
            deleted = 0
            for rel in self.rels:
                sources = set(self.rel_sources.get(rel) or ())
                if episode_id not in sources:
                    kept.append(rel)
                    continue
                sources.discard(episode_id)
                alive = {
                    ep_id for ep_id in sources
                    if self.episodes.get(ep_id, {}).get("active", True)
                }
                if alive:
                    self.rel_sources[rel] = alive
                    kept.append(rel)
                else:
                    self.rel_sources.pop(rel, None)
                    self.rel_target.pop(rel, None)
                    deleted += 1
            self.rels = kept
            return [{"deleted": deleted}]
        if query == Q.PURGE_STALE_EDGES:
            kept = []
            deleted = 0
            for rel in self.rels:
                src, rel_type, dst = rel
                if rel_type == "MENTIONED_IN" or self.rel_sources.get(rel):
                    kept.append(rel)
                    continue
                if self._active_mentions(src) and self._active_mentions(dst):
                    kept.append(rel)
                    continue
                self.rel_target.pop(rel, None)
                deleted += 1
            self.rels = kept
            return [{"deleted": deleted}]
        if query == Q.DELETE_ISOLATED_ENTITIES:
            deleted = 0
            for ent_id in params.get("ids") or []:
                if ent_id in self.entities and self._degree(ent_id) == 0:
                    self.entities.pop(ent_id, None)
                    deleted += 1
            return [{"deleted": deleted}]
        if query == Q.DELETE_EDGES_BETWEEN_NAMES:
            left = (params.get("left") or "").lower()
            right = (params.get("right") or "").lower()
            names = {ent["id"]: (ent.get("name") or "").lower() for ent in self.entities.values()}
            kept = []
            deleted = 0
            for rel in self.rels:
                src_name = names.get(rel[0], "")
                dst_name = names.get(rel[2], "")
                pair = {src_name, dst_name}
                if pair == {left, right}:
                    self.rel_sources.pop(rel, None)
                    self.rel_target.pop(rel, None)
                    deleted += 1
                else:
                    kept.append(rel)
            self.rels = kept
            return [{"deleted": deleted}]
        if query == Q.DELETE_INACTIVE_MENTIONS_BY_NAME:
            name = (params.get("name") or "").lower()
            removed = []
            for ent_id, ep_id in list(self.mentions):
                ent = self.entities.get(ent_id) or {}
                ep = self.episodes.get(ep_id) or {}
                if (ent.get("name") or "").lower() == name and not ep.get("active", True):
                    removed.append((ent_id, ep_id))
            for pair in removed:
                self.mentions.discard(pair)
                self.mention_target.pop(pair, None)
            return [{"deleted": len(removed)}]
        if query == Q.DELETE_ISOLATED_ENTITY_BY_NAME:
            name = (params.get("name") or "").lower()
            deleted = 0
            for ent_id, ent in list(self.entities.items()):
                if (ent.get("name") or "").lower() == name and self._degree(ent_id) == 0:
                    self.entities.pop(ent_id, None)
                    deleted += 1
            return [{"deleted": deleted}]
        if query == Q.BACKFILL_MENTIONED_TARGETS:
            repaired = 0
            for pair in list(self.mentions):
                ent_id, ep_id = pair
                if ep_id not in self.episodes:
                    continue
                if self.mention_target.get(pair) != ep_id:
                    self.mention_target[pair] = ep_id
                    repaired += 1
                ent = self.entities.get(ent_id)
                if ent is not None and not ent.get("source_episode_id"):
                    ent["source_episode_id"] = ep_id
            return [{"repaired": repaired}]
        if query == Q.LIST_DANGLING_MENTIONED:
            rows = []
            for pair in self.mentions:
                ent_id, ep_id = pair
                target = self.mention_target.get(pair)
                if target == ep_id and ep_id:
                    continue
                ent = self.entities.get(ent_id) or {}
                rows.append({
                    "from_id": ent_id,
                    "to_id": ep_id or "",
                    "to_is_episode": ep_id in self.episodes,
                    "source_episode_id": ent.get("source_episode_id") or "",
                })
            for src, rel_type, dst in self.rels:
                if rel_type != "MENTIONED_IN":
                    continue
                ent = self.entities.get(src) or {}
                rows.append({
                    "from_id": src,
                    "to_id": dst or "",
                    "to_is_episode": False,
                    "source_episode_id": ent.get("source_episode_id") or "",
                })
            return rows[:5000]
        if query == Q.CLEAR_BROKEN_MENTION:
            from_id = params["from_id"]
            removed = [
                pair for pair in self.mentions
                if pair[0] == from_id and self.mention_target.get(pair) != pair[1]
            ]
            for pair in removed:
                self.mentions.discard(pair)
                self.mention_target.pop(pair, None)
            kept = []
            for rel in self.rels:
                if rel[0] == from_id and rel[1] == "MENTIONED_IN":
                    self.rel_sources.pop(rel, None)
                    self.rel_target.pop(rel, None)
                else:
                    kept.append(rel)
            self.rels = kept
            return [{"deleted": len(removed)}]
        if query == Q.COUNT_NULL_MENTION_TARGETS:
            n = sum(1 for pair in self.mentions if self.mention_target.get(pair) in (None, ""))
            n += sum(1 for rel in self.rels if rel[1] == "MENTIONED_IN" and not self.rel_target.get(rel))
            return [{"n": n}]
        if query == Q.LIST_ORPHANS:
            rows = self._orphan_rows(
                int(params.get("max_degree") or 0),
                bool(params.get("include_inactive", False)),
            )
            return rows[: params.get("limit") or 5000]
        if query == Q.COUNT_ORPHANS:
            return [{
                "n": len(self._orphan_rows(
                    int(params.get("max_degree") or 0),
                    bool(params.get("include_inactive", False)),
                ))
            }]
        if query == Q.LIST_UNLINKED_OWNED_ENTITIES:
            episode_id = params.get("episode_id") or ""
            return [
                {"id": ent_id, "name": (self.entities[ent_id].get("name") or "")}
                for ent_id in self.entities
                if (self.entities[ent_id].get("source_episode_id") or "") == episode_id
                and (ent_id, episode_id) not in self.mentions
            ]
        return None


async def make_engine(llm: FakeLLM | None = None) -> tuple[MemoryEngine, FakeGraph, FakeLLM]:
    engine = MemoryEngine(Settings())
    await engine.llm.close()
    await engine.embedder.close()
    await engine.graph.close()
    graph = FakeGraph()
    extractor = llm or FakeLLM()
    engine.graph = graph
    engine.llm = extractor
    engine.embedder = FakeEmbedder()
    engine._retry_backoff = (0, 0, 0)
    engine.sleeps = []

    async def _sleep(delay: float) -> None:
        engine.sleeps.append(delay)

    engine._sleep = _sleep
    return engine, graph, extractor


async def settle(engine: MemoryEngine) -> None:
    while engine._bg_tasks:
        await __import__("asyncio").gather(*list(engine._bg_tasks))
