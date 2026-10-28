"""End-to-end tests for the retry/reprocess path against a REAL Neo4j.

Skipped automatically when Neo4j is not reachable on localhost:7687.

Why these exist: the2026-10-28 jam was invisible to the fake-graph tests.
`all` is a Cypher reserved word, so ENTITY_LINK_STATS
(``RETURN linked, count(all) AS total``) raised CypherSyntaxError on every
episode that had at least one entity linked -- but the fake graph never ran
the real Cypher, so every test stayed green while production retried forever.

These tests run the real queries, the real status transitions, and the real
retry/reprocess plumbing (fake LLM + embedder only, so no Ollama needed).
"""

from __future__ import annotations

import asyncio
import socket
import uuid

import pytest

from server.config import Settings
from server.engine import MemoryEngine
from server.graph import queries as Q
from server.graph.driver import GraphDriver
from tests.fakes import FakeEmbedder, FakeLLM

CONTENT = "Integration test episode for the retry lane repair."


def _neo4j_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 7687), timeout=2):
            return True
    except OSError:
        return False


NEO4J_OK = _neo4j_reachable()

pytestmark = pytest.mark.skipif(
    not NEO4J_OK, reason="Neo4j not reachable on localhost:7687"
)


def _mk_engine() -> tuple[MemoryEngine, GraphDriver]:
    engine = MemoryEngine(Settings())
    engine.llm = FakeLLM()
    engine.embedder = FakeEmbedder()

    async def _no_sleep(_delay: float) -> None:  # keep tests fast
        return None

    engine._sleep = _no_sleep
    return engine, engine.graph


async def _seed_episode(graph: GraphDriver, status: str = "pending") -> str:
    ep_id = f"ep_test_{uuid.uuid4().hex[:12]}"
    await graph.run(
        Q.CREATE_EPISODE,
        id=ep_id,
        content=CONTENT,
        summary=CONTENT[:60],
        tags=["test"],
        source_agent_id="integration-test",
        source_session_id="",
        request_id=f"test-{ep_id}",
        content_hash=f"hash-{ep_id}",
        key="",
    )
    if status == "failed":
        await graph.run(
            Q.SET_EPISODE_FAILURE,
            episode_id=ep_id,
            status="failed",
            error_message="seeded failure (integration test)",
            attempts=3,
        )
    return ep_id


async def _cleanup(graph: GraphDriver, ids: list[str]) -> None:
    await graph.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n", ids=ids)


# ---------------------------------------------------------------------------
# 1. The actual bug: ENTITY_LINK_STATS must execute on a real server
# ---------------------------------------------------------------------------


async def test_entity_link_stats_query_executes():
    """Regression for count(all): `all` is reserved, Neo4j rejected the query."""
    engine, graph = _mk_engine()
    eid = f"ent_test_{uuid.uuid4().hex[:12]}"
    ep_id = f"ep_test_{uuid.uuid4().hex[:12]}"
    try:
        await graph.run(
            "CREATE (e:Entity {id: $id, name: $name, type: $type, aliases: [], "
            "description: '', created_at: datetime(), updated_at: datetime()})",
            id=eid, name=f"Test Entity {eid}", type="concept",
        )
        await graph.run(
            Q.CREATE_EPISODE,
            id=ep_id, content=CONTENT, summary=CONTENT[:60], tags=[],
            source_agent_id="integration-test", source_session_id="",
            request_id=f"test-{ep_id}", content_hash=f"hash-{ep_id}", key="",
        )
        await graph.run(
            "MATCH (e:Entity {id:$eid}), (ep:Episode {id:$ep}) "
            "CREATE (e)-[:MENTIONED_IN]->(ep)",
            eid=eid, ep=ep_id,
        )

        row = await graph.run_one(Q.ENTITY_LINK_STATS, id=eid)

        assert row is not None, "ENTITY_LINK_STATS returned no row"
        assert int(row["linked"]) == 1, f"expected linked=1, got {row}"
        assert int(row["total"]) >= 1, f"expected total>=1, got {row}"

        # The engine path that actually broke in production.
        assert await engine._is_overly_common(eid) in (True, False)
    finally:
        await _cleanup(graph, [eid, ep_id])
        await engine.close()


# ---------------------------------------------------------------------------
# 2. reprocess_failed must reset status, then actually process
# ---------------------------------------------------------------------------


async def test_reprocess_failed_resets_then_processes():
    engine, graph = _mk_engine()
    ep_id = await _seed_episode(graph, status="failed")
    try:
        before = await engine.episode_status(ep_id)
        assert before["status"] == "failed"
        assert before["attempts"] == 3
        assert before["error_message"]

        result = await engine.reprocess_failed([ep_id], wait=True)

        assert ep_id in result["queued"], result
        assert ep_id in result["reset"], f"episode was not reset: {result}"

        after = await engine.episode_status(ep_id)
        assert after["status"] == "processed", after
        assert after["error_message"] == "", after
        assert after["attempts"] >= 1, after
    finally:
        await _cleanup(graph, [ep_id])
        await engine.close()


async def test_reprocess_failed_marks_pending_immediately():
    """Requeued rows must read pending/attempts=0 straight away, not stay failed."""
    engine, graph = _mk_engine()
    ep_id = await _seed_episode(graph, status="failed")
    try:
        await engine.reprocess_failed([ep_id], wait=False)
        st = await engine.episode_status(ep_id)
        assert st["status"] in ("pending", "processed"), st
        assert st["attempts"] == 0 or st["status"] == "processed", st
        if st["status"] == "pending":
            assert st["error_message"] == "", st
    finally:
        await _cleanup(graph, [ep_id])
        await engine.close()


# ---------------------------------------------------------------------------
# 3. Retries must not starve fresh episodes (the queue jam)
# ---------------------------------------------------------------------------


async def test_retry_lane_runs_while_fresh_lane_is_busy():
    """lane='retry' must not wait behind a stuck fresh-lane episode."""
    engine, graph = _mk_engine()
    ep_id = await _seed_episode(graph)
    await engine._process_sem.acquire()  # simulate the jam: fresh lane held
    try:
        await asyncio.wait_for(
            engine._process_episode(ep_id, CONTENT, "integration-test", "", [], lane="retry"),
            timeout=30,
        )
    finally:
        engine._process_sem.release()

    st = await engine.episode_status(ep_id)
    assert st["status"] == "processed", st


async def test_fresh_lane_is_actually_blocked():
    """Control for the test above: lane='new' does share the fresh semaphore."""
    engine, graph = _mk_engine()
    ep_id = await _seed_episode(graph)
    await engine._process_sem.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                engine._process_episode(ep_id, CONTENT, "integration-test", "", [], lane="new"),
                timeout=2,
            )
    finally:
        engine._process_sem.release()
        await _cleanup(graph, [ep_id])
        await engine.close()


async def test_failing_fresh_episode_hands_off_to_retry_lane():
    """Attempt 1 uses the fresh lane; a failure moves attempts 2+ to the retry lane.

    Observed directly: while _extract_and_link runs we record which semaphore is
    held. (A version that simply held the fresh lane across both attempts would
    deadlock attempt 1, which legitimately needs that lane.)
    """
    engine, graph = _mk_engine()
    engine.llm = FakeLLM(fail_times=1)  # attempt 1 fails, attempt 2 succeeds
    ep_id = await _seed_episode(graph)

    observed: list[tuple[bool, bool]] = []
    original = engine._extract_and_link

    async def spy(eid, content, agent_id, session_id, tags):
        observed.append((engine._process_sem.locked(), engine._retry_sem.locked()))
        return await original(eid, content, agent_id, session_id, tags)

    engine._extract_and_link = spy

    await engine._process_episode(ep_id, CONTENT, "integration-test", "", [], lane="new")

    st = await engine.episode_status(ep_id)
    assert st["status"] == "processed", st
    assert st["attempts"] == 2, st
    assert st["error_message"] == "", st

    assert len(observed) == 2, f"expected 2 attempts, observed {observed}"
    fresh_held, retry_held = observed[0]
    assert fresh_held and not retry_held, f"attempt 1 must use fresh lane: {observed}"
    fresh_held, retry_held = observed[1]
    assert retry_held and not fresh_held, f"attempt 2 must use retry lane: {observed}"

    await _cleanup(graph, [ep_id])
    await engine.close()


# ---------------------------------------------------------------------------
# 4. resolve guard + unresolve round-trip (throwaway entities)
# ---------------------------------------------------------------------------


async def _mk_entity(graph: GraphDriver, name: str, type_: str = "Concept") -> str:
    eid = f"ent_test_{uuid.uuid4().hex[:12]}"
    await graph.run(
        "CREATE (e:Entity {id:$id, name:$name, type:$type, aliases:[], description:'', "
        "created_at:datetime(), updated_at:datetime()})",
        id=eid, name=name, type=type_,
    )
    return eid


async def test_resolve_guard_refuses_unrelated_names():
    """dry_run reported would_merge=true for clearly different names before."""
    engine, graph = _mk_engine()
    a_name = f"Slingshot Results Hub {uuid.uuid4().hex[:6]}"
    b_name = f"Renaldi Tjahaya {uuid.uuid4().hex[:6]}"
    a_id = await _mk_entity(graph, a_name)
    b_id = await _mk_entity(graph, b_name)
    try:
        dry = await engine.resolve(a_name, b_name, dry_run=True)
        assert dry.get("would_merge") is False, dry
        assert "unrelated" in (dry.get("warning") or dry.get("error") or ""), dry

        real = await engine.resolve(a_name, b_name)
        assert real.get("merged") is False, real
        assert "unrelated" in (real.get("error") or ""), real

        # both entities must still exist after the refusals
        assert await graph.run_one(Q.GET_ENTITY_BY_ID, id=a_id)
        assert await graph.run_one(Q.GET_ENTITY_BY_ID, id=b_id)

        # force=true overrides
        forced = await engine.resolve(a_name, b_name, force=True)
        assert forced.get("merged") is True, forced
    finally:
        await _cleanup(graph, [a_id, b_id])
        await engine.close()


async def test_resolve_then_unresolve_roundtrip():
    """unresolve must work after a REAL resolve (task: never tested)."""
    engine, graph = _mk_engine()
    suffix = uuid.uuid4().hex[:6]
    a_name = f"Integration Test Harbor {suffix}"
    b_name = f"Integration Test Harbor Old {suffix}"
    c_name = f"Integration Test Lighthouse {suffix}"
    a_id = await _mk_entity(graph, a_name)
    b_id = await _mk_entity(graph, b_name)
    c_id = await _mk_entity(graph, c_name)
    ep_id = await _seed_episode(graph)
    try:
        await graph.run(Q.CREATE_RELATIONSHIP_STATIC, from_id=b_id, to_id=c_id, type="KNOWS")
        await graph.run(Q.LINK_EPISODE_ENTITY, episode_id=ep_id, entity_id=b_id)

        res = await engine.resolve(a_name, b_name)
        assert res.get("merged") is True, res
        merge_id = res["merge_id"]
        assert res.get("kept") == a_name and res.get("dropped") == b_name, res

        assert await graph.run_one(Q.GET_ENTITY_BY_ID, id=b_id) is None, "B should be deleted"
        names_row = await graph.run_one(Q.EPISODE_ENTITY_NAMES, ids=[ep_id])
        names = (names_row or {}).get("names") or []
        assert a_name in names, f"episode not re-pointed to kept entity: {names}"
        assert b_name not in names, f"dropped entity still linked: {names}"

        undone = await engine.unresolve(merge_id)
        assert undone.get("undone") is True, undone

        restored = await graph.run_one(Q.GET_ENTITY_BY_ID, id=b_id)
        assert restored is not None, "unresolve must recreate the dropped entity"
        assert restored["name"] == b_name, restored

        names_row = await graph.run_one(Q.EPISODE_ENTITY_NAMES, ids=[ep_id])
        names = (names_row or {}).get("names") or []
        assert b_name in names, f"episode mention not restored: {names}"

        a_row = await graph.run_one(Q.GET_ENTITY_BY_ID, id=a_id)
        assert b_name not in (a_row.get("aliases") or []), "kept aliases not restored"

        # a second unresolve must refuse, not corrupt state
        again = await engine.unresolve(merge_id)
        assert again.get("undone") is False, again
    finally:
        await _cleanup(graph, [a_id, b_id, c_id, ep_id])
        await engine.close()
