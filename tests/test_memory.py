"""Behavior tests for processing, recall, resolve, entities, and secrets."""

from __future__ import annotations

import json

from server.entity_rules import assign_entities, is_generic, mentions_standalone
from server.graph import queries as Q
from server.models import EntityOut
from server.secrets import find_secret
from server.textutil import estimate_tokens, snippet_around

from tests.fakes import FakeLLM, make_engine, settle

FAKE_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
)


def test_alias_updates_concatenate_lists():
    assert "+ [$alias]" in Q.ADD_ALIAS
    assert "e.aliases + $alias" not in Q.ADD_ALIAS.replace("+ [$alias]", "")
    assert "+ [b.name]" in Q.REPARENT_ENTITY
    assert "a.aliases + b.name" not in Q.REPARENT_ENTITY.replace("+ [b.name]", "")


def test_secret_patterns_and_plain_text():
    assert find_secret("password: hunter2-example")
    assert find_secret("the token is not the problem, token: abcd.1234-zzzz")
    assert find_secret(FAKE_JWT)
    assert not find_secret("We agreed a password policy for the new board.")
    assert not find_secret("Project: Widget9 shipped the board notes.")


def test_snippet_is_short_and_centered():
    content = ("alpha " * 80) + "purple zebra signal " + ("beta " * 80)
    excerpt = snippet_around(content, "purple zebra", 200)
    assert "purple" in excerpt.lower()
    assert len(excerpt) <= 200


def test_standalone_mentions_ignore_longer_names():
    long_name = "HOKI Rewards Mobile 13112611678"
    assert mentions_standalone("working on HOKI Rewards launch", "HOKI Rewards", [long_name])
    assert not mentions_standalone(long_name + " login", "HOKI Rewards", [long_name])
    assert mentions_standalone("VelaTavern deploy failed", "VelaTavern", ["Vela Tavern v3"])
    assert not mentions_standalone("Vela Tavern v3 release", "VelaTavern", ["Vela Tavern v3"])


def test_twenty_episodes_get_specific_entities():
    episodes = [
        (f"ep{i}", f"Project: Widget{i}\nBoard: SharedBoard\nnotes {i}")
        for i in range(20)
    ]
    assigned = assign_entities(episodes)
    assert is_generic("Memory")
    assert not is_generic("Widget3")
    for i in range(20):
        assert assigned[f"ep{i}"] == [f"Widget{i}"]
        assert "SharedBoard" not in assigned[f"ep{i}"]


async def test_processing_succeeds_and_is_recallable():
    engine, graph, _llm = await make_engine()
    saved = await engine.remember(
        "Project: PurpleZebra\nThe purple zebra mnemonic is ready for recall.",
        agent_id="hermes-host",
    )
    assert saved.status == "pending"
    assert saved.episode_id.startswith("ep_")
    await settle(engine)
    status = await engine.episode_status(saved.episode_id)
    assert status["status"] == "processed"
    assert status["error_message"] == ""
    assert status["attempts"] == 1
    assert status["last_attempt_at"]
    found = await engine.recall("purple zebra mnemonic", top_k=5)
    ids = [hit["episode_id"] for hit in found["results"]]
    assert saved.episode_id in ids
    hit = next(item for item in found["results"] if item["episode_id"] == saved.episode_id)
    assert hit["source"] == "hermes-host"
    assert hit["created_at"]
    assert hit["score"] > 0
    forgotten = await engine.forget(hit["episode_id"])
    assert forgotten["forgotten"] is True
    assert saved.episode_id in graph.episodes
    assert graph.episodes[saved.episode_id]["active"] is False
    again = await engine.recall("purple zebra mnemonic", top_k=5)
    assert saved.episode_id not in [item.get("episode_id") for item in again["results"]]
    await engine.close()


async def test_processing_retries_then_records_failure():
    flaky = FakeLLM(fail_times=2)
    engine, _graph, llm = await make_engine(flaky)
    body = "Project: RetryCase\nunique-retry-body-alpha"
    saved = await engine.remember(body, agent_id="tester")
    await settle(engine)
    status = await engine.episode_status(saved.episode_id)
    assert status["status"] == "processed"
    assert status["attempts"] == 3
    assert status["error_message"] == ""
    assert engine.sleeps == [0, 0]
    assert llm.calls == 3

    flaky.fail_times = 99
    failed = await engine.remember("Project: GiveUp\nunique-give-up-body-beta", agent_id="tester")
    await settle(engine)
    failed_status = await engine.episode_status(failed.episode_id)
    assert failed_status["status"] == "failed"
    assert failed_status["attempts"] == 3
    assert failed_status["last_attempt_at"]
    assert "timeout" in failed_status["error_message"]
    assert "unique-give-up-body-beta" not in failed_status["error_message"]
    assert body not in failed_status["error_message"]

    llm.fail_times = llm.calls
    repaired = await engine.reprocess_failed(episode_ids=[failed.episode_id], wait=True)
    assert repaired["queued"] == [failed.episode_id]
    assert repaired["results"][0]["status"] == "processed"
    await engine.close()


async def test_recall_entities_are_per_episode_and_snippet_is_smaller():
    llm = FakeLLM(entities=[
        EntityOut(name="Memory", type="Concept"),
        EntityOut(name="SameFive", type="Concept"),
        EntityOut(name="AlphaBoard", type="Board"),
    ])
    engine, _graph, _llm = await make_engine(llm)
    long_tail = " padding" * 400
    first = await engine.remember(
        "Project: AlphaBoard\n" + ("alpha " * 40) + "purple zebra signal" + long_tail,
        agent_id="desk",
    )
    second = await engine.remember(
        "Project: BetaBoard\nbeta board notes without the query phrase",
        agent_id="laptop",
    )
    await settle(engine)
    full = await engine.recall("purple zebra signal", top_k=5, mode="full")
    snippet = await engine.recall("purple zebra signal", top_k=5, mode="snippet")
    full_hit = next(hit for hit in full["results"] if hit["episode_id"] == first.episode_id)
    assert full_hit["entities"] == ["AlphaBoard"]
    assert "SameFive" not in full_hit["entities"]
    assert "Memory" not in full_hit["entities"]
    second_hit = next(hit for hit in full["results"] if hit.get("episode_id") == second.episode_id)
    assert second_hit["entities"] == ["BetaBoard"]
    assert full_hit["entities"] != second_hit["entities"]
    assert set(snippet["results"][0]) == {"id", "episode_id", "score", "excerpt"}
    assert snippet["results"][0]["id"] == first.episode_id
    assert len(snippet["results"][0]["excerpt"]) <= 200
    assert len(json.dumps(snippet)) <= 0.3 * len(json.dumps(full))
    await engine.close()


async def test_list_filters_and_cursor():
    engine, _graph, _llm = await make_engine()
    keep = []
    for text, source in (
        ("status check: nothing new on the desk", "host"),
        ("status check: no change since yesterday", "host"),
        ("Project: GammaWing\nshipped the gamma wing", "laptop"),
    ):
        saved = await engine.remember(text, agent_id=source)
        keep.append(saved.episode_id)
    await settle(engine)
    nothing = await engine.list_episodes(text_contains="nothing new")
    assert [row["id"] for row in nothing["episodes"]] == [keep[0]]
    assert len(nothing["episodes"][0]["preview"]) <= 160
    no_change = await engine.list_episodes(text_contains="no change")
    assert [row["id"] for row in no_change["episodes"]] == [keep[1]]

    page = await engine.list_episodes(limit=2)
    assert len(page["episodes"]) == 2
    assert page["next_cursor"]
    rest = await engine.list_episodes(limit=2, cursor=page["next_cursor"])
    ids = {row["id"] for row in page["episodes"]} | {row["id"] for row in rest["episodes"]}
    assert keep[0] in ids and keep[1] in ids and keep[2] in ids
    assert not ({row["id"] for row in page["episodes"]} & {row["id"] for row in rest["episodes"]})
    forgotten = await engine.forget(nothing["episodes"][0]["id"])
    assert forgotten["forgotten"] is True
    assert (await engine.list_episodes(text_contains="nothing new"))["episodes"] == []
    laptop = await engine.list_episodes(source="laptop")
    assert [row["id"] for row in laptop["episodes"]] == [keep[2]]
    assert laptop["episodes"][0]["source"] == "laptop"
    bad = await engine.list_episodes(cursor="not-a-cursor")
    assert bad["error"] == "invalid cursor"
    await engine.close()


async def test_merge_then_unresolve_restores_state():
    engine, graph, _llm = await make_engine()
    await graph.run(
        Q.UPSERT_ENTITY, id="ent_a", name="Alpha", type="Project", description="kept", aliases=[],
    )
    await graph.run(
        Q.UPSERT_ENTITY, id="ent_b", name="Beta", type="Product", description="dropped", aliases=["Bee"],
    )
    await graph.run(
        Q.UPSERT_ENTITY, id="ent_c", name="Gamma", type="Tool", description="", aliases=[],
    )
    await graph.run(Q.CREATE_RELATIONSHIP_STATIC, from_id="ent_b", to_id="ent_c", type="USES")
    for eid, content in (("ep_b", "Beta weekly ship"), ("ep_a", "Alpha stays")):
        await graph.run(
            Q.CREATE_EPISODE,
            id=eid, content=content, summary=content, tags=[], source_agent_id="t",
            source_session_id="", request_id="", content_hash=eid, key="",
        )
    await graph.run(Q.LINK_EPISODE_ENTITY, entity_id="ent_b", episode_id="ep_b")
    await graph.run(Q.LINK_EPISODE_ENTITY, entity_id="ent_a", episode_id="ep_a")
    await graph.run(
        Q.CREATE_FACT,
        id="fact_b", content="Beta ships weekly", subject="ent_b", confidence=0.9,
        source_agent_id="t", source_session_id="", episode_id="ep_b",
    )

    dry = await engine.resolve("Alpha", "Beta", dry_run=True)
    assert dry["dry_run"] is True and dry["would_merge"] is True
    assert dry["merge_preview"]["episodes"] == ["ep_b"]
    assert "ent_b" in graph.entities
    assert graph.merges == {}

    merged = await engine.resolve("HOKI Rewards", "Alpha")
    assert merged["merged"] is False

    merged = await engine.resolve("Alpha", "Beta")
    assert merged["merged"] is True
    assert merged["merge_id"].startswith("merge_")
    assert "ent_b" not in graph.entities
    assert "Beta" in graph.entities["ent_a"]["aliases"]
    assert ("ent_a", "USES", "ent_c") in graph.rels
    assert ("ent_a", "ep_b") in graph.mentions
    assert graph.facts["fact_b"]["subject"] == "ent_a"

    undone = await engine.unresolve(merged["merge_id"])
    assert undone["undone"] is True
    assert graph.entities["ent_b"]["name"] == "Beta"
    assert graph.entities["ent_b"]["aliases"] == ["Bee"]
    assert graph.entities["ent_b"]["description"] == "dropped"
    assert graph.entities["ent_a"]["aliases"] == []
    assert ("ent_b", "USES", "ent_c") in graph.rels
    assert ("ent_a", "USES", "ent_c") not in graph.rels
    assert ("ent_b", "ep_b") in graph.mentions
    assert ("ent_a", "ep_b") not in graph.mentions
    assert ("ent_a", "ep_a") in graph.mentions
    assert graph.facts["fact_b"]["subject"] == "ent_b"
    assert graph.merges[merged["merge_id"]]["undone"] is True
    await engine.close()


async def test_fuzzy_merge_is_opt_in_and_known_collapses_split():
    engine, graph, _llm = await make_engine()
    await graph.run(
        Q.UPSERT_ENTITY, id="ent_m", name="Mnemosyne", type="Project", description="", aliases=[],
    )
    await graph.run(
        Q.UPSERT_ENTITY, id="ent_o", name="OtherExact", type="Project", description="", aliases=[],
    )
    exact = await engine.resolve("Mnemosyn", "OtherExact", fuzzy=False, dry_run=True)
    assert exact["merged"] is False
    fuzzy = await engine.resolve("Mnemosyn", "OtherExact", fuzzy=True, dry_run=True)
    assert fuzzy["would_merge"] is True
    assert fuzzy["merge_preview"]["kept"] == "Mnemosyne"
    assert "ent_o" in graph.entities

    await graph.run(
        Q.UPSERT_ENTITY,
        id="ent_long",
        name="HOKI Rewards Mobile 13112611678",
        type="Product",
        description="",
        aliases=["HOKI Rewards"],
    )
    await graph.run(
        Q.UPSERT_ENTITY,
        id="ent_vela",
        name="Vela Tavern v3",
        type="Project",
        description="",
        aliases=["VelaTavern"],
    )
    refused = await engine.resolve("HOKI Rewards", "HOKI Rewards Mobile 13112611678")
    assert refused["merged"] is False
    episodes = {
        "ep_short": "working on HOKI Rewards launch",
        "ep_long": "HOKI Rewards Mobile 13112611678 login issue",
        "ep_vt": "VelaTavern deploy failed",
        "ep_v3": "Vela Tavern v3 release",
    }
    for eid, content in episodes.items():
        await graph.run(
            Q.CREATE_EPISODE,
            id=eid, content=content, summary=content, tags=[], source_agent_id="t",
            source_session_id="", request_id="", content_hash=eid, key="",
        )
    await graph.run(Q.LINK_EPISODE_ENTITY, entity_id="ent_long", episode_id="ep_short")
    await graph.run(Q.LINK_EPISODE_ENTITY, entity_id="ent_long", episode_id="ep_long")
    await graph.run(Q.LINK_EPISODE_ENTITY, entity_id="ent_vela", episode_id="ep_vt")
    await graph.run(Q.LINK_EPISODE_ENTITY, entity_id="ent_vela", episode_id="ep_v3")

    repaired = await engine.repair_known_merges()
    by_name = {ent["name"]: ent for ent in graph.entities.values()}
    assert by_name["HOKI Rewards"]["type"] == "Product"
    assert by_name["VelaTavern"]["type"] == "Project"
    assert "HOKI Rewards" not in by_name["HOKI Rewards Mobile 13112611678"]["aliases"]
    assert "VelaTavern" not in by_name["Vela Tavern v3"]["aliases"]
    hoki_id = by_name["HOKI Rewards"]["id"]
    vela_id = by_name["VelaTavern"]["id"]
    assert (hoki_id, "ep_short") in graph.mentions
    assert ("ent_long", "ep_short") not in graph.mentions
    assert ("ent_long", "ep_long") in graph.mentions
    assert (hoki_id, "ep_long") not in graph.mentions
    assert (vela_id, "ep_vt") in graph.mentions
    assert ("ent_vela", "ep_vt") not in graph.mentions
    assert ("ent_vela", "ep_v3") in graph.mentions
    assert (vela_id, "ep_v3") not in graph.mentions
    await engine.repair_known_merges()
    assert sum(1 for ent in graph.entities.values() if ent["name"] == "HOKI Rewards") == 1
    assert repaired["restored"]
    await engine.close()


async def test_backfill_sample_is_specific():
    engine, graph, _llm = await make_engine()
    for i in range(20):
        await graph.run(
            Q.CREATE_EPISODE,
            id=f"ep_sample_{i}",
            content=f"Project: Widget{i}\nBoard: SharedBoard\nnotes {i}",
            summary=f"Widget{i}",
            tags=[],
            source_agent_id="backfill",
            source_session_id="",
            request_id="",
            content_hash=f"hash-{i}",
            key="",
        )
    result = await engine.backfill_entities(use_llm=False)
    assert result["episodes"] == 20
    assert "SharedBoard" in result["skipped_common"]
    assert len(result["sample"]) == 20
    for i, row in enumerate(result["sample"]):
        assert row["entities"] == [f"Widget{i}"]
    await engine.close()


async def test_context_index_and_topic_card_key():
    engine, _graph, _llm = await make_engine()
    index = await engine.remember("INDEX\n" + ("alpha " * 8000), agent_id="host", key="index")
    first = await engine.remember("v1 slingshot notes", agent_id="host", key="topic-card:slingshot")
    await settle(engine)
    second = await engine.remember("v2 slingshot notes are current", agent_id="host", key="topic-card:slingshot")
    assert second.episode_id == first.episode_id
    assert second.updated is True
    await settle(engine)
    keyed = [ep for ep in engine.graph.episodes.values() if ep.get("key") == "topic-card:slingshot"]
    assert len(keyed) == 1
    loaded = await engine.get_by_key("topic-card:slingshot")
    assert loaded["found"] is True
    assert loaded["content"] == "v2 slingshot notes are current"
    assert loaded["version"] == 2
    assert loaded["history"][0]["content"] == "v1 slingshot notes"
    assert loaded["history"][0]["version"] == 1
    missing = await engine.get_by_key("topic-card:missing")
    assert missing["found"] is False

    payload = await engine.context("")
    assert payload["index"]["episode_id"] == index.episode_id
    assert payload["index"]["title"] == "INDEX"
    titles = [card["title"] for card in payload["topic_cards"]]
    assert "TOPIC CARD: slingshot" in titles
    assert payload["approx_tokens"] <= 2000
    assert estimate_tokens(json.dumps(payload)) <= 2000
    named = await engine.context("Alpha")
    assert named["entity"] == "Alpha"
    assert "edges" in named
    await engine.close()


async def test_secrets_are_rejected_and_normal_text_saves():
    engine, graph, _llm = await make_engine()
    rejected = await engine.remember("password: hunter2-example", agent_id="host")
    assert rejected.rejected is True
    assert rejected.episode_id == ""
    assert "hunter2" not in rejected.error
    jwt = await engine.remember(f"debug dump {FAKE_JWT}", agent_id="host")
    assert jwt.rejected is True
    assert jwt.status == "rejected"
    assert FAKE_JWT not in jwt.error
    assert graph.episodes == {}
    saved = await engine.remember("Project: Widget9\nshipped the board notes", agent_id="host")
    assert saved.rejected is False
    assert saved.episode_id in graph.episodes
    await engine.close()
