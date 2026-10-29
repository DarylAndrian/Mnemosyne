"""Admin commands for the memory graph.

    python -m server.maintenance reprocess-failed --ids ep_abc,ep_def
    python -m server.maintenance repair-merges
    python -m server.maintenance backfill-entities

Prints ids, statuses, and counts only. Episode text is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import get_settings
from .engine import MemoryEngine
from .graph.schema import ensure_schema


async def _open() -> MemoryEngine:
    engine = MemoryEngine(get_settings())
    if not await engine.graph.ping():
        await engine.close()
        raise SystemExit("Neo4j is unreachable; nothing was changed.")
    await ensure_schema(engine.graph)
    return engine


async def _reprocess(episode_ids: list[str] | None) -> dict:
    engine = await _open()
    try:
        return await engine.reprocess_failed(episode_ids=episode_ids, wait=True)
    finally:
        await engine.close()


async def _repair() -> dict:
    engine = await _open()
    try:
        return await engine.repair_known_merges()
    finally:
        await engine.close()


async def _backfill(limit: int | None) -> dict:
    engine = await _open()
    try:
        return await engine.backfill_entities(limit=limit, use_llm=True)
    finally:
        await engine.close()


async def _repair_connectivity() -> dict:
    """Backfill MENTIONED_IN targets, drop the stale Vela edge, reprocess the stuck index."""
    from .graph import queries as Q

    stuck_id = "ep_3093510dda5c45c1"
    engine = await _open()
    try:
        backfill = await engine.backfill_mentioned_in_targets()
        stale = await engine.graph.run(Q.PURGE_STALE_EDGES)
        await engine.remove_named_edge("Mnemosyne", "Vela Tavern v3")
        reprocess = await engine.reprocess_failed([stuck_id], wait=True)
        # Run again after reprocess so a rebuilt edge from deleted episodes does not stick.
        vela = await engine.remove_named_edge("Mnemosyne", "Vela Tavern v3")
        nulls = await engine.graph.run_one(Q.COUNT_NULL_MENTION_TARGETS)
        orphans = await engine.list_orphans()
        direct = await engine.graph.run_one(Q.COUNT_ORPHANS, max_degree=0)
        status = await engine.episode_status(stuck_id)
        links = await engine.graph.run_one(
            "MATCH (e:Entity)-[r:MENTIONED_IN]->(ep:Episode {id: $id}) "
            "WHERE r.target = ep.id "
            "RETURN count(e) AS n",
            id=stuck_id,
        )
        vela_left = await engine.graph.run_one(
            "MATCH (e:Entity) WHERE toLower(e.name) = 'vela tavern v3' RETURN count(e) AS n"
        )
        return {
            "backfill": backfill,
            "stale_edges_removed": int((stale[0].get("deleted") if stale else 0) or 0),
            "vela": vela,
            "vela_nodes_remaining": int((vela_left or {}).get("n") or 0),
            "null_targets": int((nulls or {}).get("n") or 0),
            "orphans_count": orphans["count"],
            "orphans_direct": int((direct or {}).get("n") or 0),
            "orphans_match": orphans["count"] == int((direct or {}).get("n") or 0),
            "stuck_status": status.get("status"),
            "stuck_attempts": status.get("attempts"),
            "reprocess_queued": reprocess.get("queued"),
            "reprocess_missing": reprocess.get("missing"),
            "index_topic_links": int((links or {}).get("n") or 0),
        }
    finally:
        await engine.close()


def _print(payload: dict) -> None:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mnemosyne maintenance")
    sub = parser.add_subparsers(dest="command", required=True)

    reprocess = sub.add_parser("reprocess-failed", help="Re-queue failed episodes")
    reprocess.add_argument(
        "--ids",
        default="",
        help="Comma-separated episode ids. Omit to re-queue every failed episode.",
    )

    sub.add_parser("repair-merges", help="Split the known bad entity merges")

    backfill = sub.add_parser("backfill-entities", help="Re-extract entities for existing episodes")
    backfill.add_argument("--limit", type=int, default=0, help="Max episodes (0 = all)")

    sub.add_parser(
        "repair-connectivity",
        help="Backfill MENTIONED_IN targets, drop Vela Tavern v3, reprocess the stuck index",
    )

    args = parser.parse_args()
    if args.command == "reprocess-failed":
        ids = [part.strip() for part in args.ids.split(",") if part.strip()] or None
        _print(asyncio.run(_reprocess(ids)))
        return
    if args.command == "repair-merges":
        _print(asyncio.run(_repair()))
        return
    if args.command == "backfill-entities":
        _print(asyncio.run(_backfill(args.limit or None)))
        return
    if args.command == "repair-connectivity":
        _print(asyncio.run(_repair_connectivity()))
        return
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
