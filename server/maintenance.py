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
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
