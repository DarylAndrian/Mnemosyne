#!/usr/bin/env python
"""Entity-extraction recheck (task 9): how many episodes actually have entities.

Run from the Mnemosyne repo root with its venv so it hits the same Neo4j.
Reads NEO4J_* from .env via server.config. Prints counts only - no content,
no secrets.
"""
import asyncio, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import Settings
from server.graph.driver import GraphDriver

Q_STATUS = """
MATCH (e:Episode)
WITH coalesce(e.status, 'processed') AS status, count(*) AS n
RETURN status, n
ORDER BY status
"""

Q_LINKED = """
MATCH (e:Episode)
WHERE coalesce(e.active, true) = true
WITH coalesce(e.status, 'processed') AS status, e AS ep
OPTIONAL MATCH (ent:Entity)-[:MENTIONED_IN]->(ep)
WITH status, ep, count(ent) AS ents
RETURN status, count(*) AS episodes, sum(CASE WHEN ents > 0 THEN 1 ELSE 0 END) AS with_entities,
       sum(ents) AS total_links, avg(ents) AS avg_per_episode
ORDER BY status
"""

Q_TOP_ENTITIES = """
MATCH (ent:Entity)-[:MENTIONED_IN]->(e:Episode)
WHERE coalesce(e.active, true) = true
WITH ent.name AS name, count(*) AS links
ORDER BY links DESC
LIMIT 12
RETURN name, links
"""

Q_TOTAL_ENTITIES = "MATCH (e:Entity) RETURN count(e) AS n"

Q_RECENT_EMPTY = """
MATCH (e:Episode)
WHERE coalesce(e.active, true) = true AND coalesce(e.status,'') = 'processed'
WITH e ORDER BY e.created_at DESC LIMIT 25
OPTIONAL MATCH (ent:Entity)-[:MENTIONED_IN]->(e)
WITH e, count(ent) AS ents
WHERE ents = 0
RETURN e.id AS id
"""


async def main() -> None:
    g = GraphDriver(Settings())
    try:
        print("== episode status counts ==")
        for r in await g.run(Q_STATUS):
            print(f"   {r['status']:<12} {r['n']}")

        print("\n== entity linkage per status ==")
        for r in await g.run(Q_LINKED):
            print(f"   {r['status']:<12} episodes={r['episodes']:<5} "
                  f"with_entities={r['with_entities']:<5} "
                  f"total_links={r['total_links']:<6} "
                  f"avg/ep={(r['avg_per_episode'] or 0):.2f}")

        tot = await g.run_one(Q_TOTAL_ENTITIES)
        print(f"\ntotal distinct entities: {tot['n'] if tot else '?'}")

        print("\ntop entities by link count:")
        for r in await g.run(Q_TOP_ENTITIES):
            print(f"   {r['links']:>4}  {r['name']}")

        empty = await g.run(Q_RECENT_EMPTY)
        print(f"\nrecent 200-processed window: {len(empty)} of the last 25 processed episodes have ZERO entities")
        if empty:
            print("   ids:", [r["id"] for r in empty[:8]])
    finally:
        await g.close()


asyncio.run(main())
