"""Neo4j schema creation: constraints, indexes, full-text + vector indexes.

Idempotent -- safe to run on every startup.
"""

from __future__ import annotations

from .driver import GraphDriver

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Episode) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Fact) REQUIRE n.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Episode) ON (n.created_at)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Fact) ON (n.subject)",
]

FULLTEXT_INDEXES = [
    "CREATE FULLTEXT INDEX memory_fulltext IF NOT EXISTS FOR (e:Episode) ON EACH [e.content, e.summary]",
    "CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS FOR (f:Fact) ON EACH [f.content]",
]

VECTOR_INDEXES = [
    # 768-dim = nomic-embed-text output
    """CREATE VECTOR INDEX memory_vector IF NOT EXISTS
       FOR (e:Episode) ON (e.embedding)
       OPTIONS {indexConfig: {
         `vector.dimensions`: 768,
         `vector.similarity_function`: 'cosine'
       }}""",
    """CREATE VECTOR INDEX entity_vector IF NOT EXISTS
       FOR (e:Entity) ON (e.summary_embedding)
       OPTIONS {indexConfig: {
         `vector.dimensions`: 768,
         `vector.similarity_function`: 'cosine'
       }}""",
]


async def ensure_schema(graph: GraphDriver) -> dict:
    """Create all constraints and indexes. Returns a summary."""
    created = {"constraints": 0, "indexes": 0, "fulltext": 0, "vector": 0}
    for q in CONSTRAINTS:
        await graph.run(q)
        created["constraints"] += 1
    for q in INDEXES:
        await graph.run(q)
        created["indexes"] += 1
    for q in FULLTEXT_INDEXES:
        await graph.run(q)
        created["fulltext"] += 1
    for q in VECTOR_INDEXES:
        await graph.run(q)
        created["vector"] += 1
    return created
