"""Cypher query templates for the memory graph."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

CREATE_EPISODE = """
CREATE (ep:Episode {
  id: $id,
  content: $content,
  summary: $summary,
  tags: $tags,
  source_agent_id: $source_agent_id,
  source_session_id: $source_session_id,
  request_id: $request_id,
  content_hash: $content_hash,
  status: 'pending',
  created_at: datetime(),
  updated_at: datetime(),
  active: true,
  access_count: 0
})
RETURN ep.id AS id
"""

# Idempotency: find the episode created for a given client request_id
FIND_EPISODE_BY_REQUEST_ID = """
MATCH (ep:Episode)
WHERE ep.request_id = $rid AND ep.request_id <> ''
RETURN ep.id AS id, ep.status AS status
LIMIT 1
"""

# Exact content dedup: same hash within the dedup window
FIND_EPISODE_BY_HASH_RECENT = """
MATCH (ep:Episode)
WHERE ep.content_hash = $content_hash AND ep.content_hash <> ''
  AND ep.active = true
  AND ep.created_at > datetime() - duration('PT5M')
RETURN ep.id AS id, ep.status AS status
ORDER BY ep.created_at DESC
LIMIT 1
"""

# Near-duplicate check: recent episodes for token-overlap comparison
RECENT_EPISODES_IN_WINDOW = """
MATCH (ep:Episode)
WHERE ep.active = true
  AND ep.created_at > datetime() - duration('PT5M')
RETURN ep.id AS id, ep.content AS content, ep.status AS status
ORDER BY ep.created_at DESC
LIMIT 20
"""

SET_EPISODE_STATUS = """
MATCH (ep:Episode {id: $episode_id})
SET ep.status = $status, ep.updated_at = datetime()
RETURN ep.id AS id
"""

GET_EPISODE_STATUS = """
MATCH (ep:Episode {id: $episode_id})
RETURN ep.id AS id, ep.status AS status, ep.request_id AS request_id
"""

SET_EPISODE_EMBEDDING = """
MATCH (ep:Episode {id: $episode_id})
CALL db.create.setNodeVectorProperty(ep, 'embedding', $vector)
RETURN ep.id AS id
"""

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

FIND_ENTITY_EXACT = """
MATCH (e:Entity)
WHERE e.name = $name OR $name IN e.aliases
RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases,
       e.description AS description
LIMIT 1
"""

FIND_ENTITY_FUZZY = """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($needle)
   OR any(a IN e.aliases WHERE toLower(a) CONTAINS toLower($needle))
RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases
LIMIT 5
"""

UPSERT_ENTITY = """
MERGE (e:Entity {id: $id})
SET e.name = $name,
    e.type = $type,
    e.description = $description,
    e.aliases = $aliases,
    e.updated_at = datetime(),
    e.created_at = coalesce(e.created_at, datetime()),
    e.access_count = coalesce(e.access_count, 0)
RETURN e.id AS id, e.name AS name
"""

ADD_ALIAS = """
MATCH (e:Entity {id: $id})
SET e.aliases = CASE WHEN $alias IN e.aliases THEN e.aliases ELSE e.aliases + $alias END
RETURN e.id AS id
"""

SET_ENTITY_EMBEDDING = """
MATCH (e:Entity {id: $entity_id})
CALL db.create.setNodeVectorProperty(e, 'summary_embedding', $vector)
RETURN e.id AS id
"""

# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

CREATE_RELATIONSHIP = """
MATCH (a:Entity {id: $from_id}), (b:Entity {id: $to_id})
CALL apoc.create.relationship(a, $type, $props, b) YIELD rel
RETURN id(rel) AS rel_id
"""

# Fallback when APOC is not available -- covers the standard type set with
# explicit branches (relationship types cannot be parameterized in plain Cypher).
CREATE_RELATIONSHIP_STATIC = """
MATCH (a:Entity {id: $from_id}), (b:Entity {id: $to_id})
FOREACH (_ IN CASE WHEN $type = 'USES' THEN [1] ELSE [] END | MERGE (a)-[:USES]->(b))
FOREACH (_ IN CASE WHEN $type = 'DEPENDS_ON' THEN [1] ELSE [] END | MERGE (a)-[:DEPENDS_ON]->(b))
FOREACH (_ IN CASE WHEN $type = 'INTEGRATES_WITH' THEN [1] ELSE [] END | MERGE (a)-[:INTEGRATES_WITH]->(b))
FOREACH (_ IN CASE WHEN $type = 'MANAGED_BY' THEN [1] ELSE [] END | MERGE (a)-[:MANAGED_BY]->(b))
FOREACH (_ IN CASE WHEN $type = 'OWNED_BY' THEN [1] ELSE [] END | MERGE (a)-[:OWNED_BY]->(b))
FOREACH (_ IN CASE WHEN $type = 'CREATED_BY' THEN [1] ELSE [] END | MERGE (a)-[:CREATED_BY]->(b))
FOREACH (_ IN CASE WHEN $type = 'CONTRADICTS' THEN [1] ELSE [] END | MERGE (a)-[:CONTRADICTS]->(b))
FOREACH (_ IN CASE WHEN $type = 'SUPERSEDES' THEN [1] ELSE [] END | MERGE (a)-[:SUPERSEDES]->(b))
FOREACH (_ IN CASE WHEN $type = 'HAS_PROPERTY' THEN [1] ELSE [] END | MERGE (a)-[:HAS_PROPERTY]->(b))
FOREACH (_ IN CASE WHEN $type = 'HAS_BEHAVIOR' THEN [1] ELSE [] END | MERGE (a)-[:HAS_BEHAVIOR]->(b))
FOREACH (_ IN CASE WHEN $type = 'DECIDED_ON' THEN [1] ELSE [] END | MERGE (a)-[:DECIDED_ON]->(b))
FOREACH (_ IN CASE WHEN $type = 'WORKING_ON' THEN [1] ELSE [] END | MERGE (a)-[:WORKING_ON]->(b))
FOREACH (_ IN CASE WHEN $type = 'MENTIONED_IN' THEN [1] ELSE [] END | MERGE (a)-[:MENTIONED_IN]->(b))
FOREACH (_ IN CASE WHEN NOT $type IN ['USES','DEPENDS_ON','INTEGRATES_WITH','MANAGED_BY','OWNED_BY','CREATED_BY','CONTRADICTS','SUPERSEDES','HAS_PROPERTY','HAS_BEHAVIOR','DECIDED_ON','WORKING_ON','MENTIONED_IN'] THEN [1] ELSE [] END | MERGE (a)-[:RELATED_TO]->(b))
RETURN count(*) AS done
"""

LINK_EPISODE_ENTITY = """
MATCH (ep:Episode {id: $episode_id}), (e:Entity {id: $entity_id})
MERGE (e)-[:MENTIONED_IN]->(ep)
RETURN e.id AS id
"""

# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

CREATE_FACT = """
CREATE (f:Fact {
  id: $id,
  content: $content,
  subject: $subject,
  confidence: $confidence,
  created_at: datetime(),
  updated_at: datetime(),
  valid_until: null,
  superseded_by: null,
  source_agent_id: $source_agent_id,
  source_session_id: $source_session_id
})
RETURN f.id AS id
"""

FIND_FACTS_ON_SUBJECT = """
MATCH (f:Fact)
WHERE f.subject = $subject_id AND f.valid_until IS NULL
RETURN f.id AS id, f.content AS content, f.confidence AS confidence,
       f.source_agent_id AS source_agent_id, f.created_at AS created_at
"""

SUPERSEDE_FACT = """
MATCH (old:Fact {id: $old_id}), (new:Fact {id: $new_id})
SET old.valid_until = datetime(),
    old.superseded_by = new.id,
    old.updated_at = datetime()
RETURN old.id AS superseded_id
"""

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

VECTOR_SEARCH_EPISODES = """
CALL db.index.vector.queryNodes('memory_vector', $limit, $query_vector)
YIELD node, score
WHERE node.active = true
RETURN node.id AS id, node.content AS content, score,
       node.source_agent_id AS source_agent_id,
       node.source_session_id AS source_session_id,
       toString(node.created_at) AS created_at,
       node.tags AS tags
"""

FULLTEXT_SEARCH_EPISODES = """
CALL db.index.fulltext.queryNodes('memory_fulltext', $q)
YIELD node, score
WHERE node.active = true
RETURN node.id AS id, node.content AS content, score,
       node.source_agent_id AS source_agent_id,
       node.source_session_id AS source_session_id,
       toString(node.created_at) AS created_at,
       node.tags AS tags
LIMIT $limit
"""

FULLTEXT_SEARCH_FACTS = """
CALL db.index.fulltext.queryNodes('fact_fulltext', $q)
YIELD node, score
WHERE node.valid_until IS NULL
RETURN node.id AS id, node.content AS content, score,
       node.subject AS subject_id, node.confidence AS confidence,
       node.source_agent_id AS source_agent_id,
       toString(node.created_at) AS created_at
LIMIT $limit
"""

VECTOR_SEARCH_ENTITIES = """
CALL db.index.vector.queryNodes('entity_vector', $limit, $query_vector)
YIELD node, score
RETURN node.id AS id, node.name AS name, node.type AS type, score
"""

# Graph expansion from matched entities (1-2 hops), returning paths as text
GRAPH_EXPAND = """
MATCH (e:Entity)
WHERE e.id IN $entity_ids
MATCH path = (e)-[*1..2]-(related)
UNWIND relationships(path) AS r
WITH e, r, related, startNode(r) AS s, endNode(r) AS t
RETURN DISTINCT e.name AS root, type(r) AS rel_type,
       s.name AS from_name, t.name AS to_name,
       coalesce(related.name, '') AS related_name
LIMIT $limit
"""

FACTS_FOR_ENTITIES = """
MATCH (f:Fact)
WHERE f.subject IN $entity_ids AND f.valid_until IS NULL
RETURN f.id AS id, f.content AS content, f.subject AS subject_id,
       f.confidence AS confidence, toString(f.created_at) AS created_at
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Context / resolve / forget
# ---------------------------------------------------------------------------

ENTITY_NEIGHBORHOOD = """
MATCH (e:Entity)
WHERE e.name = $name OR $name IN e.aliases
WITH e LIMIT 1
MATCH (e)-[r*1..3]-(related)
UNWIND r AS rel
RETURN DISTINCT
       e.name AS root_name, e.type AS root_type,
       startNode(rel).name AS from_name, type(rel) AS rel_type,
       endNode(rel).name AS to_name,
       related.name AS neighbor, related.type AS neighbor_type
LIMIT $limit
"""

REPARENT_ENTITY = """
// Merge entity_b into entity_a: move relationships, collect aliases
MATCH (a:Entity {id: $keep_id}), (b:Entity {id: $drop_id})
OPTIONAL MATCH (b)-[r_in]->(x) WHERE x <> a
WITH a, b, collect(DISTINCT [startNode(r_in).id, type(r_in)]) AS out_rels
OPTIONAL MATCH (y)-[r_out]->(b) WHERE y <> a
WITH a, b, out_rels, collect(DISTINCT [y.id, type(r_out)]) AS in_rels
SET a.aliases = CASE WHEN b.name IN a.aliases THEN a.aliases ELSE a.aliases + b.name END,
    a.updated_at = datetime()
RETURN a.id AS keep_id, a.aliases AS aliases, size(out_rels) AS moved_out, size(in_rels) AS moved_in
"""

REPARENT_EDGES = """
// Re-create edges that pointed at the dropped entity, on the kept one
MATCH (a:Entity {id: $keep_id})
UNWIND $edges AS edge
  MATCH (other:Entity {id: edge.other_id})
  CALL apoc.create.relationship(
    CASE WHEN edge.direction = 'out' THEN a ELSE other END,
    edge.rel_type, {},
    CASE WHEN edge.direction = 'out' THEN other ELSE a END
  ) YIELD rel
RETURN count(rel) AS reconnected
"""

COLLECT_EDGES_OF = """
MATCH (b:Entity {id: $drop_id})-[r]-(other)
WHERE other.id <> $keep_id
RETURN type(r) AS rel_type, other.id AS other_id,
       CASE WHEN startNode(r).id = $drop_id THEN 'out' ELSE 'in' END AS direction
"""

DELETE_ENTITY = """
MATCH (b:Entity {id: $drop_id})
DETACH DELETE b
RETURN $drop_id AS deleted
"""

SOFT_DELETE_EPISODE = """
MATCH (ep:Episode {id: $episode_id})
SET ep.active = false,
    ep.deleted_at = datetime(),
    ep.updated_at = datetime()
WITH ep
OPTIONAL MATCH (f:Fact {source_session_id: ep.source_session_id, source_agent_id: ep.source_agent_id})
WHERE f.valid_until IS NULL
SET f.valid_until = datetime(), f.superseded_by = 'forgotten'
RETURN ep.id AS id
"""

BUMP_ACCESS = """
MATCH (n) WHERE n.id = $id
SET n.access_count = coalesce(n.access_count, 0) + 1
RETURN n.id AS id
"""

HEALTH_STATS = """
MATCH (n)
RETURN count{n:Episode} AS episodes,
       count{n:Entity} AS entities,
       count{n:Fact} AS facts
"""
