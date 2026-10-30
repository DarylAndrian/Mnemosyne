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
  key: $key,
  version: 1,
  status: 'pending',
  error_message: '',
  attempts: 0,
  last_attempt_at: null,
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
RETURN ep.id AS id,
       coalesce(ep.status, 'processed') AS status,
       coalesce(ep.request_id, '') AS request_id,
       coalesce(ep.error_message, '') AS error_message,
       coalesce(ep.attempts, 0) AS attempts,
       toString(ep.last_attempt_at) AS last_attempt_at
"""

SET_EPISODE_FAILURE = """
MATCH (ep:Episode {id: $episode_id})
SET ep.status = $status,
    ep.error_message = $error_message,
    ep.attempts = $attempts,
    ep.last_attempt_at = datetime(),
    ep.updated_at = datetime()
RETURN ep.id AS id, ep.attempts AS attempts
"""

MARK_EPISODE_PROCESSED = """
MATCH (ep:Episode {id: $episode_id})
SET ep.status = 'processed',
    ep.error_message = '',
    ep.attempts = $attempts,
    ep.last_attempt_at = datetime(),
    ep.updated_at = datetime()
RETURN ep.id AS id
"""

GET_EPISODE_FOR_PROCESSING = """
MATCH (ep:Episode {id: $episode_id})
RETURN ep.id AS id,
       ep.content AS content,
       coalesce(ep.source_agent_id, '') AS source_agent_id,
       coalesce(ep.source_session_id, '') AS source_session_id,
       coalesce(ep.tags, []) AS tags,
       coalesce(ep.active, true) AS active,
       coalesce(ep.status, 'processed') AS status
"""

GET_EPISODES_BY_IDS = """
MATCH (ep:Episode)
WHERE ep.id IN $ids
RETURN ep.id AS id,
       ep.content AS content,
       coalesce(ep.source_agent_id, '') AS source_agent_id,
       coalesce(ep.source_session_id, '') AS source_session_id,
       coalesce(ep.tags, []) AS tags,
       coalesce(ep.active, true) AS active,
       coalesce(ep.status, 'processed') AS status
"""

LIST_FAILED_EPISODES = """
MATCH (ep:Episode)
WHERE coalesce(ep.status, '') = 'failed' AND coalesce(ep.active, true) = true
RETURN ep.id AS id,
       ep.content AS content,
       coalesce(ep.source_agent_id, '') AS source_agent_id,
       coalesce(ep.source_session_id, '') AS source_session_id,
       coalesce(ep.tags, []) AS tags,
       coalesce(ep.active, true) AS active,
       coalesce(ep.status, 'failed') AS status
LIMIT $limit
"""

CLEAR_EPISODE_LINKS = """
MATCH (e:Entity)-[r:MENTIONED_IN]->(ep:Episode {id: $episode_id})
WITH collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

DELETE_EPISODE_FACTS = """
MATCH (f:Fact {source_episode_id: $episode_id})
WITH collect(f) AS facts
FOREACH (f IN facts | DETACH DELETE f)
RETURN size(facts) AS deleted
"""

FIND_EPISODE_BY_KEY = """
MATCH (ep:Episode)
WHERE ep.key = $key AND ep.key <> '' AND coalesce(ep.active, true) = true
RETURN ep.id AS id,
       ep.content AS content,
       coalesce(ep.summary, '') AS summary,
       coalesce(ep.version, 1) AS version,
       coalesce(ep.status, 'processed') AS status,
       coalesce(ep.source_agent_id, '') AS source_agent_id,
       coalesce(ep.source_session_id, '') AS source_session_id,
       coalesce(ep.tags, []) AS tags,
       coalesce(ep.key, '') AS key,
       toString(ep.created_at) AS created_at,
       toString(ep.updated_at) AS updated_at
ORDER BY ep.updated_at DESC
LIMIT 1
"""

UPDATE_EPISODE_CONTENT = """
MATCH (ep:Episode {id: $episode_id})
SET ep.content = $content,
    ep.summary = $summary,
    ep.content_hash = $content_hash,
    ep.tags = $tags,
    ep.status = 'pending',
    ep.error_message = '',
    ep.version = $version,
    ep.updated_at = datetime()
RETURN ep.id AS id
"""

INSERT_EPISODE_VERSION = """
CREATE (v:EpisodeVersion {
  id: $id,
  episode_id: $episode_id,
  content: $content,
  summary: $summary,
  version: $version,
  created_at: datetime()
})
RETURN v.id AS id
"""

LIST_EPISODE_VERSIONS = """
MATCH (v:EpisodeVersion {episode_id: $episode_id})
RETURN v.version AS version,
       v.content AS content,
       toString(v.created_at) AS created_at
ORDER BY v.version ASC
"""

LIST_EPISODES = """
MATCH (ep:Episode)
WHERE coalesce(ep.active, true) = true
  AND CASE WHEN $text = '' THEN true
      ELSE toLower(ep.content) CONTAINS toLower($text)
        OR toLower(coalesce(ep.summary, '')) CONTAINS toLower($text) END
  AND CASE WHEN $source = '' THEN true ELSE ep.source_agent_id = $source END
  AND CASE WHEN $status = '' THEN true ELSE coalesce(ep.status, 'processed') = $status END
  AND CASE WHEN $created_after = '' THEN true ELSE ep.created_at >= datetime($created_after) END
  AND CASE WHEN $created_before = '' THEN true ELSE ep.created_at <= datetime($created_before) END
  AND CASE WHEN $cursor_time = '' THEN true
      ELSE ep.created_at < datetime($cursor_time)
        OR (ep.created_at = datetime($cursor_time) AND ep.id < $cursor_id) END
  AND CASE WHEN $entity = '' THEN true ELSE EXISTS {
      MATCH (e:Entity)-[:MENTIONED_IN]->(ep)
      WHERE toLower(e.name) = toLower($entity)
         OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) = toLower($entity))
  } END
RETURN ep.id AS id,
       toString(ep.created_at) AS created_at,
       coalesce(ep.source_agent_id, '') AS source,
       coalesce(ep.status, 'processed') AS status,
       substring(coalesce(ep.content, ''), 0, 160) AS preview
ORDER BY ep.created_at DESC, ep.id DESC
LIMIT $limit
"""

INDEX_EPISODE = """
MATCH (ep:Episode)
WHERE coalesce(ep.active, true) = true AND (
  toLower(coalesce(ep.key, '')) = 'index'
  OR toLower(substring(coalesce(ep.summary, ''), 0, 5)) = 'index'
  OR toLower(substring(coalesce(ep.content, ''), 0, 5)) = 'index'
)
RETURN ep.id AS id,
       ep.content AS content,
       coalesce(ep.summary, '') AS summary,
       toString(ep.updated_at) AS updated_at
ORDER BY ep.updated_at DESC
LIMIT 1
"""

TOPIC_CARD_EPISODES = """
MATCH (ep:Episode)
WHERE coalesce(ep.active, true) = true AND (
  toLower(substring(coalesce(ep.content, ''), 0, 11)) = 'topic card:'
  OR toLower(substring(coalesce(ep.summary, ''), 0, 11)) = 'topic card:'
  OR toLower(coalesce(ep.key, '')) STARTS WITH 'topic-card:'
)
RETURN ep.id AS id,
       coalesce(ep.key, '') AS key,
       substring(coalesce(ep.content, ''), 0, 160) AS content_head,
       substring(coalesce(ep.summary, ''), 0, 160) AS summary_head
ORDER BY ep.key, ep.id
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
SET e.aliases = CASE
  WHEN $alias IN coalesce(e.aliases, []) THEN coalesce(e.aliases, [])
  ELSE coalesce(e.aliases, []) + [$alias]
END
RETURN e.id AS id
"""

GET_ENTITY_BY_ID = """
MATCH (e:Entity {id: $id})
RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases,
       coalesce(e.description, '') AS description
"""

UNLINK_EPISODE_ENTITY = """
MATCH (e:Entity {id: $entity_id})-[r:MENTIONED_IN]->(ep:Episode {id: $episode_id})
WITH collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

FIND_ENTITY_BY_NAME_CI = """
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($name)
RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases,
       e.description AS description
LIMIT 1
"""

LIST_ALL_ENTITIES = """
MATCH (e:Entity)
RETURN e.id AS id, e.name AS name, e.type AS type,
       e.aliases AS aliases, coalesce(e.description, '') AS description
LIMIT $limit
"""

ENTITY_LINK_STATS = """
MATCH (e:Entity {id: $id})
OPTIONAL MATCH (e)-[:MENTIONED_IN]->(ep:Episode)
WHERE coalesce(ep.active, true) = true
WITH count(ep) AS linked
MATCH (tot:Episode)
WHERE coalesce(tot.active, true) = true
RETURN linked, count(tot) AS total
"""

RESET_EPISODE_FOR_REPROCESS = """
MATCH (ep:Episode {id: $episode_id})
SET ep.status = 'pending',
    ep.attempts = 0,
    ep.error_message = '',
    ep.last_attempt_at = null,
    ep.updated_at = datetime()
RETURN ep.id AS id, coalesce(ep.attempts, 0) AS attempts, coalesce(ep.status, 'pending') AS status
"""

EPISODE_ENTITY_NAMES = """
MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
WHERE ep.id IN $ids
RETURN ep.id AS episode_id, collect(DISTINCT e.name) AS names
"""

SET_ENTITY_ALIASES = """
MATCH (e:Entity {id: $id})
SET e.aliases = $aliases, e.updated_at = datetime()
RETURN e.id AS id
"""

RECREATE_ENTITY = """
CREATE (e:Entity {
  id: $id,
  name: $name,
  type: $type,
  description: $description,
  aliases: $aliases,
  created_at: datetime(),
  updated_at: datetime(),
  access_count: 0
})
RETURN e.id AS id
"""

LIST_ENTITY_EDGES = """
MATCH (b:Entity {id: $id})-[r]-(other:Entity)
RETURN type(r) AS rel_type,
       other.id AS other_id,
       CASE WHEN startNode(r) = b THEN 'out' ELSE 'in' END AS direction
"""

LIST_ENTITY_EPISODES = """
MATCH (e:Entity {id: $id})-[:MENTIONED_IN]->(ep:Episode)
RETURN ep.id AS id, ep.content AS content
"""

EPISODES_CONTAINING = """
MATCH (ep:Episode)
WHERE coalesce(ep.active, true) = true AND ep.content CONTAINS $phrase
RETURN ep.id AS id, ep.content AS content
"""

DELETE_RELATIONSHIP = """
MATCH (a {id: $from_id})-[r]->(b {id: $to_id})
WHERE type(r) = $rel_type
WITH collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

REPOINT_FACTS = """
MATCH (f:Fact)
WHERE f.subject = $old OR f.id IN $ids
SET f.subject = $new
RETURN count(f) AS n
"""

CREATE_MERGE_LOG = """
CREATE (m:MergeLog {
  id: $id,
  before_state: $before_state,
  kept_name: $kept_name,
  dropped_name: $dropped_name,
  undone: false,
  created_at: datetime()
})
RETURN m.id AS id
"""

GET_MERGE_LOG = """
MATCH (m:MergeLog {id: $id})
RETURN m.id AS id,
       m.before_state AS before_state,
       m.kept_name AS kept_name,
       m.dropped_name AS dropped_name,
       coalesce(m.undone, false) AS undone
"""

LIST_MERGE_LOGS = """
MATCH (m:MergeLog)
RETURN m.id AS id,
       m.before_state AS before_state,
       m.kept_name AS kept_name,
       m.dropped_name AS dropped_name,
       coalesce(m.undone, false) AS undone
"""

MARK_MERGE_UNDONE = """
MATCH (m:MergeLog {id: $id})
SET m.undone = true, m.undone_at = datetime()
RETURN m.id AS id
"""

ALL_ACTIVE_EPISODES = """
MATCH (ep:Episode)
WHERE coalesce(ep.active, true) = true
RETURN ep.id AS id, ep.content AS content
ORDER BY ep.created_at
LIMIT $limit
"""

COUNT_ACTIVE_EPISODES = """
MATCH (ep:Episode)
WHERE coalesce(ep.active, true) = true
RETURN count(ep) AS total
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
MERGE (e)-[r:MENTIONED_IN]->(ep)
SET r.source = e.id,
    r.target = ep.id,
    r.source_episode_id = ep.id,
    e.source_episode_id = coalesce(e.source_episode_id, $episode_id)
RETURN e.id AS id, r.target AS target
"""

# Existing episode-ended MENTIONED_IN edges were stored with no target property,
# so readers saw target = null and could not attach the episode node.
BACKFILL_MENTIONED_TARGETS = """
MATCH (e:Entity)-[r:MENTIONED_IN]->(ep:Episode)
WHERE ep.id IS NOT NULL AND (r.target IS NULL OR r.target <> ep.id)
SET r.source = e.id,
    r.target = ep.id,
    r.source_episode_id = coalesce(r.source_episode_id, ep.id),
    e.source_episode_id = coalesce(e.source_episode_id, ep.id)
RETURN count(r) AS repaired
"""

LIST_DANGLING_MENTIONED = """
MATCH (a:Entity)-[r:MENTIONED_IN]->(b)
WHERE b.id IS NULL OR r.target IS NULL OR NOT b:Episode OR r.target <> b.id
RETURN a.id AS from_id,
       coalesce(b.id, '') AS to_id,
       b:Episode AS to_is_episode,
       coalesce(a.source_episode_id, '') AS source_episode_id
LIMIT 5000
"""

CLEAR_BROKEN_MENTION = """
MATCH (a:Entity {id: $from_id})-[r:MENTIONED_IN]->(b)
WHERE b.id IS NULL OR r.target IS NULL OR NOT b:Episode OR r.target <> b.id
WITH collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

COUNT_NULL_MENTION_TARGETS = """
MATCH ()-[r:MENTIONED_IN]->(n)
WHERE r.target IS NULL OR n.id IS NULL
RETURN count(r) AS n
"""

STAMP_RELATIONSHIP = """
MATCH (a {id: $from_id})-[r]->(b {id: $to_id})
WHERE type(r) = $rel_type AND type(r) <> 'MENTIONED_IN'
SET r.source = a.id,
    r.target = b.id,
    r.source_episode_ids = CASE
      WHEN $episode_id = '' OR $episode_id IN coalesce(r.source_episode_ids, [])
        THEN coalesce(r.source_episode_ids, [])
      ELSE coalesce(r.source_episode_ids, []) + $episode_id
    END
RETURN count(r) AS stamped
"""

GET_EPISODE_BRIEF = """
MATCH (ep:Episode {id: $episode_id})
RETURN ep.id AS id,
       coalesce(ep.key, '') AS key,
       substring(coalesce(ep.content, ''), 0, 80) AS content_head,
       coalesce(ep.active, true) AS active
"""

LIST_PENDING_IDS = """
MATCH (ep:Episode)
WHERE coalesce(ep.active, true) = true AND coalesce(ep.status, '') = 'pending'
RETURN ep.id AS id
LIMIT $limit
"""

INVALIDATE_EPISODE_FACTS = """
MATCH (f:Fact {source_episode_id: $episode_id})
WHERE f.valid_until IS NULL
SET f.valid_until = datetime(), f.superseded_by = 'forgotten'
RETURN count(f) AS invalidated
"""

ENTITY_IDS_ONLY_ON_EPISODE = """
MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode {id: $episode_id})
WHERE NOT EXISTS {
  MATCH (e)-[:MENTIONED_IN]->(other:Episode)
  WHERE other.id <> $episode_id AND coalesce(other.active, true) = true
}
RETURN DISTINCT e.id AS id
"""

DROP_PROVENANCE_EDGES = """
MATCH ()-[r]->()
WHERE $episode_id IN coalesce(r.source_episode_ids, [])
WITH r, [x IN coalesce(r.source_episode_ids, []) WHERE x <> $episode_id] AS rest
UNWIND CASE WHEN size(rest) = 0 THEN [null] ELSE rest END AS candidate
OPTIONAL MATCH (ep:Episode {id: candidate})
WHERE coalesce(ep.active, true) = true
WITH r, [x IN collect(DISTINCT ep.id) WHERE x IS NOT NULL] AS alive
SET r.source_episode_ids = alive
WITH r, alive
WHERE size(alive) = 0
WITH collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

PURGE_STALE_EDGES = """
MATCH (a:Entity)-[r]->(b:Entity)
WHERE type(r) <> 'MENTIONED_IN'
  AND size(coalesce(r.source_episode_ids, [])) = 0
  AND (
    NOT EXISTS {
      MATCH (a)-[:MENTIONED_IN]->(ep:Episode)
      WHERE coalesce(ep.active, true) = true
    }
    OR NOT EXISTS {
      MATCH (b)-[:MENTIONED_IN]->(ep:Episode)
      WHERE coalesce(ep.active, true) = true
    }
  )
WITH collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

DELETE_EDGES_BETWEEN_NAMES = """
MATCH (a:Entity)-[r]-(b:Entity)
WHERE toLower(a.name) = toLower($left) AND toLower(b.name) = toLower($right)
WITH collect(DISTINCT r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

DELETE_INACTIVE_MENTIONS_BY_NAME = """
MATCH (e:Entity)-[r:MENTIONED_IN]->(ep:Episode)
WHERE toLower(e.name) = toLower($name) AND coalesce(ep.active, true) = false
WITH collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

COUNT_ACTIVE_MENTIONS_BY_NAME = """
MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
WHERE toLower(e.name) = toLower($name) AND coalesce(ep.active, true) = true
RETURN count(*) AS n
"""

# An entity whose only backing episodes were forgotten still keeps its
# derived entity-entity edges (USES, WORKING_ON, ...). Those edges came from
# deleted episodes too, so drop them to let the node itself be collected.
DROP_ALL_EDGES_BY_NAME = """
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($name)
OPTIONAL MATCH (e)-[r]-()
WITH e, collect(r) AS rels
FOREACH (rel IN rels | DELETE rel)
RETURN size(rels) AS deleted
"""

DELETE_ISOLATED_ENTITY_BY_NAME = """
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($name) AND NOT (e)--()
WITH collect(e) AS nodes
FOREACH (n IN nodes | DETACH DELETE n)
RETURN size(nodes) AS deleted
"""

DELETE_ISOLATED_ENTITIES = """
MATCH (e:Entity)
WHERE e.id IN $ids AND NOT (e)--()
WITH collect(e) AS nodes
FOREACH (n IN nodes | DETACH DELETE n)
RETURN size(nodes) AS deleted
"""

# Entities the index already owns (source_episode_id) but lost their edge
# when CLEAR_EPISODE_LINKS ran and the LLM did not re-mine them.
LIST_UNLINKED_OWNED_ENTITIES = """
MATCH (e:Entity)
WHERE coalesce(e.source_episode_id, '') = $episode_id
  AND NOT (e)-[:MENTIONED_IN]->(:Episode {id: $episode_id})
RETURN e.id AS id, coalesce(e.name, '') AS name
"""

LIST_ORPHANS = """
MATCH (n)
WHERE n:Entity OR n:Episode
OPTIONAL MATCH (n)-[:MENTIONED_IN]->(ep:Episode)
WITH n,
     count(ep) AS mentions,
     sum(CASE WHEN coalesce(ep.active, true) = true THEN 1 ELSE 0 END) AS live_mentions
WHERE $include_inactive = true
   OR (n:Episode AND coalesce(n.active, true) = true)
   OR (n:Entity AND (mentions = 0 OR live_mentions > 0))
OPTIONAL MATCH (n)-[r]-()
WITH n, mentions, live_mentions, count(r) AS degree
WHERE degree <= $max_degree
OPTIONAL MATCH (n)-[:MENTIONED_IN]->(mep:Episode)
WITH n, degree, mentions, live_mentions, head(collect(DISTINCT mep.id)) AS mentioned_ep
RETURN n.id AS id,
       CASE
         WHEN n:Episode THEN coalesce(n.key, substring(coalesce(n.summary, n.content, n.id), 0, 80))
         ELSE coalesce(n.name, n.id)
       END AS name,
       CASE WHEN n:Episode THEN 'Episode' ELSE coalesce(n.type, 'Entity') END AS type,
       CASE
         WHEN n:Episode THEN n.id
         ELSE coalesce(n.source_episode_id, mentioned_ep, '')
       END AS source_episode_id,
       degree
ORDER BY name, id
LIMIT $limit
"""

# Mirrors LIST_ORPHANS (eligibility + degree) exactly so the list count and
# the count query always agree: forgotten episodes are hidden, and so are
# entities whose every mention points at a forgotten episode. Entities with
# no episode at all stay visible - they are genuine orphans.
COUNT_ORPHANS = """
MATCH (n)
WHERE n:Entity OR n:Episode
OPTIONAL MATCH (n)-[:MENTIONED_IN]->(ep:Episode)
WITH n,
     count(ep) AS mentions,
     sum(CASE WHEN coalesce(ep.active, true) = true THEN 1 ELSE 0 END) AS live_mentions
WHERE $include_inactive = true
   OR (n:Episode AND coalesce(n.active, true) = true)
   OR (n:Entity AND (mentions = 0 OR live_mentions > 0))
OPTIONAL MATCH (n)-[r]-()
WITH n, count(r) AS degree
WHERE degree <= $max_degree
RETURN count(n) AS n
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
  source_session_id: $source_session_id,
  source_episode_id: $episode_id
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
       startNode(rel).id AS from_id,
       coalesce(startNode(rel).name, startNode(rel).key, startNode(rel).id) AS from_name,
       type(rel) AS rel_type,
       endNode(rel).id AS to_id,
       coalesce(endNode(rel).name, endNode(rel).key, endNode(rel).id) AS to_name,
       endNode(rel).id AS target,
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
SET a.aliases = CASE
      WHEN b.name IN coalesce(a.aliases, []) THEN coalesce(a.aliases, [])
      ELSE coalesce(a.aliases, []) + [b.name]
    END,
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
