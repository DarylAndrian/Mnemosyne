# Mnemosyne

Shared GraphRAG memory MCP server for 3 Hermes agents. Fully local -- no data leaves the network.

## Architecture

```
Machine 1 (HOST: this PC)         Machine 2         Machine 3
+--------------------------+    +-------------+    +-------------+
| Neo4j (Docker)           |    | Hermes      |    | Hermes      |
| Ollama (native)          |<---| Agent       |    | Agent       |
| MCP server (Python)      | MCP| (client)    |    | (client)    |
|   Port 8080              |    +-------------+    +-------------+
+--------------------------+
```

- Machine 1 runs everything. Machines 2 & 3 are pure MCP clients.
- Neo4j handles graph + vector + full-text in one container.
- Ollama runs natively (not Docker) for simplicity.
- MCP is Hermes's native protocol -- agents get memory tools as first-class capabilities.

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/DarylAndrian/Mnemosyne.git
cd Mnemosyne
cp .env.example .env
# Edit .env with your passwords

# 2. Start Neo4j
docker compose up -d

# 3. Install Python deps
uv venv .venv
uv pip install -r requirements.txt

# 4. Start the server
python -m server.main
```

The server starts on `http://0.0.0.0:8080/mcp`. Health check at `/health`.

## Memory Graph Frontend

An Obsidian-style graph viewer is served at `http://<HOST_IP>:8080/` (same port as MCP).

- Force-directed graph of entities, colored by type
- Click a node: aliases, facts, episodes, 1-3 hop neighborhood
- Double-click: expand neighborhood
- Recall box: hybrid RAG search (same pipeline as the `recall` tool)
- Filters by entity type, node limit, include episodes
- Paste `MCP_API_KEY` or an agent token in the top-right to unlock API calls

No build step and no internet needed -- plain HTML/JS with vis-network vendored in
`frontend/vendor/`. The API endpoints (`/api/graph`, `/api/entity`, `/api/recall`,
`/api/episodes`, `/api/stats`) require the same Bearer token as MCP; static files are public.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | (required) | Neo4j password |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `EXTRACT_MODEL` | `qwen2.5:3b` | LLM for entity extraction |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `MCP_HOST` | `0.0.0.0` | Bind address |
| `MCP_PORT` | `8080` | Listen port |
| `MCP_API_KEY` | (required) | Shared API key for auth |

## MCP Tools

### `remember(content, agent_id, session_id?, tags?, request_id?)`

Store a memory. Returns immediately with `episode_id` and `processing: true`;
entity/relationship/fact extraction and embedding run in the background.
Detects conflicts with existing facts.

### `episode_status(episode_id)`

Check the background processing status of a stored episode:
`pending`, `processed`, `failed`, or `not_found`.

## Latency & Retries (read this before integrating)

- **`remember` returns in milliseconds.** The DB write happens first; the slow
  part (LLM extraction + embeddings, 10-20s on CPU-only hosts) runs in the
  background. No client timeout issues.
- **Always send a `request_id`** (a UUID per logical store intent). If your
  client times out and retries with the same `request_id`, the server returns
  the original episode (`deduplicated: true`) instead of creating a duplicate.
  Idempotency keys never expire.
- **Automatic dedup**: identical content (SHA-256 of normalized text) or
  near-identical content (>= 97% token overlap) stored within a 5-minute
  window returns the existing episode with `deduplicated: true`.
- Use `episode_status` to poll background completion if your agent needs
  extracted entities/facts to be ready; full-text recall works immediately,
  vector recall once embedding completes.

### `recall(query, top_k?, agent_id?)`

Hybrid RAG search: vector similarity + keyword + graph neighborhood expansion.
Fused with recency and access-count boosts.

### `context(entity_name, depth_limit?)`

Graph neighborhood traversal. Returns all edges connected to an entity within
1-3 hops.

### `resolve(entity_a, entity_b)`

Merge duplicate entities. Re-points all edges, keeps both names as aliases.

### `forget(memory_id)`

Soft-delete an episode. Preserves provenance.

## Agent Configuration

Add to your Hermes config:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "url": "http://<HOST_IP>:8080/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_API_KEY>"
      }
    }
  }
}
```

## Infrastructure

- **Neo4j 5.26** (Community) -- graph + vector + full-text in one container
- **Ollama 0.32+** -- qwen2.5:3b (extraction) + nomic-embed-text (embeddings)
- **Python 3.11+** -- FastMCP server with Streamable HTTP transport
- **Docker Compose** -- Neo4j only (Ollama stays native)

## Development

```bash
# Run integration tests (requires live stack)
.venv/Scripts/python.exe -c "from tests.test_integration import *; ..."
```

## License

MIT
