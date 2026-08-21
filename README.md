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

### `remember(content, agent_id, session_id?, tags?)`

Store a memory. Server extracts entities, relationships, and facts into the
knowledge graph. Detects conflicts with existing facts.

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
