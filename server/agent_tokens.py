"""Agent token management for Daryl-Memories.

Each agent gets a unique token mapped to an agent_id.
The auth middleware looks up the token and auto-sets agent_id on tool calls.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

TOKENS_FILE = Path(__file__).resolve().parent.parent / "agent-tokens.json"


def load_tokens() -> dict:
    """Load token -> agent_id mapping from disk."""
    if TOKENS_FILE.exists():
        return json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    return {}


def save_tokens(tokens: dict) -> None:
    """Persist token -> agent_id mapping."""
    TOKENS_FILE.write_text(
        json.dumps(tokens, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_agent(name: str, display_name: str = "") -> dict:
    """Create a new agent with a unique token. Returns the agent info."""
    tokens = load_tokens()
    token = f"daryl_mem_{secrets.token_urlsafe(32)}"
    agent_id = f"hermes-{name}"
    tokens[token] = {
        "agent_id": agent_id,
        "display_name": display_name or name,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    save_tokens(tokens)
    return {"token": token, "agent_id": agent_id, "display_name": display_name or name}


def revoke_agent(token: str) -> bool:
    """Remove a token. Returns True if found and removed."""
    tokens = load_tokens()
    if token in tokens:
        del tokens[token]
        save_tokens(tokens)
        return True
    return False


def list_agents() -> list[dict]:
    """List all registered agents."""
    tokens = load_tokens()
    result = []
    for tok, info in tokens.items():
        result.append({
            "token_prefix": tok[:16] + "...",
            "agent_id": info["agent_id"],
            "display_name": info.get("display_name", ""),
            "created_at": info.get("created_at", ""),
        })
    return result


def lookup_agent(token: str) -> str | None:
    """Look up agent_id by token. Returns None if not found."""
    tokens = load_tokens()
    info = tokens.get(token)
    return info["agent_id"] if info else None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python agent_tokens.py create <name> [display_name]")
        print("  python agent_tokens.py list")
        print("  python agent_tokens.py revoke <token>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "create":
        name = sys.argv[2] if len(sys.argv) > 2 else "agent"
        display = sys.argv[3] if len(sys.argv) > 3 else ""
        info = create_agent(name, display)
        print(f"Agent created:")
        print(f"  agent_id:      {info['agent_id']}")
        print(f"  display_name:  {info['display_name']}")
        print(f"  token:         {info['token']}")
        print(f"\nAdd to their Hermes config:")
        print(f'  "mcpServers": {{')
        print(f'    "daryl-memories": {{')
        print(f'      "url": "http://<THIS_HOST_IP>:8080/mcp",')
        print(f'      "headers": {{"Authorization": "Bearer {info["token"]}"}}')
        print(f'    }}')
        print(f'  }}')
    elif cmd == "list":
        agents = list_agents()
        if not agents:
            print("No agents registered.")
        for a in agents:
            print(f"  {a['agent_id']:20} {a['token_prefix']:20} {a['display_name']}")
    elif cmd == "revoke":
        token = sys.argv[2] if len(sys.argv) > 2 else ""
        if revoke_agent(token):
            print("Revoked.")
        else:
            print("Token not found.")
    else:
        print(f"Unknown command: {cmd}")
