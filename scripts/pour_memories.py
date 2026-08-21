"""Pour all Hermes memories into Daryl-Memories shared memory system.

Calls the engine directly (bypasses MCP session management).
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.config import Settings
from server.engine import MemoryEngine
from server.graph.schema import ensure_schema

MEMORIES = [
    # === From MEMORY.md (agent notes) ===
    (
        "Xiaomi MiMo API: endpoint is https://token-plan-sgp.xiaomimimo.com/v1 "
        "(NOT api.xiaomi.com). Model limits: mimo-v2.5-pro has 1M context, "
        "mimo-v2.5 and mimo-v2-flash have 256K context, 32K output. "
        "Default provider set in Hermes config.yaml.",
        ["api", "mimo", "xiaomi", "config"]
    ),
    (
        "Daryl writing style: plain ASCII only in any markdown or documentation. "
        "No emoji, no Unicode box-drawing characters, no em-dashes. Use hyphens instead. "
        "He explicitly rejected emoji saying I dont want that weird icons. "
        "Use pipe-tables, setext-style headings, hyphens for separators. "
        "Applies to README files, release notes, GitHub release bodies, plan docs, "
        "and any repo file he reads on screen.",
        ["daryl", "style", "writing", "preference"]
    ),
    (
        "Daryl execution style: once confirmed, go end-to-end. "
        "Mobile-responsive UI matters for web apps. "
        "SillyTavern story images must be LOCAL in characters/StoryName/ folder "
        "and referenced as /characters/StoryName/file.ext -- NOT Cloud Drive URLs. "
        "Cloud Drive is just his personal backup, not for SillyTavern image serving.",
        ["daryl", "style", "execution", "sillytavern", "images"]
    ),
    (
        "CloudDrive: Use MCP cloud-drive-jeannette for uploads. "
        "Known folder IDs: Three Realms=t--qSqhLBoeCjkvJOLcXs, "
        "Final Chapter=fbuzT4Wo0L_PbTq2G36yV, "
        "Black Wing=qZQq2pk-5jrOAC4ZgVTf-, "
        "Product Logo=WvH4pWDGc9LW5pgWnopi-, "
        "Story=6eRQ2ZL7p9qa8wtPulQxK. "
        "API has duplication bugs -- re-upload fresh copies when needed. "
        "CivitAI image generation requires browser OAuth, not API key.",
        ["clouddrive", "mcp", "folders", "upload"]
    ),
    (
        "Three Realms story: 135 characters (90 main + 45 minor), "
        "174 World Info entries, RAG configured with Vector Storage + Summarize. "
        "Files stored in SillyTavern/stories/three-realms/ and "
        "CloudDrive Three Realms Under One Sky folder. "
        "World systems: Dragon Standard currency (unified), "
        "Ascension Scale combat (10-tier), 3 languages plus Rift, "
        "3 religions, 3 hierarchies, 18+ unique items. "
        "Uses free-role MC creation, 1500+ message pacing.",
        ["story", "three-realms", "sillytavern", "world-building"]
    ),
    (
        "CivitAI: API key fcb3eea80ce7e93386179d07cdf7c74e. "
        "Model One obsession v23 (ID 1318945, version 3118448). "
        "Resolution 1216x832 or 832x1216. "
        "Negative prompt: bad figure, incomplete hand, crippled, more than one person. "
        "Generation requires browser OAuth (API key is for reading only). "
        "Animal-features fix for portraits: use Booru tags "
        "1boy, solo, human man plus round human ears, NOT animal ears.",
        ["civitai", "image-generation", "portrait", "config"]
    ),
    (
        "Story preferences: 1000-2000 message pacing. "
        "Free-role MC creation where player defines character per chat. "
        "The Final Chapter: dimension rulers battling in goddess game. "
        "CloudDrive folder ID for Final Chapter: fbuzT4Wo0L_PbTq2G36yV.",
        ["story", "preferences", "final-chapter", "pacing"]
    ),
    (
        "Daryl planning style: ask 2-4 clarifying questions on big architectural "
        "changes UNLESS the user has already given a detailed PRD or said "
        "check whether that is doable -- in that case, just implement. "
        "Daryl prefers decisive execution over prolonged planning "
        "when he has already provided the spec.",
        ["daryl", "workflow", "planning", "preference"]
    ),
    (
        "Daryl workflow: README + commit + push + release in one flow. "
        "Preflight before service-swap: check cloudflared config, pm2 list, "
        "netstat, curl health. Always verify infrastructure before deploying.",
        ["daryl", "workflow", "deployment", "preflight"]
    ),
    (
        "Story creation preferences: Daryl wants majority-female rosters "
        "(65% female, 25% male). Romance options aged 18-25. "
        "Characters should be successors of historical figures, not direct copies. "
        "Comfortable with women in all power roles. "
        "Wants massive stories with 90+ characters.",
        ["story", "preferences", "characters", "roster"]
    ),
    (
        "Daryl documentation style confirmed preference: Plain ASCII ONLY in "
        "any markdown or docs. No emoji, no Unicode box-drawing, no em-dashes "
        "(use hyphens). He explicitly rejected emoji. "
        "Use pipe-tables, setext-style headings, hyphens for separators. "
        "This applies to README files, release notes, GitHub release bodies, "
        "plan docs, and any repo file he reads on screen.",
        ["daryl", "documentation", "style", "ascii"]
    ),
    # === Infrastructure facts from this session ===
    (
        "Machine 1 (HOST) specs: Intel i5-12400 (6 cores, 12 threads), "
        "32GB RAM, Intel UHD 730 (integrated, no CUDA), Windows 11 Home. "
        "Ollama 0.32.15 installed via winget. "
        "Docker Desktop 4.87.0 installed via winget. "
        "Neo4j 5.26.3 runs in Docker with 2GB heap, 1GB pagecache. "
        "Python 3.11.15 via uv. Machine hostname: DESKTOP-GVVMM5A.",
        ["machine", "specs", "host", "infrastructure"]
    ),
    (
        "Daryl-Memories MCP server: Python + FastMCP + Streamable HTTP. "
        "Runs on port 8080. Neo4j on localhost:7687 (bolt). "
        "Ollama on localhost:11434 (native, not Docker). "
        "API key auth via Bearer token. "
        "Models: qwen2.5:3b (extraction) + nomic-embed-text (768-dim embeddings). "
        "GitHub repo: https://github.com/DarylAndrian/Daryl-Memories",
        ["daryl-memories", "mcp", "server", "infrastructure"]
    ),
    (
        "CloudDrive server: Fastify on port 3000, managed by pm2 (clouddrive). "
        "Users: darylll (admin/owner), jeannette (admin), lydia (admin), freddy (admin). "
        "Shared cloud access granted to all users. "
        "Public URL: https://cloud-drive.velagroup.my.id. "
        "API: /api/auth/login for JWT, /api/files/upload for files, "
        "/api/shared-cloud/ for shared folder operations.",
        ["clouddrive", "server", "api", "users"]
    ),
]


async def main():
    settings = Settings()
    engine = MemoryEngine(settings)

    print(f"Pouring {len(MEMORIES)} memories into shared memory system...\n")

    for i, (content, tags) in enumerate(MEMORIES):
        label = tags[0] if tags else "?"
        print(f"  [{i+1}/{len(MEMORIES)}] {label}...", end=" ", flush=True)
        try:
            result = await engine.remember(
                content=content,
                agent_id="hermes-host",
                session_id="memory-migration",
                tags=tags,
            )
            ents = len(result.entities_created)
            merged = len(result.entities_merged)
            facts = result.facts_stored
            confl = len(result.conflicts_detected)
            emb = result.embedding_status
            status = "OK" if emb == "ok" else f"emb-err"
            print(f"ents={ents} merged={merged} facts={facts} conflicts={confl} {status}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. All {len(MEMORIES)} memories poured.")
    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
