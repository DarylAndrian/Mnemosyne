"""GitHub webhook auto-deploy for Mnemosyne.

Same proven pattern as CloudDrive and char-catalogue:
- HMAC-SHA256 signature verification (X-Hub-Signature-256)
- Only 'push' events on the 'main' branch trigger a deploy
- The actual deploy runs in a DETACHED finisher script, because
  `pm2 restart mnemosyne` kills this process mid-flight. The finisher
  pulls, restarts, health-polls, and AUTO-ROLLBACKS if the new code
  fails its health check within the budget.

Data safety: the deploy only touches code files. The memory graph
lives in the Neo4j Docker volume (neo4j_data), which nothing here
can affect.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

log = logging.getLogger("mnemosyne.deploy")

REPO_DIR = Path(__file__).resolve().parent.parent
STATUS_FILE = REPO_DIR / "deploy-status.json"
FINISHER = REPO_DIR / "scripts" / "deploy-finisher.sh"

# IMPORTANT: bare "bash" resolves to Windows' WSL shim (System32\bash.exe)
# in subprocess contexts and fails with "execvpe(/bin/bash) failed".
# Use git-bash explicitly, with a PATH fallback.
_BASH_CANDIDATES = [
    os.environ.get("GIT_BASH"),
    str(Path.home() / "AppData" / "Local" / "hermes" / "git" / "usr" / "bin" / "bash.exe"),
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
]


def _find_bash() -> str:
    for c in _BASH_CANDIDATES:
        if c and Path(c).exists():
            return c
    return "bash"  # last resort


def _read_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"status": "none", "time": None, "commit": None}


async def deploy_status(request: Request) -> JSONResponse:
    return JSONResponse(_read_status())


def _verify_signature(payload: bytes, header: str | None, secret: str) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(expected, header)
    except Exception:  # noqa: BLE001
        return False


async def deploy_webhook(request: Request) -> JSONResponse:
    from .config import get_settings

    secret = get_settings().deploy_webhook_secret
    if not secret:
        return JSONResponse({"error": "deploy webhook not configured"}, status_code=503)

    payload = await request.body()
    sig = request.headers.get("x-hub-signature-256")
    if not _verify_signature(payload, sig, secret):
        return JSONResponse({"error": "invalid signature"}, status_code=401)

    event = request.headers.get("x-github-event", "")
    if event == "ping":
        return JSONResponse({"ok": True, "pong": True})
    if event != "push":
        return JSONResponse({"skipped": True, "reason": f"not a push event ({event})"})

    try:
        body = json.loads(payload.decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "bad payload"}, status_code=400)

    branch = str(body.get("ref", "")).replace("refs/heads/", "")
    if branch != "main":
        return JSONResponse({"skipped": True, "reason": f"branch {branch} not main"})

    commit = (body.get("head_commit") or {}).get("id", "unknown")[:7]
    pusher = (body.get("pusher") or {}).get("name", "unknown")

    # Mark deploying BEFORE spawning - the finisher updates the file later.
    try:
        STATUS_FILE.write_text(
            json.dumps({"status": "deploying", "time": _now(), "commit": commit, "pusher": pusher}),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass

    # Detached finisher survives our own pm2 restart.
    # NOTE: forward slashes -- git-bash's cd chokes on backslash paths,
    # which would leave REPO empty and silently kill the deploy.
    # NOTE: explicit git-bash -- bare "bash" is the WSL shim and fails.
    try:
        subprocess.Popen(
            [_find_bash(), str(FINISHER).replace("\\", "/")],
            cwd=str(REPO_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to spawn deploy finisher: %s", exc)
        return JSONResponse({"error": f"finisher spawn failed: {exc}"}, status_code=500)

    log.info("Deploy started for commit %s (pusher %s)", commit, pusher)
    return JSONResponse({"deploying": True, "commit": commit})


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def mount_deploy(app) -> bool:
    """Register deploy routes. Returns True if the webhook is armed."""
    from .config import get_settings

    routes = [
        Route("/api/deploy/status", deploy_status, methods=["GET", "OPTIONS"]),
        Route("/api/deploy/webhook", deploy_webhook, methods=["POST"]),
    ]
    for r in reversed(routes):
        app.routes.insert(0, r)

    armed = bool(get_settings().deploy_webhook_secret)
    log.info("Deploy webhook %s", "ARMED" if armed else "DISABLED (no DEPLOY_WEBHOOK_SECRET)")
    return armed
