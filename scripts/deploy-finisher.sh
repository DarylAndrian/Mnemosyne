#!/usr/bin/env bash
# Mnemosyne deploy finisher - spawned detached by the webhook handler.
# Sequence: git pull -> pm2 restart -> health poll -> AUTO-ROLLBACK on failure.
# Writes deploy-status.json at every stage (read by GET /api/deploy/status).

set -u
REPO="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"
if [ -z "$REPO" ]; then
  echo "FATAL: cannot resolve repo dir from \$0=$0" >&2
  exit 1
fi
STATUS="$REPO/deploy-status.json"
HEALTH="http://localhost:8080/health"
LOG="$REPO/logs/deploy.log"
mkdir -p "$REPO/logs"
echo "[$(date '+%F %T')] finisher LAUNCHED (pid $$)" >> "$LOG"

write_status() {
  # $1=status $2=commit $3=extra message
  printf '{"status":"%s","time":"%s","commit":"%s","detail":"%s"}' \
    "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" "$3" > "$STATUS"
  echo "[$(date '+%F %T')] $1: $3" >> "$LOG"
}

# Locate pm2 (detached children may inherit a sparse PATH)
PM2_BIN="$(command -v pm2 || echo "/c/Users/hokishop jayajaya/AppData/Local/hermes/node/pm2")"
if [ ! -x "$PM2_BIN" ] && [ ! -f "$PM2_BIN" ]; then
  write_status "failed" "?" "pm2 not found"
  exit 1
fi

cd "$REPO" || { write_status "failed" "?" "cd $REPO failed"; exit 1; }

OLD_COMMIT="$(git rev-parse HEAD)"
NEW_COMMIT="${OLD_COMMIT:0:7}"
write_status "deploying" "$NEW_COMMIT" "pulling"

# 1. Pull
if ! git pull origin main >> "$LOG" 2>&1; then
  write_status "failed" "${OLD_COMMIT:0:7}" "git pull failed"
  exit 1
fi
NEW_COMMIT="$(git rev-parse --short=7 HEAD)"

# 2. Restart the service (ecosystem config keeps env clean)
if ! "$PM2_BIN" restart mnemosyne --update-env >> "$LOG" 2>&1; then
  write_status "failed" "$NEW_COMMIT" "pm2 restart failed - rolling back"
  git reset --hard "$OLD_COMMIT" >> "$LOG" 2>&1
  "$PM2_BIN" restart mnemosyne --update-env >> "$LOG" 2>&1
  exit 1
fi

# 3. Health poll (budget 120s; uvicorn + schema checks need a few seconds)
OK=0
for i in $(seq 1 40); do
  sleep 3
  BODY="$(curl -s --max-time 5 "$HEALTH" 2>/dev/null)"
  if echo "$BODY" | grep -q '"ok":true'; then
    OK=1
    break
  fi
done

if [ "$OK" = "1" ]; then
  write_status "success" "$NEW_COMMIT" "health ok"
  echo "[$(date '+%F %T')] Deploy SUCCESS: $NEW_COMMIT" >> "$LOG"
  exit 0
fi

# 4. Auto-rollback
write_status "rolling_back" "$NEW_COMMIT" "health check failed after 120s"
git reset --hard "$OLD_COMMIT" >> "$LOG" 2>&1
"$PM2_BIN" restart mnemosyne --update-env >> "$LOG" 2>&1

OK=0
for i in $(seq 1 20); do
  sleep 3
  BODY="$(curl -s --max-time 5 "$HEALTH" 2>/dev/null)"
  if echo "$BODY" | grep -q '"ok":true'; then
    OK=1
    break
  fi
done

if [ "$OK" = "1" ]; then
  write_status "rolled_back" "${OLD_COMMIT:0:7}" "reverted to previous commit, health ok"
else
  write_status "failed" "${OLD_COMMIT:0:7}" "rollback done but health still failing - MANUAL ATTENTION NEEDED"
fi
exit 1
