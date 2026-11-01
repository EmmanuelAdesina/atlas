#!/bin/bash
# atlas-deploy-smoke.sh — post-deploy health gate for CD.
# Runs after atlas-apply.sh: everything must come back within the
# window or CD rolls back. Exit 0 = healthy, 1 = rollback needed.
set -u
DEADLINE=$(( $(date +%s) + ${ATLAS_CD_SMOKE_WINDOW:-300} ))
SERVICES="atlas-xvfb atlas-browser atlas-loop"
HB=/opt/atlas/data/ops/heartbeat.json
DASH_URL=${ATLAS_DASH_URL:-http://100.93.175.33:8077/status}

fresh() {
  [ -f "$HB" ] || return 1
  age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
  [ "$age" -lt 150 ]
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  # any unit DOWN in a way restart didn't fix?
  broken=""
  for s in $SERVICES; do
    [ "$(systemctl is-active "$s" 2>/dev/null)" = "active" ] || broken="$broken $s"
  done
  if [ -n "$broken" ]; then
    echo "smoke: down units:$broken — waiting/retrying"
    sleep 12
    continue
  fi
  # daemon liveness: precise check of the extension transport
  if ! curl -s -m 6 http://127.0.0.1:10086/status | grep -q '"extension_connected":true'; then
    echo "smoke: daemon/extension not connected yet"
    sleep 12
    continue
  fi
  # heartbeat freshness (the loop must be cycling)
  if ! fresh; then
    echo "smoke: heartbeat not fresh yet (warm-up after restart is normal for ~3 min)"
    sleep 12
    continue
  fi
  # observer API must respond
  if ! curl -sf -m 6 "$DASH_URL" >/dev/null 2>&1; then
    echo "smoke: dashboard not responding"
    sleep 12
    continue
  fi
  echo "smoke: healthy (units active, extension attached, heartbeat fresh, dashboard answering)"
  exit 0
done

echo "smoke: TIMEOUT waiting for healthy state"
exit 1
