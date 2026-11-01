#!/bin/bash
# atlas-deploy-smoke.sh — post-deploy health gate for CD.
# Runs after atlas-apply.sh: everything must come back within the
# window or CD rolls back. Exit 0 = healthy, 1 = rollback needed.
#
# MONEY-GATE AWARE (2026-11-02). The loop unit only starts when
# /opt/atlas/live-armed exists (ConditionPathExists in
# atlas-loop.service). When the operator has DISARMED money on purpose,
# atlas-loop is EXPECTED to stay down, so the smoke must not require its
# liveness or heartbeat freshness. A smoke that insists a disarmed loop
# be cycling can never pass, and CD then rolls every deploy backward
# into a stale lastgood — 2026-11-02 00:20/00:35, two wall-less
# rollbacks to a pre-campaign commit came from exactly this shape.
set -u
DEADLINE=$(( $(date +%s) + ${ATLAS_CD_SMOKE_WINDOW:-300} ))
HB=/opt/atlas/data/ops/heartbeat.json
ARM_FLAG=${ATLAS_CD_ARM_FLAG:-/opt/atlas/live-armed}

# The dashboard may bind a public interface (ATLAS_HOST from the units'
# EnvironmentFile) rather than 127.0.0.1, so the probe must resolve the
# SAME address the deployment itself uses — probing loopback on a box
# that only binds a public IP probes a socket that is not there.
DASH_HOST=$(grep -h '^ATLAS_HOST=' /opt/atlas/.env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d ' \r')
DASH_PORT=$(grep -h '^ATLAS_PORT=' /opt/atlas/.env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d ' \r')
DASH_URL=${ATLAS_DASH_URL:-"http://${DASH_HOST:-127.0.0.1}:${DASH_PORT:-8077}/status"}

fresh() {
  [ -f "$HB" ] || return 1
  age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
  [ "$age" -lt 150 ]
}

# The control API must be up AND answering the truth-bearing contract:
# the computed /status keys, not a mirror of anything.
status_ok() {
  curl -s -m 8 "$DASH_URL" | python3 -c '
import json, sys
try:
    s = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
for k in ("loop_alive", "arm_flag_present", "campaign_frozen", "submission_basis"):
    if k not in s:
        raise SystemExit(1)
' 2>/dev/null
}

if [ -f "$ARM_FLAG" ]; then
  GATE="armed"
  # Armed: the loop must be cycling — full gate, unchanged.
  SERVICES="atlas-xvfb atlas-browser atlas-loop"
else
  GATE="disarmed"
  # Disarmed money: the operator gate owns the loop; it stays down BY
  # DESIGN. xvfb + browser + the extension are still required (the
  # capture substrate and its liveness), and the dashboard must answer —
  # but the loop and its heartbeat are excluded from the gate.
  SERVICES="atlas-xvfb atlas-browser"
fi
echo "smoke: gate=$GATE — units required:$SERVICES; dashboard $DASH_URL"

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
  # heartbeat freshness ONLY under the armed gate: on a disarmed box the
  # last heartbeat of the stopped loop is deliberately stale, and that
  # stale record is the TRUTH, not a failure.
  if [ "$GATE" = "armed" ] && ! fresh; then
    echo "smoke: heartbeat not fresh yet (warm-up after restart is normal for ~3 min)"
    sleep 12
    continue
  fi
  if ! status_ok; then
    echo "smoke: control dashboard not healthy yet ($DASH_URL)"
    sleep 12
    continue
  fi
  echo "smoke: healthy (gate=$GATE: units active, extension attached, dashboard answering)"
  exit 0
done

echo "smoke: TIMEOUT waiting for healthy state"
exit 1
