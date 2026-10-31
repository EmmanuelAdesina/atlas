#!/bin/bash
# ATLAS wedge watchdog: Chrome dead / daemon unreachable / extension lost.
# Self-heals by restarting the failing plane and EMAILS the operator.
# Runs as root (systemctl needs it) every 10 min via atlas-watchdog.timer.
set -u
STATE=/var/tmp/atlas-watchdog.state
PREV=""; [ -f "$STATE" ] && PREV=$(cat "$STATE")
NOW=$(date +%s)

alert() {
  logger -t atlas-watchdog "ALERT: $1"
  /opt/atlas/.venv/bin/python /opt/atlas/scripts/atlas_alert.py \
    "ATLAS watchdog: $1" \
    "host: $(hostname) | at: $(date -u +%FT%TZ) | action: $2" 2>/dev/null
}

# 1. Chrome alive?
if ! pgrep -f "user-data-dir=/opt/atlas/chrome-profile" >/dev/null; then
  alert "chrome not running" "restarting atlas-browser.service"
  systemctl restart atlas-browser
  echo "restarted-browser $NOW" > "$STATE"
  exit 0
fi

# 2. Daemon reachable?
WD=$(/home/reliastra/.kimi-webbridge/bin/kimi-webbridge status 2>/dev/null || echo '{}')
if ! echo "$WD" | grep -q '"running":true'; then
  alert "webbridge daemon unreachable" "restarting kimi-webbridge.service"
  systemctl restart kimi-webbridge
  echo "restarted-daemon $NOW" > "$STATE"
  exit 0
fi

# 3. Extension attached? (two consecutive ticks -> restart the browser)
if echo "$WD" | grep -q '"extension_connected":false'; then
  if [ "$PREV" = "extGone" ]; then
    alert "extension disconnected twice" "restarting atlas-browser.service"
    systemctl restart atlas-browser
    rm -f "$STATE"
  else
    echo "extGone" > "$STATE"
  fi
  exit 0
fi

# Healthy: clear any alert latch, notify once on recovery.
if [ -n "$PREV" ]; then
  logger -t atlas-watchdog "recovered: chrome alive, daemon up, extension connected"
  rm -f "$STATE"
fi
