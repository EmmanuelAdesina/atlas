#!/bin/bash
# ATLAS wedge watchdog v2: Chrome dead / daemon unreachable / extension wedged.
# v2 fixes: probe the daemon over HTTP (the old $HOME-dependent CLI status
# lied when run as root), and treat a twice-in-a-row disconnected extension
# as the known Chrome-wedge (deploy/README.md #operating-notes) whose cure is
# a Chrome restart, not another daemon restart.
# Runs as root every 10 min via atlas-watchdog.timer.
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

# 2. Chrome DevTools (CDP) healthy? A wedged debug server passes the
# process check (pgrep sees chrome) and often the daemon check (the
# extension keeps answering), but it starves BOTH the loop's capture and
# the hourly reconcile (2026-11-02: three wedges inside 90 min while the
# box sat at ~80 MB free). Probe 9222 directly; twice in a row ->
# restart the browser plane, same cure as the extension wedge.
CDP=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:9222/json/version)
if [ "$CDP" != "200" ]; then
  if [ "$PREV" = "cdpGone" ]; then
    alert "chrome devtools wedged twice" "restarting atlas-browser.service"
    systemctl restart atlas-browser
    rm -f "$STATE"
  else
    echo "cdpGone" > "$STATE"
    logger -t atlas-watchdog "cdp probe failed (http=$CDP); armed - second miss recycles the browser"
  fi
  exit 0
fi

# 3. Daemon reachable? (HTTP probe on the fixed port, no $HOME involved)
WD=$(curl -s -m 8 http://127.0.0.1:10086/status || echo '{}')
if ! echo "$WD" | grep -q '"running":true'; then
  alert "webbridge daemon unreachable" "restarting kimi-webbridge.service"
  systemctl restart kimi-webbridge
  echo "restarted-daemon $NOW" > "$STATE"
  exit 0
fi

# 3. Extension attached? A single miss waits (it self-reconnects); twice in
# a row means the known Chrome wedge -> restart the browser plane.
if echo "$WD" | grep -q '"extension_connected":false'; then
  if [ "$PREV" = "extGone" ]; then
    alert "extension wedged twice" "restarting atlas-browser.service"
    systemctl restart atlas-browser
    rm -f "$STATE"
  else
    echo "extGone" > "$STATE"
  fi
  exit 0
fi

# Healthy: clear any alert latch, note recovery once.
if [ -n "$PREV" ]; then
  logger -t atlas-watchdog "recovered: chrome alive, daemon up, extension connected"
  rm -f "$STATE"
fi
