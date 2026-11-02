#!/bin/bash
# atlas-browser-cdp-preflight.sh — health gate BEFORE a CDP-based job
# (the hourly settle scrape). Makes sure the DevTools endpoint answers
# verdict-style: never let the fresh scrape race an unhealthy plane.
#
#   exit 0  -> DevTools healthy  (run the scrape)
#   exit 42 -> healed by restart, but scariness remains / or placement is
#              inflight (recycle is NOT safe then) — the scrape run is a
#              loud skip, the STALE line in reconcile stays honest.
#
# Runs as root (oneshot intent semantics); the queue/inflight guard
# mirrors atlas-browser-recycle.sh: never recycle a live placement.
set -u
HB=/opt/atlas/data/ops/heartbeat.json

cdp_ok() {
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:9222/json/version)" = "200" ]
}

# placement active -> never recycle; skip loudly instead
Q=$(python3 -c "import json;h=json.load(open(\"$HB\"));print(h.get(\"place_queue_depth\",0))" 2>/dev/null)
I=$(python3 -c "import json;h=json.load(open(\"$HB\"));print(len(h.get(\"place_inflight\") or []))" 2>/dev/null)
if [ "${Q:-0}" != "0" ] || [ "${I:-0}" != "0" ]; then
  logger -t atlas-cdp-preflight "placement active (queue=$Q inflight=$I) — refusing preflight recycle"
  exit 42
fi

if cdp_ok; then
  exit 0
fi

logger -t atlas-cdp-preflight "devtools wedged — recycling atlas-browser before scrape"
systemctl restart atlas-browser

# Chrome takes ~40s to fully init under this box's memory pressure (the
# fixed 14s sleep probed mid-boot). Poll up to 60s instead of hoping.
RECOVERED=0
for _ in $(seq 1 12); do
  sleep 5
  if cdp_ok; then RECOVERED=1; break; fi
done

if [ "$RECOVERED" = "1" ]; then
  logger -t atlas-cdp-preflight "devtools recovered after recycle"
  exit 0
fi

logger -t atlas-cdp-preflight "devtools STILL unreachable after recycle — scrape will skip loudly"
exit 42
