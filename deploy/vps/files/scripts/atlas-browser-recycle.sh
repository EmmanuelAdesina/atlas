#!/bin/bash
# Recycle the Atlas browser BEFORE its renderer working set bloats into swap
# thrash (proven failure mode: 45s evaluate timeouts + zygote fork failures).
# Guard: only recycle when no placement is queued/legal — never interrupt money.
set -u
HB=/opt/atlas/data/ops/heartbeat.json
if [ ! -f "$HB" ]; then logger -t atlas-recycle "no heartbeat, skip"; exit 0; fi
Q=$(python3 -c "import json;h=json.load(open(\"$HB\"));print(h.get(\"place_queue_depth\",0))" 2>/dev/null)
I=$(python3 -c "import json;h=json.load(open(\"$HB\"));print(len(h.get(\"place_inflight\") or []))" 2>/dev/null)
if [ "${Q:-0}" != "0" ] || [ "${I:-0}" != "0" ]; then
  logger -t atlas-recycle "placement active (queue=$Q inflight=$I) — recycle deferred"
  exit 0
fi
if [ "${ATLAS_RECYCLE_DRY_RUN:-0}" = "1" ]; then logger -t atlas-recycle "dry-run ok (queue=$Q inflight=$I)"; exit 0; fi
logger -t atlas-recycle "recycling atlas-browser (queue=$Q inflight=$I)"
systemctl restart atlas-browser
/opt/atlas/.venv/bin/python /opt/atlas/scripts/atlas_alert.py \
  "ATLAS browser recycled (scheduled)" \
  "Chrome renderer recycled to prevent swap-thrash bloat. No placement was inflight." 2>/dev/null
