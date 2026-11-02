#!/bin/bash
# atlas-apply.sh — Idempotent IaC apply for the Atlas VPS plane.
# The repo copy under /opt/atlas/deploy/vps IS the source of truth.
# Re-running is always safe. Backs up anything it overwrites to
# /opt/atlas/.cd-backup/<stamp>/ (used by CD rollback).
#
# Usage: atlas-apply.sh [--check]
#   --check   report drift only (exit 1 if any), write nothing.
#
# Restart policy: changed service units are restarted, but atlas-loop
# and atlas-browser are guarded — a restart is DEFERRED (marker file)
# while a placement is queued/inflight, because interrupting a live
# Confirm click is worse than a delayed deploy. ATLAS_CD_NEED_LOOP=1
# (set by atlas-cd.sh) forces a loop restart for code-only changes.
set -u

REPO=${ATLAS_REPO:-/opt/atlas}
cd "$REPO" || { echo "apply: cannot cd $REPO"; exit 1; }
CHECK=false
[ "${1:-}" = "--check" ] && CHECK=true
STAMP=$(date -u +%Y%m%d-%H%M%S)
BACKUP_DIR="$REPO/.cd-backup/$STAMP"
DEFER=/var/tmp/atlas-cd.defer
DRIFT=0
CHANGED_UNITS=""

alert() {
  set -a; [ -f "$REPO/.env" ] && . "$REPO/.env"; set +a
  python3 "$REPO/scripts/atlas_alert.py" "$1" "$2" >/dev/null 2>&1 || true
}

hb() { python3 -c "
import json,sys
try:
    h=json.load(open('$REPO/data/ops/heartbeat.json'))
    print(h.get('place_queue_depth',0), len(h.get('place_inflight') or []))
except Exception:
    print(0,0)
" 2>/dev/null; }

placement_busy() {
  set -- $(hb)
  [ "$1" != "0" ] || [ "$2" != "0" ]
}

# src(dst-absolute) pairs, with install metadata
add_file() { REPO_PAIRS+=("$1|$2|$3|$4"); }

REPO_PAIRS=()
add_file "deploy/vps/atlas-loop.service"                        "/etc/systemd/system/atlas-loop.service" root 644
add_file "deploy/vps/units/atlas-xvfb.service"                  "/etc/systemd/system/atlas-xvfb.service" root 644
add_file "deploy/vps/units/atlas-browser.service"               "/etc/systemd/system/atlas-browser.service" root 644
add_file "deploy/vps/units/atlas-watchdog.service"              "/etc/systemd/system/atlas-watchdog.service" root 644
add_file "deploy/vps/units/atlas-watchdog.timer"                "/etc/systemd/system/atlas-watchdog.timer" root 644
add_file "deploy/vps/units/atlas-backup.service"                "/etc/systemd/system/atlas-backup.service" root 644
add_file "deploy/vps/units/atlas-backup.timer"                  "/etc/systemd/system/atlas-backup.timer" root 644
add_file "deploy/vps/units/atlas-docker-prune.service"          "/etc/systemd/system/atlas-docker-prune.service" root 644
add_file "deploy/vps/units/atlas-docker-prune.timer"            "/etc/systemd/system/atlas-docker-prune.timer" root 644
add_file "deploy/vps/units/atlas-browser-recycle.service"       "/etc/systemd/system/atlas-browser-recycle.service" root 644
add_file "deploy/vps/units/atlas-browser-recycle.timer"         "/etc/systemd/system/atlas-browser-recycle.timer" root 644
add_file "deploy/vps/units/atlas-cd.service"                     "/etc/systemd/system/atlas-cd.service" root 644
add_file "deploy/vps/units/atlas-cd.timer"                      "/etc/systemd/system/atlas-cd.timer" root 644
add_file "deploy/vps/units/atlas-settle.service"                "/etc/systemd/system/atlas-settle.service" root 644
add_file "deploy/vps/units/atlas-settle.timer"                  "/etc/systemd/system/atlas-settle.timer" root 644
add_file "deploy/vps/atlas-watchdog.sh"                         "/usr/local/sbin/atlas-watchdog.sh" root 755
add_file "deploy/vps/files/scripts/atlas-backup.sh"             "/usr/local/sbin/atlas-backup.sh" root 755
add_file "deploy/vps/files/scripts/atlas-docker-prune.sh"       "/usr/local/sbin/atlas-docker-prune.sh" root 755
add_file "deploy/vps/files/scripts/atlas-browser-recycle.sh"    "/usr/local/sbin/atlas-browser-recycle.sh" root 755
add_file "deploy/vps/files/scripts/atlas-browser-cdp-preflight.sh" "/usr/local/sbin/atlas-browser-cdp-preflight.sh" root 755
add_file "deploy/vps/files/99-atlas-memory.conf"                "/etc/sysctl.d/99-atlas-memory.conf" root 644
add_file "deploy/vps/files/logrotate-atlas"                     "/etc/logrotate.d/atlas" root 644
add_file "deploy/vps/files/chrome-policy.json"                  "/etc/opt/chrome/policies/managed/kimi-webbridge.json" root 644
add_file "deploy/vps/atlas_alert.py"                            "$REPO/scripts/atlas_alert.py" root 755
add_file "deploy/vps/atlas-apply.sh"                             "/usr/local/sbin/atlas-apply.sh" root 755
add_file "deploy/vps/atlas-cd.sh"                                "/usr/local/sbin/atlas-cd.sh" root 755
add_file "deploy/vps/atlas-deploy-smoke.sh"                      "/usr/local/sbin/atlas-deploy-smoke.sh" root 755

TIMERS="atlas-backup.timer atlas-docker-prune.timer atlas-browser-recycle.timer atlas-cd.timer atlas-settle.timer"

need_sysctl_reload=false
need_daemon_reload=false

for pair in "${REPO_PAIRS[@]}"; do
  IFS='|' read -r src dst owner mode <<<"$pair"
  [ -f "$src" ] || { echo "apply: missing repo file $src"; DRIFT=1; continue; }
  if cmp -s "$src" "$dst" 2>/dev/null; then
    # file identical; ensure mode (cheap, silent)
    cur=$(stat -c %a "$dst" 2>/dev/null)
    if [ "$cur" != "$mode" ]; then
      echo "apply: mode fix $dst ($cur -> $mode)"
      if ! $CHECK; then chmod "$mode" "$dst"; fi
      DRIFT=1
    fi
    continue
  fi
  DRIFT=1
  if $CHECK; then
    echo "check: drift $dst (repo differs from installed)"
    continue
  fi
  echo "apply: updating $dst"
  mkdir -p "$BACKUP_DIR"
  [ -f "$dst" ] && cp -p "$dst" "$BACKUP_DIR/$(basename "$dst")"
  install -D -m "$mode" "$src" "$dst"
  case "$dst" in
    *systemd*) need_daemon_reload=true;;
    /etc/sysctl.d/*) need_sysctl_reload=true;;
  esac
  base=$(basename "$dst")
  case "$base" in
    atlas-*.service|atlas-*.timer) CHANGED_UNITS="$CHANGED_UNITS $base";;
  esac
done

if $CHECK; then
  [ "$DRIFT" = "0" ] && echo "check: clean — installed config matches repo"
  exit $DRIFT
fi

if [ "$need_sysctl_reload" = "true" ]; then
  sysctl --system >/dev/null 2>&1 || true
  echo "apply: sysctl reloaded"
fi

if [ "$need_daemon_reload" = "true" ]; then
  systemctl daemon-reload
  for t in $TIMERS; do
    systemctl enable --now "$t" >/dev/null 2>&1 || true
  done
  echo "apply: daemon reloaded, timers ensured"
fi

# restart changed units, guards on the two money-path units
restart_unit() {
  local u="$1"
  case "$u" in
    atlas-loop.service)
      if ! systemctl is-enabled atlas-loop >/dev/null 2>&1; then
        echo "apply: $u changed but unit is disabled/gated — skipping start (operator gate owns it)"; return 0
      fi
      if [ ! -f "${ATLAS_CD_ARM_FLAG:-/opt/atlas/live-armed}" ]; then
        echo "apply: $u changed but arm flag absent — skipping start"; return 0
      fi
      if placement_busy && [ "${ATLAS_CD_FORCE_RESTART:-0}" != "1" ]; then
        echo "apply: $u restart DEFERRED (placement busy)"
        touch "$DEFER"
        return 0
      fi
      systemctl restart atlas-loop
      echo "apply: restarted atlas-loop"
      [ -f "$DEFER" ] && rm -f "$DEFER"
      ;;
    atlas-browser.service)
      if placement_busy && [ "${ATLAS_CD_FORCE_RESTART:-0}" != "1" ]; then
        echo "apply: $u restart DEFERRED (placement busy)"
        touch "$DEFER"
        return 0
      fi
      systemctl restart atlas-browser
      echo "apply: restarted atlas-browser"
      ;;
    atlas-cd.service)
      systemctl restart "$u" 2>/dev/null || true
      echo "apply: restarted $u"
      ;;
    atlas-xvfb.service)
      # restarting X requires browser restart to re-attach
      if placement_busy; then touch "$DEFER"; echo "apply: $u restart DEFERRED (placement busy)"; return 0; fi
      systemctl restart atlas-xvfb && systemctl restart atlas-browser && echo "apply: restarted xvfb+browser"
      ;;
    *.timer)
      # timer unit file change only needs daemon-reload, already done
      ;;
    *)
      systemctl restart "${u%.service}" 2>/dev/null || true
      ;;
  esac
}

for u in $CHANGED_UNITS; do
  restart_unit "$u"
done

# Code-only deploys do not change any unit file, but the running loop is
# still serving the old module code from memory — restart it to actually
# deploy (ATLAS_CD_NEED_LOOP is set by atlas-cd.sh when atlas/ tests/
# pyproject/requirements changed). Same guards as a unit restart: armed
# + not placement-busy; defer otherwise.
if [ "${ATLAS_CD_NEED_LOOP:-0}" = "1" ] && [ -f "$REPO/live-armed" ]; then
  if placement_busy && [ "${ATLAS_CD_FORCE_RESTART:-0}" != "1" ]; then
    echo "apply: code-deploy atlas-loop restart DEFERRED (placement busy)"
    touch "$DEFER"
  elif systemctl is-active --quiet atlas-loop; then
    systemctl restart atlas-loop
    echo "apply: code deploy — restarted atlas-loop"
    [ -f "$DEFER" ] && rm -f "$DEFER"
  elif [ ! -f "$DEFER" ]; then
    echo "apply: code deploy — atlas-loop not active, not starting it (operator gate owns that)"
  fi
fi

# if a previous deploy deferred us and nothing is busy now, drain it
if [ -f "$DEFER" ]; then
  if ! placement_busy; then
    echo "apply: draining previous deferred restart"
    rm -f "$DEFER"
    if [ -f /opt/atlas/live-armed ] && systemctl is-active --quiet atlas-loop; then
      systemctl restart atlas-loop && echo "apply: deferred atlas-loop restart applied"
    fi
  fi
fi

echo "apply: done (stamp $STAMP)"
