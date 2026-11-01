#!/bin/bash
# atlas-cd.sh — Pull-based continuous deployment.
# Self-hosted on the VPS (via atlas-cd.timer, every 15 min). The VPS pulls
# origin/main; when the repo moves (or installed config drifts from the
# repo), it applies via atlas-apply.sh and health-gates the result.
#
# Failure semantics (NEVER silent):
#   fetch/pull fail      -> email + non-zero exit
#   dirty tree           -> email, refuse to deploy
#   apply smoke failure  -> rollback to the previous known-good state,
#                           restore config backups, email CRITICAL
#   success              -> email once per actual deployment
set -u

REPO=${ATLAS_REPO:-/opt/atlas}
TAG=atlas-cd
DEFER=/var/tmp/atlas-cd.defer
DRY_RUN=false
[ "${ATLAS_CD_DRY_RUN:-0}" = "1" ] && DRY_RUN=true

log() { logger -t "$TAG" "$*"; echo "$TAG: $*"; }

alert() {
  set -a; [ -f "$REPO/.env" ] && . "$REPO/.env"; set +a
  python3 "$REPO/scripts/atlas_alert.py" "$1" "$2" >/dev/null 2>&1 || true
}

g() { cd "$REPO" && sudo -u reliastra git "$@"; }

# MONEY FLOOR (2026-11-02). A rollback target that predates the
# fail-closed campaign verdict and the flat NGN-10 unit cap would retire
# the live-money refusal walls while the operator still believes walls
# are deployed — 2026-11-02 00:20/00:35, a smoke timeout (the old smoke
# required the loop to be RUNNING while money was deliberately
# disarmed) rolled the tree back to 99f3954, several walls earlier in
# history. A backward move past a money wall is never a deploy
# judgment; it needs the operator's explicit bypass.
money_walls_present() {
  g show "$1":atlas/ops/campaign.py 2>/dev/null | grep -q UNIT_STAKE_NGN \
    && g show "$1":atlas/ops/campaign.py 2>/dev/null | grep -q '"proven"'
}

# ---- fetch -----------------------------------------------------------------
if ! g fetch -q origin main; then
  log "git fetch origin/main FAILED (exit $?)"
  alert "ATLAS CD: git fetch failed" \
    "Could not fetch origin/main on $(hostname) at $(date -u +%FT%TZ). Deployment is stalled — investigate network/SSH."
  exit 1
fi

LOCAL=$(g rev-parse HEAD) || exit 1
REMOTE=$(g rev-parse origin/main) || exit 1

# ---- dirty tree guard ------------------------------------------------------
if ! g diff --quiet HEAD -- 2>/dev/null || [ -n "$(g status --porcelain 2>/dev/null | head -5)" ]; then
  log "working tree dirty — refusing deploy (operator intervention)
$(g status --porcelain | head -5)"
  alert "ATLAS CD: dirty tree, deploy refused" \
    "/opt/atlas has uncommitted local changes. The CD will not deploy over them. Commit or clean them."
  exit 0
fi

# ---- no-op checks ----------------------------------------------------------
if [ "$LOCAL" = "$REMOTE" ] && [ -f "$DEFER" ]; then
  # drain a previously deferred restart
  if ! /usr/local/sbin/atlas-apply.sh --check >/dev/null 2>&1; then
    log "deferred marker + repo drift — running apply to drain"
  else
    exit 0
  fi
elif [ "$LOCAL" = "$REMOTE" ]; then
  # same commit: reconcile config drift only
  if /usr/local/sbin/atlas-apply.sh --check >/dev/null 2>&1; then
    log "up to date, config clean"
    exit 0
  fi
  log "same commit, but installed config drifted from repo — reconciling"
  DRIFT_FIX=1
fi

if $DRY_RUN; then
  log "dry-run: would deploy $REMOTE"
  exit 0
fi

LASTGOOD=$(cat "$REPO/.cd-lastgood" 2>/dev/null || echo "")
CODE_CHG=$(g diff --name-only "$LOCAL" "$REMOTE" -- atlas/ tests/ pyproject.toml requirements.txt 2>/dev/null | head -5)
if [ -n "$CODE_CHG" ]; then
  export ATLAS_CD_NEED_LOOP=1
fi

log "deploying $REMOTE (from $LOCAL)"

# ---- pull ------------------------------------------------------------------
if ! g pull --ff-only -q origin main; then
  log "git pull FAILED (exit $?)"
  alert "ATLAS CD: git pull failed" \
    "Could not fast-forward /opt/atlas to origin/main ($REMOTE). Investigate locally: $(g log --oneline -1)"
  exit 1
fi

# ---- apply + smoke ---------------------------------------------------------
if ! /usr/local/sbin/atlas-apply.sh; then
  APPLY_RC=$?
  log "atlas-apply.sh EXIT $APPLY_RC"
  if [ -n "$LASTGOOD" ] && money_walls_present "$LASTGOOD"; then
    if g reset --hard "$LASTGOOD" >/dev/null 2>&1; then
      log "rolled back to $LASTGOOD"
    fi
    alert "ATLAS CD: apply failed, rollback executed" \
      "apply.sh exited $APPLY_RC; repo reset to last-known-good $LASTGOOD. Check atlas-cd journal for the step that failed."
  else
    log "MONEY FLOOR: rollback target ${LASTGOOD:-unset} predates the fail-closed campaign + flat-unit walls — refusing rollback; /opt/atlas stays on $REMOTE"
    alert "ATLAS CD: apply failed — rollback REFUSED (money floor)" \
      "apply.sh exited $APPLY_RC. .cd-lastgood (${LASTGOOD:-unset}) predates the money walls; refused to roll /opt/atlas backward past them. The tree stays on $REMOTE — money stays frozen at the walls regardless; investigate manually."
  fi
  exit 1
fi

if ! timeout "${ATLAS_CD_SMOKE_S:-330}" /usr/local/sbin/atlas-deploy-smoke.sh; then
  RC=$?
  log "SMOKE FAILED (exit $RC)"
  if [ -n "$LASTGOOD" ] && money_walls_present "$LASTGOOD"; then
    if g reset --hard "$LASTGOOD" >/dev/null 2>&1; then
      log "code rolled back to $LASTGOOD"
      # restore unit/config backups taken by apply
      LATEST_BK=$(ls -1dt "$REPO"/.cd-backup/*/ 2>/dev/null | head -1)
      if [ -n "$LATEST_BK" ]; then
        cp -p "$LATEST_BK"* /etc/systemd/system/ 2>/dev/null || true
        cp -p "$LATEST_BK"* /usr/local/sbin/ 2>/dev/null || true
        systemctl daemon-reload 2>/dev/null || true
        # MONEY GATE: never (re)start the trading loop when the operator
        # has disarmed money — the rollback path honors the same gate as
        # atlas-apply.sh's restart guards (2026-11-02: this rollback
        # used to restart atlas-loop unguarded; only the unit's
        # ConditionPathExists stopped an armed start into restored old
        # config).
        if [ -f "${ATLAS_CD_ARM_FLAG:-/opt/atlas/live-armed}" ]; then
          systemctl restart atlas-loop 2>/dev/null || true
        else
          log "config restored; atlas-loop stays down (arm flag absent — the operator gate owns it)"
        fi
        systemctl restart atlas-dashboard 2>/dev/null || true
        log "restored system config from $LATEST_BK"
      fi
    fi
    alert "ATLAS CD: deploy SMOKE FAILED — ROLLED BACK" \
      "Commit $REMOTE failed health checks (exit $RC); repo and config restored to $LASTGOOD. Atlas keeps running the previous version. Investigate before re-merging."
  else
    log "MONEY FLOOR: rollback target ${LASTGOOD:-unset} predates the fail-closed campaign + flat-unit walls — refusing rollback; keeping $REMOTE"
    alert "ATLAS CD: deploy SMOKE FAILED — rollback REFUSED (money floor)" \
      "Commit $REMOTE failed health checks (exit $RC). .cd-lastgood (${LASTGOOD:-unset}) predates the money walls; refused to roll /opt/atlas backward past them. The tree stays on $REMOTE — money is still frozen at the walls; verify the new tree's health manually."
  fi
  exit 1
fi

echo "$REMOTE" > "$REPO/.cd-lastgood"
log "deploy OK: now at $REMOTE"
alert "ATLAS CD: deployed $(g log --oneline -1 | head -c 60)" \
  "Deployed via CD at $(date -u +%FT%TZ). Health checks passed. This is an informational notice — no action needed."
exit 0
