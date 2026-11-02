#!/bin/bash
# atlas-chrome-session-clean.sh — break the chrome crash-restore loop.
#
# Every wedge-kill (SIGTERM under memory squeeze, renderer burn) leaves
# the profile with `exit_type: Crashed`. The next boot then replays the
# WHOLE previous tab stack: Default/Sessions/Tabs_* reached ~1.4 MB
# (dozens of sportybet SPA pages), which instantly re-soaks the 1.6 GB
# box and re-wedges chrome within minutes of every boot.
#
# This ExecStartPre resets that restore state ONLY. Cookies, local
# storage and extension data — the betting login and wallet tab — live
# in separate profile files and are never touched. Runs as the service
# user before every atlas-browser boot.
set -u
D=/opt/atlas/chrome-profile/Default
[ -d "$D" ] || exit 0

python3 - "$D/Preferences" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
except Exception:
    raise SystemExit(0)
changed = False

prof = d.get("profile") or {}
if prof.get("exit_type", "Normal") != "Normal":
    prof["exit_type"] = "Normal"
    d["profile"] = prof
    changed = True

# Explicitly boot to a New Tab: no URL-list replay either.
sess = d.get("session") or {}
if sess.get("restore_on_startup", 5) != 5:
    sess["restore_on_startup"] = 5
    d["session"] = sess
    changed = True

if changed:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f)
PY

# Drop persisted session bundles so a boot starts EMPTY (chrome
# regenerates these on its next clean exit).
rm -f "$D/Sessions/Tabs_"* "$D/Sessions/Session_"* 2>/dev/null
rm -f "$D/Current Session" "$D/Current Tabs" 2>/dev/null
rm -f "$D/Current Session.tmp" "$D/Current Tabs.tmp" 2>/dev/null
exit 0
