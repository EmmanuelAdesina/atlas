# atlas-browser unit notes (VPS)

## 2026-11-02 — refresh of the wedge mechanism; unit retuned; capacity verdict

Live-money session was dark all afternoon: capture failing, `priced_selections: 0`
for ~10 hours while the operator saw 28-35 live matches on a healthy desktop.
Probes (raw CDP over 9222, /status, journal, cgroup counters) resolved the chain:

1. **The unit had drifted back to `MemoryHigh=650M`** (the 2026-11-01 note says
   `MemoryHigh` MUST stay unset; forcing internal reclaim wedges Chrome). Fixed:
   removed again; `MemoryMax=900M` retained as the prod-protecting OOM magnet.
2. **`--force-renderer-accessibility` was the crown jewel of the churn**: the a11y
   tree of the day's board is enormous, and `capture.read_markets_dom()` (JS) is
   already the PRIMARY path; the a11y snapshot is only a fallback that the code
   itself documents as failing with daemon 502s on the heavy board. Removed.
3. Added `--process-per-site` (one renderer per site instead of per frame);
   genuine tab hoards (6+ sporty instances) shared a single core.
4. **Still not enough.** With the plane clean, cgroup `cpu.stat` showed ~96%
   SYSTEM time, chromium anon resident ~2.5 MB (heap fully swapped), main
   thread in D-state, /json reachable in narrow windows only, and evaluate on
   any sporty page timing out at 40 s while the same evaluate on about:blank
   returns instantly. Kernel page-fault/swap churn on a 1-vCPU box is the
   bottleneck; no Chrome flag fixes that.
5. **The load side grew**: Sporty's SPA shipped a new Single View UI with much
   deeper live feeds on 2026-11-02 (35+ live matches vs the old Multi View
   board), so the page itself is heavier than on the 2026-11-01 cutover day.

Operational backups: `/etc/systemd/system/atlas-browser.service.bak-1102`
(the pre-tune unit). Raw-CDP probe: `/tmp/cdp_eval.py` (stdlib-only websocket
client; bypasses the extension when the daemon flap makes tool calls wedge).

Verdict unchanged but now proven at the counter level: the stack needs its own
RAM (4 GB+ upgrade, or split RELIASTRA and Atlas onto separate boxes). Until
then the plane can complete captures only in calm windows (overnight pattern).


## 2026-11-01 — capture-stall deep fix (cutover day)

The armed loop was stuck in `cycle 1: capture` for 300s+ at Sunday midday
load. Root cause chain, proven with kernel counters, and what fixed each:

1. **swap thrash**: `vm.swappiness=100` (distro default on this build) had the
   kernel eagerly paging into zram; zstd compress/decompress + disk-backed
   spill (`swapfile2` 2G + `/dev/sda3` partition) charged the single vCPU with
   ~86% *system* time. Fixed: `vm.swappiness=20`, persisted at
   `/etc/sysctl.d/99-atlas-memory.conf`.
2. **CPU starvation**: `CPUWeight=60` made Chrome lose the core to prod's
   dockerd/celery. Fixed: `CPUWeight=120` on `atlas-browser.service` only.
3. **synchronous reclaim**: a `MemoryHigh=700M` I set mid-diagnosis forced
   reclaim inside Chrome. Removed (`MemoryHigh=infinity`); only
   `MemoryMax=900M` remains — the OOM magnet so prod never dies.
4. **renderer working-set bloat**: SportyBet's live board accumulates renderer
   memory until Chrome cannot even spawn new renderers
   (`zygote_host_impl_linux` fork failures). Fixes, layered:
   - `--js-flags=--max-old-space-size=384` on ExecStart: renderer GCs at a
     capped heap instead of growing into swap.
   - image/plugin/notification loading blocked via Chrome managed policy
     (`/etc/opt/chrome/policies/managed/kimi-webbridge.json`, keep the
     `ExtensionInstallForcelist` key — it force-installs the WebBridge ext).
   - `atlas-browser-recycle.timer` (02,06,10,14,18,22:15 UTC): guarded recycle
     via `/usr/local/sbin/atlas-browser-recycle.sh` — restarts the browser
     ONLY if the heartbeat shows no queued/inflight placement, then emails.
     Post-recycle the loop re-warms through cycle 1 in ~2–3 min (observed).
5. **set-property live values** (`CPUWeight`, `MemoryHigh`) only apply to the
   RUNNING unit; `daemon-reload` replays them on restart. Verify both paths.

Return-to-shape evidence after the fix stack: first sample CPU 86.6%sy →
83%us/17%sy; swap-in 20–35 MB/s → 1.4 MB/s; cycles 1→5 advancing with
`ready: True`, live balance read (NGN 868.18), submission [ENABLED].
