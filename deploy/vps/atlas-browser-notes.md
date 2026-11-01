# atlas-browser unit notes (VPS)

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
