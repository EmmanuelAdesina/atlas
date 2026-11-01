#!/usr/bin/env python3
"""Ledger truth report — what the ledger can and cannot say.

Written against the REAL ledger schema (2026-11-02):

    every line is {"event": "bet_placed"|"bet_settled", "timestamp": ...,
    "record": {bet_id, match_id, market_key, selection, odds, stake_ngn,
               model_p, ...}}

IMPORTANT LIMITS (do not fabricate beyond these):
- the ledger records PLACEMENTS and a few legacy 'bet_settled' events;
  legacy settled records carry NO outcome fields (no result/payout), so
  realized PnL per market is NOT derivable from the ledger alone — the
  book's outcome truth (SportyBet Bet History) lands via the
  settle_history reconcile (Phase B).
- an honest money window still holds: sum(stake) vs the OBSERVED
  balance (heartbeat.json balance_ngn / body of /status).

Usage:
    python3 analyze_ledger.py [LEDGER_JSONL]
                              [--heartbeat /opt/atlas/data/ops/heartbeat.json]
"""
import json
import math
import sys
from collections import Counter, defaultdict

L = len(sys.argv) > 1 and not sys.argv[1].startswith("--") and sys.argv[1] or \
    "/opt/atlas/data/ops/ledger.jsonl"
HB = "/opt/atlas/data/ops/heartbeat.json"
if "--heartbeat" in sys.argv:
    HB = sys.argv[sys.argv.index("--heartbeat") + 1]

placed, settled = [], []
for line in open(L, encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        j = json.loads(line)
    except Exception:
        continue
    r = j.get("record") or {}
    row = dict(
        ts=str(j.get("timestamp"))[:16],
        mk=r.get("market_key"),
        sel=r.get("selection"),
        stake=r.get("stake_ngn") or 0.0,
        odds=r.get("odds"),
        p=r.get("model_p"),
        label=r.get("match_label") or r.get("event_id") or r.get("match_id") or "",
    )
    (settled if j.get("event") == "bet_settled" else placed).append(row)

tot = sum(x["stake"] for x in placed)
print(f"bet_placed: {len(placed)}   total staked: NGN {tot:.0f}")
print(f"bet_settled (legacy rows, outcome fields ABSENT): {len(settled)}")

balance = None
try:
    balance = json.load(open(HB, encoding="utf-8")).get("balance_ngn")
except Exception:
    pass
if balance is not None:
    print(f"observed balance: NGN {balance:.2f}")
    print(f"staked in window: NGN {tot:.0f}; a 0-return-only view would read "
          f"{balance - tot:+.2f} net — the real net needs the settle reconcile")

print("\n=== BY MARKET (n | staked | mean stake | mean claimed p | mean implied p | claimed edge) ===")
agg = defaultdict(lambda: dict(n=0, s=0.0, ps=[], imps=[]))
for x in placed:
    a = agg[x["mk"]]
    a["n"] += 1
    a["s"] += x["stake"]
    if x["odds"]:
        a["imps"].append(1.0 / x["odds"])
    if x["p"]:
        a["ps"].append(x["p"])
for mk, a in sorted(agg.items(), key=lambda kv: -kv[1]["s"]):
    mp = sum(a["ps"]) / len(a["ps"]) if a["ps"] else float('nan')
    mi = sum(a["imps"]) / len(a["imps"]) if a["imps"] else float('nan')
    edge = (mp - mi) if a["ps"] and a["imps"] else float('nan')
    print(f"  {mk:22s} n={a['n']:3d} staked={a['s']:6.0f} meanStake={a['s']/a['n']:5.1f} "
          f"p={mp:.3f} imp={mi:.3f} edge={(edge if not math.isnan(edge) else float('nan')):+.3f}")

print("\n=== SAME-MATCH CORRELATION CLUSTERS (by recorded label/event_id) ===")
clusters = defaultdict(list)
for x in placed:
    clusters[x["label"] or f"(no label) bet_{x['ts']}"].append(x)
for k, xs in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    if len(xs) >= 2:
        print(f"  [{len(xs)} tickets | NGN {sum(x['stake'] for x in xs):.0f}] {k[:64]}")

print("\n=== SETTLED RECORDS (legacy — no outcome fields recorded) ===")
for x in settled:
    print(f"  {x['ts']} {x['mk']:18s} {str(x['sel'])[:16]:16s} stake={x['stake']} odds={x['odds']}")

print("\nNOTE: realized per-market PnL requires the book's outcome truth "
      "(atlas.ops.settle_history reconcile); the ledger's own settle rows "
      "do not carry it.")
