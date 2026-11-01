import json
ledger = []
for line in open('/opt/atlas/data/ops/ledger.jsonl'):
    line = line.strip()
    if line:
        ledger.append(json.loads(line))

placed = [e for e in ledger if e['event'] == 'bet_placed']
settled = [e for e in ledger if e['event'] == 'bet_settled']

print(f"Total bet_placed: {len(placed)}")
print(f"Total bet_settled: {len(settled)}")

# All placed bets details
print("\n=== ALL PLACED BETS ===")
for e in placed:
    r = e['record']
    print(f"{r['placed_at'][:16]} | {r['market_key']:20s} | {r['selection'][:30]:30s} | @{r['odds']:.2f} | stake={r['stake_ngn']:.0f} | p={r.get('model_p','?')}")

# Settled bets
print("\n=== SETTLED BETS ===")
for e in settled:
    r = e['record']
    print(f"{r['placed_at'][:16]} | {r['market_key']:20s} | {r['selection'][:30]:30s} | @{r['odds']:.2f} | stake={r['stake_ngn']:.0f} | pnl={r.get('pnl_ngn',0):+.1f} | status={r.get('status')}")

# PnL
total_pnl = sum(r.get('pnl_ngn',0) for e in settled for r in [e['record']])
print(f"\n=== TOTAL PnL: {total_pnl:+.2f} NGN ===")
print(f"Total staked: {sum(r['stake_ngn'] for e in placed for r in [e['record']]):.2f} NGN")

# Market family analysis
from collections import defaultdict
family_stats = defaultdict(lambda: {'count':0, 'pnl':0.0, 'staked':0.0, 'won':0, 'lost':0})
for e in placed:
    r = e['record']
    mk = r['market_key']
    family_stats[mk]['count'] += 1
    family_stats[mk]['staked'] += r['stake_ngn']
for e in settled:
    r = e['record']
    mk = r['market_key']
    pnl = r.get('pnl_ngn',0)
    family_stats[mk]['pnl'] += pnl
    if pnl > 0:
        family_stats[mk]['won'] += 1
    elif pnl < 0:
        family_stats[mk]['lost'] += 1

print("\n=== MARKET FAMILY ANALYSIS ===")
print(f"{'Market':25s} | {'Bets':>5s} | {'Staked':>8s} | {'PnL':>9s} | {'Won':>4s} | {'Lost':>4s} | {'ROI':>7s}")
for mk, s in sorted(family_stats.items(), key=lambda x: -x[1]['pnl']):
    roi = s['pnl']/s['staked']*100 if s['staked'] > 0 else 0
    print(f"{mk:25s} | {s['count']:5d} | {s['staked']:8.2f} | {s['pnl']:+9.2f} | {s['won']:4d} | {s['lost']:4d} | {roi:+7.1f}%")
