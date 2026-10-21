import numpy as np, math, json, sys
sys.path.insert(0, r'C:\Quant\atlas\.playwright-mcp')
from deep_core import *

d, legs, unmapped = load_legs()
print(f"board: {len(d['markets'])} market instances")
print(f"mappable legs: {len(legs)}")
uniq = {}
for mid, nm, ds in unmapped:
    uniq.setdefault((mid, nm), 0)
    uniq[(mid, nm)] += 1
print(f"\nunmapped market families ({len(uniq)}):")
for (mid, nm), c in sorted(uniq.items(), key=lambda x: x[0][0]):
    print(f"  id={mid:>7}  n={c:>2}  {nm}")

P = np.array([l['p'] for l in legs])
ODDS = np.array([l['odds'] for l in legs])
print("\npayout matrix:", P.shape, " atoms:", NA)

# sanity: sum of payout mass per leg should be <= odds (i.e. E[payout] <= o)
mx = P.max(axis=1)
print("legs where payout multiple exceeds odds (should be 0):", int((mx > ODDS + 1e-9).sum()))

# ---------------------------------------------------------------- model fit
def poisson_pmf(lams):
    out = []
    for l in lams:
        k = np.arange(0, 20)
        p = np.exp(-l) * l ** k / np.array([math.factorial(int(x)) for x in k], dtype=float)
        out.append(p)
    return out

def model_probs(lam_h, lam_a):
    ph = np.array([math.exp(-lam_h) * lam_h ** k / math.factorial(k) for k in range(HMAX + 1)])
    pa = np.array([math.exp(-lam_a) * lam_a ** k / math.factorial(k) for k in range(HMAX + 1)])
    # cap 8 -> 8-or-more: lump tail into last bucket
    def cap(p):
        q = p.copy()
        q[-1] = sum(p[HMAX:])
        return q
    ph, pa = cap(ph), cap(pa)
    M = np.outer(ph, pa)
    return M / M.sum()

# Shin de-vig for a set of legs (partition assumption)
def shin(odds):
    q = 1.0 / np.asarray(odds, dtype=float)
    B = q.sum()
    def P(z): return (np.sqrt(z * z + 4 * (1 - z) * q * q / B) - z) / (2 * (1 - z))
    lo, hi = 0.0, 0.95
    if sum(P(0.0)) < 1: return q / B, 0.0, B
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(P(mid)) > 1: lo = mid
        else: hi = mid
    z = (lo + hi) / 2
    return P(z), z, B

def market_of(mid):
    """group legs by (mid, mdesc)"""
    g = {}
    for l in legs:
        g.setdefault((l['mid'], l['mdesc']), []).append(l)
    return g

G = market_of(None)
print(f"\ndistinct market instances (mapped): {len(G)}")
