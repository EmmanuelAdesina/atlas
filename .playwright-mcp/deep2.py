import numpy as np, math, json, sys, itertools
sys.path.insert(0, r'C:\Quant\atlas\.playwright-mcp')
from deep_core import *

d, legs, unmapped = load_legs()
P = np.array([l['p'] for l in legs])
ODDS = np.array([l['odds'] for l in legs])
# win mask = atoms paying exactly the decimal odds (full win); push pays 1.0, half-win pays (o+1)/2
WIN = np.array([(np.abs(l['p'] - l['odds']) < 1e-9) & (l['p'] > 0) for l in legs])
PUSH = np.array([(np.abs(l['p'] - 1.0) < 1e-9) & (np.abs(l['odds'] - 1.0) > 1e-9) for l in legs])
print("legs with full-win atoms:", int(WIN.sum(axis=1).astype(bool).sum()), "/", len(legs))

# ------------------------------------------------------------------ de-vig
def shin(odds):
    q = 1.0 / np.asarray(odds, float); B = q.sum()
    def f(z):
        return (np.sqrt(z*z + 4*(1-z)*q*q/B) - z) / (2*(1-z))
    if f(0.0).sum() < 1: return q / B, 0.0, B
    lo, hi = 0.0, 0.95
    for _ in range(200):
        mid = (lo+hi)/2
        if f(mid).sum() > 1: lo = mid
        else: hi = mid
    z = (lo+hi)/2
    return f(z), z, B

# group legs into market instances
INST = {}
for i, l in enumerate(legs):
    INST.setdefault((l['mid'], l['mdesc']), []).append(i)

# ------------------------------------------------------------------ model
def model_probs(lh, la):
    ph = np.array([math.exp(-lh)*lh**k/math.factorial(k) for k in range(HMAX+1)])
    pa = np.array([math.exp(-la)*la**k/math.factorial(k) for k in range(HMAX+1)])
    ph[-1] = sum(ph[HMAX:]); pa[-1] = sum(pa[HMAX:])
    M = np.outer(ph, pa); return M / M.sum()

# build fit targets: (leg_index, implied_prob) using per-market de-vig
FIT_MIDS = {'1','18','29','45','21','25','30','15','548','26','19','20','47' if False else '551','37','35','36'}
targets = []
for key, idxs in INST.items():
    mid, mdesc = key
    if mid not in FIT_MIDS: continue
    if mid == '37' or mid == '35' or mid == '36':
        pass  # handled per-instance below
    odds = ODDS[idxs]
    p, z, B = shin(odds)
    for k, i in enumerate(idxs):
        targets.append((i, p[k], key))
targets = [(i, p, k) for (i, p, k) in targets]
print(f"model fit targets: {len(targets)} legs")

def loss(lh, la):
    M = model_probs(lh, la).reshape(-1)
    err = [float(M[WIN[i]] @ np.ones(int(WIN[i].sum()))) * 0 + (M[WIN[i]].sum() - p) for (i, p, k) in targets]
    return float(np.dot(err, err))

best = (1e9, None)
for lh in np.arange(0.20, 3.001, 0.01):
    for la in np.arange(0.20, 3.001, 0.01):
        v = loss(lh, la)
        if v < best[0]: best = (v, (lh, la))
# local refine
lh, la = best[1]
step = 0.01
for _ in range(60):
    improved = False
    for dl, da in ((step,0),(-step,0),(0,step),(0,-step)):
        v = loss(lh+dl, la+da)
        if v < loss(lh, la):
            lh += dl; la += da; improved = True
    if not improved: step /= 2
    if step < 1e-7: break
LH, LA = lh, la
LOSS = loss(LH, LA)
print(f"fitted independent-Poisson: lambda_home={LH:.4f}  lambda_away={LA:.4f}  SSE={LOSS:.6f}  RMSE={math.sqrt(LOSS/len(targets)):.5f}")
Mp = model_probs(LH, LA).reshape(-1)
print(f"model implied: P(H)={Mp[RES_H].sum():.4f} P(D)={Mp[RES_D].sum():.4f} P(A)={Mp[RES_A].sum():.4f} "
      f"P(BTTS)={Mp[BTTS].sum():.4f} P(O2.5)={Mp[_tot>2.5].sum():.4f} P(Odd)={Mp[ODD].sum():.4f} "
      f"E[total]={float((_tot*Mp).sum()):.3f}")

# ------------------------------------------------------------------ dual LP (Frank-Wolfe)
def dual_lp(Pm, y0=None, iters=4000, verbose=False):
    n, m = Pm.shape
    y = np.ones(m)/m if y0 is None else y0.copy()
    y = y / y.sum()
    Pt = Pm.T
    for it in range(iters):
        vals = Pt @ y
        i = int(np.argmax(vals))
        f0 = vals[i]
        # atom where the binding leg pays least
        v = int(np.argmin(Pm[i]))
        d = np.zeros(m); d[v] = 1.0
        # exact line search on gamma in [0,1] for the PL objective
        lo, hi = 0.0, 1.0
        for _ in range(70):
            m1 = lo + (hi-lo)/3; m2 = hi - (hi-lo)/3
            y1 = (1-m1)*y + m1*d; y2 = (1-m2)*y + m2*d
            f1 = float(np.max(Pt @ y1)); f2 = float(np.max(Pt @ y2))
            if f1 <= f2: hi = m2
            else: lo = m1
        g = (lo+hi)/2
        y = (1-g)*y + g*d
        if it % 500 == 0 and verbose:
            print(f"   iter {it:5d}  f={float(np.max(Pt@y)):.6f}")
    vals = Pt @ y
    return y, float(np.max(vals)), int(np.argmax(vals))

print("\n--- solver validation on known partitions ---")
for name, key in [('1X2', ('1','')), ]:
    pass
def inst_idx(mid, mdesc_contains=None):
    for k, idxs in INST.items():
        if k[0] == mid and (mdesc_contains is None or mdesc_contains in k[1]):
            return idxs
    return None

ix_1x2 = inst_idx('1')
_, t12, _ = dual_lp(P[ix_1x2], iters=1500)
print(f"  1X2  dual optimum = {t12:.6f}   (analytic 1/B = {1/sum(1/ODDS[ix_1x2]):.6f})")
ix_oe = inst_idx('26')
_, toe, _ = dual_lp(P[ix_oe], iters=1500)
print(f"  Odd/Even dual optimum = {toe:.6f}  (analytic 1/B = {1/sum(1/ODDS[ix_oe]):.6f})")

print("\n--- GLOBAL certificate over all mapped legs ---")
y, tstar, binding = dual_lp(P, y0=Mp, iters=6000, verbose=True)
vals = P @ y
print(f"\nt* = min_y max_i E_y[payout_i] = {tstar:.6f}")
print(f"binding leg: {legs[binding]['mname']} / {legs[binding]['sel']} @ {legs[binding]['odds']}")
print(f"legs with E_y[payout] > 1.0 : {int((vals>1.0).sum())} / {len(legs)}")
print(f"legs with E_y[payout] > 0.99: {int((vals>0.99).sum())}")
order = np.argsort(-vals)
print("\n  top 12 legs by E_y[payout] (closest to 1 = 'sharpest' prices):")
for i in order[:12]:
    l = legs[i]
    print(f"   {l['mname'][:34]:34} | {l['sel'][:26]:26} | o={l['odds']:>7}  E_y[payout]={vals[i]:.4f}  EV={vals[i]-1:+.4f}")
print("\n  legs with E_y[payout] > 1 (model-value candidates):")
for i in order:
    if vals[i] <= 1.0: break
    l = legs[i]
    print(f"   {l['mname'][:34]:34} | {l['sel'][:26]:26} | o={l['odds']:>7}  E_y[payout]={vals[i]:.4f}  EV={vals[i]-1:+.4f}")

np.save(r'C:\Quant\atlas\.playwright-mcp\cert_y.npy', y)
json.dump({'LH': LH, 'LA': LA, 'tstar': tstar}, open(r'C:\Quant\atlas\.playwright-mcp\fit.json','w'))
print(f"\nsaved certificate y (support size {int((y>1e-9).sum())}), t*={tstar:.6f}")
