"""
Core engine for SportyBet board analysis.

Outcome space  : exact scoreline grid (h,a) with h,a in 0..8, where 8 == "8 or more".
Leg semantics  : each selection maps to a payout VECTOR over atoms:
                   o  = full win payout multiple
                   1  = void / push (stake returned)
                   0  = loss
                   fractional = half-win/half-push Asian settlement
Arbitrage test : a portfolio guarantees return R if for every atom w
                   sum_i s_i * P_i(w) >= R * sum_i s_i
Dual (Stiemke): R*_A = min_{y in simplex} max_{i in A} <P_i, y>
                   so if we find ANY y with max_i <P_i,y> < 1 for ALL legs,
                   then NO portfolio on this board can guarantee >= 1.0.
                   t* = min_y max_i <P_i,y> is an upper bound on the best
                   achievable guaranteed return.
"""
import json, math, re
import numpy as np

BOARD = r'C:\Quant\atlas\.playwright-mcp\pk35_jaro.json'

# ---------------------------------------------------------------- outcome space
HMAX = 8  # 8 means 8 or more
ATOMS = [(h, a) for h in range(HMAX + 1) for a in range(HMAX + 1)]
NA = len(ATOMS)
IDX = {w: i for i, w in enumerate(ATOMS)}

_h = np.array([w[0] for w in ATOMS], dtype=float)
_a = np.array([w[1] for w in ATOMS], dtype=float)
_tot = _h + _a
RES_H = _h > _a
RES_D = _h == _a
RES_A = _h < _a
BTTS = (_h > 0) & (_a > 0)
ODD = (_tot % 2) == 1
EVEN = (~ODD)

def _mask(h=None, a=None, tot=None):
    m = np.ones(NA, dtype=bool)
    if h is not None: m &= (_h == h)
    if a is not None: m &= (_a == a)
    if tot is not None: m &= (_tot == tot)
    return m

def _tot_in(lo, hi):
    """inclusive total-goal range; hi=None means open-ended (+)."""
    m = _tot >= lo
    if hi is None: return m
    return m & (_tot <= hi)

def _rng(desc, suffix='+'):
    """'1-5+' -> (1, None) meaning 1 or more? we treat 'a-b+' as a..b inclusive plus b+ -> >= b+1.
    Returns (lo, hi) using inclusive bounds, hi=None for unbounded."""
    d = desc.strip()
    if d.endswith('+'):
        core = d[:-1]
        if '-' in core:
            lo, hi = core.split('-', 1)
            return int(lo), None          # a-b+  ->  >= a   (ambiguous, handled by caller)
        return int(core), None
    if '-' in d:
        lo, hi = d.split('-', 1)
        return int(lo), int(hi)
    return int(d), int(d)

# ---------------------------------------------------------------- leg building
def _leg(win, push=None, o=None, half=0.0):
    """Build payout vector.
       win  : bool mask of full-win atoms
       push : bool mask of void atoms (optional)
       o    : decimal odds (payout multiple on full win)
       half : fraction of the stake that is 'half win half push' (Asian)
    """
    p = np.zeros(NA)
    if o is None:
        return p
    p[win] = o
    if push is not None and push.any():
        p[push] = 1.0
    if half:
        # half of stake wins at o, half is returned
        p[win] = o * 0.5 + 0.5
    return p

def _ou_leg(desc, line, is_over):
    """Over/Under with void on whole-ball lines."""
    if abs(line - int(line)) < 1e-9:
        push = _mask(tot=int(line))
        win = (_tot > line) if is_over else (_tot < line)
        if is_over:
            win = _tot > line
        else:
            win = _tot < line
        p = np.zeros(NA); p[win] = 0
        return ('push', win, push)
    else:
        win = (_tot > line) if is_over else (_tot < line)
        return ('plain', win, None)

# ---------------------------------------------------------------- market mapping
def map_market(m):
    """Return list of (desc, odds, payout_vector) or None if not mappable."""
    mid = m['id']
    name = m.get('name') or ''
    desc = m.get('desc') or ''
    outs = [(o['desc'], float(o['odds'])) for o in (m.get('outcomes') or [])
            if o.get('odds') and o.get('isActive', 1)]
    if not outs:
        return None

    def res_mask(r):
        return {'Home': RES_H, 'Draw': RES_D, 'Away': RES_A,
                'Home or Draw': RES_H | RES_D, 'Home or Away': RES_H | RES_A,
                'Draw or Away': RES_D | RES_A}[r]

    legs = []

    # ---- 1X2
    if mid == '1':
        for d, o in outs:
            legs.append((d, o, _leg(res_mask(d), o=o)))
        return legs

    # ---- Double Chance
    if mid == '10':
        for d, o in outs:
            legs.append((d, o, _leg(res_mask(d), o=o)))
        return legs

    # ---- Draw No Bet (void on draw)
    if mid == '11':
        for d, o in outs:
            win = RES_H if d == 'Home' else RES_A
            legs.append((d, o, _leg(win, push=RES_D, o=o)))
        return legs

    # ---- Home No Bet (void if home wins)  /  Away No Bet (void if away wins)
    if mid in ('12', '13'):
        void = RES_H if mid == '12' else RES_A
        for d, o in outs:
            win = res_mask(d)
            legs.append((d, o, _leg(win & ~void, push=void, o=o)))
        return legs

    # ---- Over/Under (total goals), whole lines push
    if mid == '18':
        for d, o in outs:
            m_ = re.match(r'^(Over|Under)\s+([0-9.]+)$', d)
            if not m_: return None
            side, line = m_.group(1), float(m_.group(2))
            kind, win, push = _ou_leg(d, line, side == 'Over')
            p = np.zeros(NA); p[win] = o
            if push is not None: p[push] = 1.0
            legs.append((d, o, p))
        return legs

    # ---- 3-way Handicap (x:y)  -> home +x, away +y
    if mid == '14':
        m_ = re.search(r'(\d+)\s*:\s*(\d+)', desc)
        if not m_: return None
        x, y = int(m_.group(1)), int(m_.group(2))
        diff = (_h + x) - (_a + y)
        for d, o in outs:
            side = d.split(' ')[0]
            win = (diff > 0) if side == 'Home' else ((diff == 0) if side == 'Draw' else (diff < 0))
            legs.append((d, o, _leg(win, o=o)))
        return legs

    # ---- Home O/U / Away O/U (team goal totals)
    if mid in ('19', '20'):
        base = _h if mid == '19' else _a
        for d, o in outs:
            m_ = re.match(r'^(Over|Under)\s+([0-9.]+)$', d)
            if not m_: return None
            side, line = m_.group(1), float(m_.group(2))
            whole = abs(line - int(line)) < 1e-9
            win = (base > line) if side == 'Over' else (base < line)
            push = (base == line) if whole else None
            p = np.zeros(NA); p[win] = o
            if push is not None: p[push] = 1.0
            legs.append((d, o, p))
        return legs

    # ---- 2-way Asian Handicap: Home(x) compares h+x vs a ; Away(y) compares a+y vs h
    if mid == '16':
        for d, o in outs:
            m_ = re.match(r'^(Home|Away)\s+\(([+-]?[0-9.]+)\)$', d)
            if not m_: return None
            side, line = m_.group(1), float(m_.group(2))
            if side == 'Home':
                diff = (_h + line) - _a
            else:
                diff = (_a + line) - _h
            whole = abs(line - int(line)) < 1e-9
            win = diff > 0
            push = (diff == 0) if whole else None
            legs.append((d, o, _leg(win, push=push, o=o)))
        return legs

    # ---- GG/NG  and  GG/NG 2+
    if mid == '29':
        for d, o in outs:
            legs.append((d, o, _leg(BTTS if d == 'Yes' else ~BTTS, o=o)))
        return legs
    if mid == '60000':
        w = (_h >= 2) & (_a >= 2)
        for d, o in outs:
            legs.append((d, o, _leg(w if d == 'Yes' else ~w, o=o)))
        return legs

    # ---- Odd/Even (total), home, away
    if mid in ('26', '27', '28'):
        if mid == '26':   base = _tot
        elif mid == '27': base = _h
        else:             base = _a
        par = (base % 2) == 1
        for d, o in outs:
            legs.append((d, o, _leg(par if d == 'Odd' else ~par, o=o)))
        return legs

    # ---- Correct Score (with Other)  and  Correct Score [x:y] (void outside list)
    if mid in ('45', '41'):
        listed = []
        for d, o in outs:
            m_ = re.match(r'^(\d+):(\d+)$', d)
            if m_:
                hh, aa = int(m_.group(1)), int(m_.group(2))
                listed.append((hh, aa))
        if mid == '45':
            covered = np.zeros(NA, dtype=bool)
            for hh, aa in listed:
                covered |= _mask(h=hh, a=aa)
            for d, o in outs:
                if d == 'Other':
                    legs.append((d, o, _leg(~covered, o=o)))
                else:
                    hh, aa = [int(x) for x in d.split(':')]
                    legs.append((d, o, _leg(_mask(h=hh, a=aa), o=o)))
            return legs
        else:
            for d, o in outs:
                hh, aa = [int(x) for x in d.split(':')]
                legs.append((d, o, _leg(_mask(h=hh, a=aa), push=~_mask(h=hh, a=aa), o=o)))
            return legs

    # ---- Exact Goals (total), Home/Away team exact goals
    if mid in ('21', '23', '24'):
        for d, o in outs:
            if d.endswith('+'):
                k = int(d[:-1])
                if mid == '21': msk = _tot >= k
                elif mid == '23': msk = _h >= k
                else: msk = _a >= k
            else:
                k = int(d)
                if mid == '21': msk = _mask(tot=k)
                elif mid == '23': msk = _mask(h=k)
                else: msk = _mask(a=k)
            legs.append((d, o, _leg(msk, o=o)))
        return legs

    # ---- Goal Range 0-1 / 2-3 / 4-6 / 7+
    if mid == '25':
        for d, o in outs:
            lo, hi = _rng(d)
            legs.append((d, o, _leg(_tot_in(lo, hi), o=o)))
        return legs

    # ---- Winning Margin
    if mid == '15':
        for d, o in outs:
            if d == 'Draw': msk = RES_D
            elif d.startswith('Home by 1'): msk = _h == _a + 1
            elif d.startswith('Home by 2'): msk = _h == _a + 2
            elif d.startswith('Home by 3'): msk = _h >= _a + 3
            elif d.startswith('Away by 1'): msk = _a == _h + 1
            elif d.startswith('Away by 2'): msk = _a == _h + 2
            elif d.startswith('Away by 3'): msk = _a >= _h + 3
            else: return None
            legs.append((d, o, _leg(msk, o=o)))
        return legs

    # ---- Which Team To Score
    if mid == '30':
        for d, o in outs:
            if d == 'None': msk = _mask(h=0, a=0)
            elif d == 'Only Home': msk = (_h > 0) & (_a == 0)
            elif d == 'Only Away': msk = (_h == 0) & (_a > 0)
            elif d == 'Both teams': msk = (_h > 0) & (_a > 0)
            else: return None
            legs.append((d, o, _leg(msk, o=o)))
        return legs

    # ---- Clean sheets / win to nil
    if mid == '31':
        for d, o in outs: legs.append((d, o, _leg(_a == 0 if d == 'Yes' else _a > 0, o=o)))
        return legs
    if mid == '32':
        for d, o in outs: legs.append((d, o, _leg(_h == 0 if d == 'Yes' else _h > 0, o=o)))
        return legs
    if mid == '33':
        for d, o in outs: legs.append((d, o, _leg(RES_H & (_a == 0) if d == 'Yes' else ~(RES_H & (_a == 0)), o=o)))
        return legs
    if mid == '34':
        for d, o in outs: legs.append((d, o, _leg(RES_A & (_h == 0) if d == 'Yes' else ~(RES_A & (_h == 0)), o=o)))
        return legs

    # ---- Multigoals (total / home / away)
    if mid in ('548', '549', '550'):
        base = {'548': _tot, '549': _h, '550': _a}[mid]
        for d, o in outs:
            if d == 'No goal': msk = base == 0
            else:
                lo, hi = _rng(d)
                msk = (base >= lo) & ((base <= hi) if hi is not None else (base >= lo))
                if d.endswith('+') and '-' not in d: msk = base >= lo
                elif d.endswith('+'):
                    # '1-5+' : 1..5 plus 6+  =>  >=1
                    msk = base >= lo
            legs.append((d, o, _leg(msk, o=o)))
        return legs

    # ---- Multiscores
    if mid == '551':
        listed_scores = []
        for d, o in outs:
            if d in ('Other Homewin', 'Other Awaywin', 'Draw'): continue
            for tok in re.findall(r'(\d+):(\d+)', d):
                listed_scores.append((int(tok[0]), int(tok[1])))
        cov = np.zeros(NA, dtype=bool)
        for hh, aa in listed_scores: cov |= _mask(h=hh, a=aa)
        for d, o in outs:
            if d == 'Other Homewin': msk = RES_H & ~cov
            elif d == 'Other Awaywin': msk = RES_A & ~cov
            elif d == 'Draw': msk = RES_D
            else:
                msk = np.zeros(NA, dtype=bool)
                for tok in re.findall(r'(\d+):(\d+)', d):
                    msk |= _mask(h=int(tok[0]), a=int(tok[1]))
            legs.append((d, o, _leg(msk, o=o)))
        return legs

    # ---- 1X2 & Over/Under   (1X2 & GG/NG)   (O/U & GG/NG)   (DC & GG/NG)  (DC & O/U)
    if mid in ('37', '35', '36', '546', '547'):
        for d, o in outs:
            low = d.lower()
            res = None
            if 'home/draw' in low: res = RES_H | RES_D
            elif 'home/away' in low: res = RES_H | RES_A
            elif 'draw/away' in low: res = RES_D | RES_A
            elif low.startswith('home'): res = RES_H
            elif low.startswith('draw'): res = RES_D
            elif low.startswith('away'): res = RES_A
            over = under = None
            m_ = re.search(r'over\s+([0-9.]+)', low)
            if m_: over = float(m_.group(1))
            m_ = re.search(r'under\s+([0-9.]+)', low)
            if m_: under = float(m_.group(1))
            if mid == '36':
                # 'Over 2.5 & Yes' : no result component, only total + BTTS
                if over is None and under is None: return None
                msk = np.ones(NA, dtype=bool)
                if over is not None: msk &= (_tot > over)
                if under is not None: msk &= (_tot < under)
                if ' & yes' in low: msk &= BTTS
                if ' & no' in low: msk &= ~BTTS
                legs.append((d, o, _leg(msk, o=o)))
                continue
            if res is None: return None
            msk = res.copy()
            if over is not None: msk = msk & (_tot > over)
            if under is not None: msk = msk & (_tot < under)
            if ' & yes' in low: msk = msk & BTTS
            if ' & no' in low: msk = msk & ~BTTS
            legs.append((d, o, _leg(msk, o=o)))
        return legs

    # ---- "X Or Y" combos (or-of-two markets)
    if mid in ('854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '900041'):
        for d, o in outs:
            low = desc.lower()
            res = None
            if 'home team or' in low or low.startswith('home team or'): res = RES_H
            elif 'away team or' in low or low.startswith('away or'): res = RES_A
            elif low.startswith('draw or'): res = RES_D
            elif 'home or any' in low: res = RES_H
            elif 'away or any' in low: res = RES_A
            elif 'no draw both' in low: res = None
            if 'no draw both' in low:
                msk = (~RES_D) & BTTS
            else:
                msk = res.copy()
                m_ = re.search(r'over\s+([0-9.]+)', low)
                if m_: msk = msk | (_tot > float(m_.group(1)))
                m_ = re.search(r'under\s+([0-9.]+)', low)
                if m_: msk = msk | (_tot < float(m_.group(1)))
                if 'gg/ng' in low: msk = msk | BTTS
                if 'clean sheet' in low: msk = msk | (_a == 0) | (_h == 0)
            if d == 'No': msk = ~msk
            legs.append((d, o, _leg(msk, o=o)))
        return legs

    return None


def load_legs(path=BOARD):
    d = json.load(open(path, encoding='utf-8'))['data']
    legs, unmapped = [], []
    for m in d['markets']:
        r = map_market(m)
        if r is None:
            unmapped.append((m['id'], m.get('name'), m.get('desc')))
        else:
            for dsc, o, p in r:
                legs.append({
                    'mid': m['id'], 'mname': m.get('name') or m.get('desc'),
                    'mdesc': m.get('desc') or '', 'sel': dsc, 'odds': o, 'p': p,
                })
    return d, legs, unmapped
