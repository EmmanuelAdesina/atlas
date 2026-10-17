"""Backtest LiveStateSpaceModel on the Fortuna Cologne vs VfB Stuttgart II fixture."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from atlas.live.observe import LiveMatchState, TeamStats
from atlas.live.estimate import LiveStateSpaceModel
from atlas.live.math_engine import devig_proportional, devig_shin, devig_power

fixture = json.loads(Path("tests/fixtures/fortuna_stuttgart_90plus.json").read_text())

s = fixture["state"]
state = LiveMatchState(
    match_id=fixture["match_id"],
    competition=s["competition"],
    home=TeamStats(
        name=s["home"]["name"],
        goals=s["home"]["goals"],
        possession_pct=s["home"]["possession_pct"],
        corners=s["home"]["corners"],
        yellow_cards=s["home"]["yellow_cards"],
        red_cards=s["home"]["red_cards"],
    ),
    away=TeamStats(
        name=s["away"]["name"],
        goals=s["away"]["goals"],
        possession_pct=s["away"]["possession_pct"],
        corners=s["away"]["corners"],
        yellow_cards=s["away"]["yellow_cards"],
        red_cards=s["away"]["red_cards"],
    ),
    period=s["period"],
    clock_minute=s["clock_minute"],
    clock_label=s["clock_label"],
    half_home_goals=s["half_home_goals"],
    half_away_goals=s["half_away_goals"],
    remaining_unknown=s["remaining_unknown"],
    observed_at=datetime.fromisoformat(fixture["meta"]["observed_at"]),
    source=fixture["meta"]["source"],
    source_url=fixture["meta"]["source_url"],
)

model = LiveStateSpaceModel(now=datetime.fromisoformat(fixture["meta"]["observed_at"]))

open_markets = [m for m in fixture["markets"] if m["status"] == "open"]

print("=" * 100)
print(f"BACKTEST: {fixture['meta']['fixture']} | Clock: {s['clock_label']} | Score: {s['home']['goals']}:{s['away']['goals']} (HT {s['half_home_goals']}:{s['half_away_goals']})")
print(f"Outcome: FT {fixture['outcome']['home_goals']}:{fixture['outcome']['away_goals']} | 2H settled: {fixture['outcome']['settled']}")
print("=" * 100)

results = []
for m in open_markets:
    mkey = m["market_key"]
    selections = {sel["name"]: sel["odds"] for sel in m["selections"]}

    # De-vig each method
    dv_prop = devig_proportional(selections)
    dv_shin, shin_z = devig_shin(selections)
    dv_power = devig_power(selections)

    print(f"\n{'-' * 100}")
    print(f"MARKET: {m['market_label']} ({mkey}) | line={m.get('line')} | period={m['period']}")
    print(f"{'-' * 100}")
    print(f"{'Selection':<15} {'Odds':>7} {'Impl(prop)':>11} {'Impl(Shin)':>11} {'Impl(Power)':>11} {'Model p':>9} {'EV(prop)':>9} {'EV(Shin)':>9} {'Bet?':>6} {'Won?':>6}")
    print(f"{'-' * 100}")

    for sel_name, odds in selections.items():
        est = model.estimate(state, mkey, sel_name)
        p_model = est.p
        ev_prop = est.expected_value(odds) if p_model is not None else None
        # EV using Shin de-vigged prob
        p_shin = dv_shin.get(sel_name)
        ev_shin = (p_shin * (odds - 1) - (1 - p_shin)) if p_shin is not None else None

        bet = "YES" if (ev_prop is not None and ev_prop > 0) else "no"

        # Determine if selection won
        won = ""
        if mkey == "ou_1_5":
            won = "WON" if (sel_name == "Under 1.5" and fixture["outcome"]["settled"]["ou_1_5"] == "Under 1.5") else "LOST"
        elif mkey == "1x2_2h":
            won = "WON" if sel_name == fixture["outcome"]["settled"]["1x2_2h"] else "LOST"

        print(f"{sel_name:<15} {odds:>7.2f} {dv_prop.get(sel_name, 0):>11.4f} {p_shin:>11.4f} {dv_power.get(sel_name, 0):>11.4f} {p_model:>9.4f} {ev_prop:>9.4f} {ev_shin:>9.4f} {bet:>6} {won:>6}")

        results.append({
            "market": mkey,
            "selection": sel_name,
            "odds": odds,
            "implied_prop": dv_prop.get(sel_name),
            "implied_shin": p_shin,
            "implied_power": dv_power.get(sel_name),
            "model_p": p_model,
            "ev_prop": ev_prop,
            "ev_shin": ev_shin,
            "bet": bet == "YES",
            "won": won,
            "confidence": est.confidence,
            "p_lower": est.p_lower,
            "p_upper": est.p_upper,
            "diagnostics": est.diagnostics,
        })

# Print diagnostics for first selection of each market
print(f"\n{'=' * 100}")
print("MODEL DIAGNOSTICS (nominal scenario)")
print(f"{'=' * 100}")
seen_markets = set()
for r in results:
    if r["market"] not in seen_markets:
        seen_markets.add(r["market"])
        d = r["diagnostics"]
        print(f"\n{r['market']}:")
        print(f"  mu_home_rem={d['mu_home_rem']}, mu_away_rem={d['mu_away_rem']}")
        print(f"  lambda_home_90={d['lambda_home_90']}, lambda_away_90={d['lambda_away_90']}")
        print(f"  rem_minutes_mean={d['rem_minutes_mean']}, bounds={d['rem_minutes_bounds']}")
        print(f"  red_card_mult={d['red_card_mult']}")
        print(f"  shrinkage_weight={d['shrinkage_weight']}")
        print(f"  uncertainty_width={d['uncertainty_width']}")
        print(f"  confidence={r['confidence']}, p=[{r['p_lower']:.4f}, {r['model_p']:.4f}, {r['p_upper']:.4f}]")

# Summary
print(f"\n{'=' * 100}")
print("SUMMARY")
print(f"{'=' * 100}")
bets = [r for r in results if r["bet"]]
print(f"Total selections evaluated: {len(results)}")
print(f"Bets placed (EV>0): {len(bets)}")
for b in bets:
    print(f"  - {b['market']}/{b['selection']} @ {b['odds']} (model_p={b['model_p']:.4f}, EV={b['ev_prop']:.4f}) -> {b['won']}")
wins = [b for b in bets if b["won"] == "WON"]
losses = [b for b in bets if b["won"] == "LOST"]
print(f"Bets won: {len(wins)}, lost: {len(losses)}")
if bets:
    total_ev = sum(b["ev_prop"] for b in bets)
    print(f"Sum of EV across bets: {total_ev:.4f}")
