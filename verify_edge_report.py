#!/usr/bin/env python3
"""ATLAS deterministic verifier for external edge-AI superedge reports.

Doctrine (docs/engineering.md): the LLM proposes, the deterministic kernel
verifies. Nothing an external edge-AI reports is trusted until every
reconstructible number has been recomputed here with ATLAS's own math.

Usage:
    python3 verify_edge_report.py reports/edge_ai/2026-10-21_pk35-jaro/report.json

The report's directory must contain `reconstructed_odds.json`: the odds and
prose-claimed figures recoverable from the report's own text (the edge-AI
board dump itself, if it attached one, would extend coverage further).

Checks performed:
  1. cross-family pairs -- implied sums, guaranteed profit, arb decision
  2. family booksums   -- sum(1/odds) vs claimed
  3. Shin de-vig       -- ATLAS devig_shin z and fair probabilities vs claimed
  4. EV audit          -- every reconstructible leg priced against the same
                          board's fair probabilities (DC, handicap, BTTS model)
  5. ATLAS arb scan    -- find_surething_arbs / find_cross_family_arbs on the
                          reconstructible board; must agree with the report
  6. phantom guards    -- path markets (never-down) and void markets (no-bet)
                          must be structurally excluded, never arbitraged
  7. prose claims      -- arithmetic embedded in the narrative

Exit codes: 0 = CONFIRMED, 1 = REFUTED, 2 = INCONCLUSIVE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from atlas.live.arb import (  # noqa: E402
    DEFAULT_MIN_EDGE,
    EXHAUSTIVE_FAMILIES,
    Selection,
    find_cross_family_arbs,
    find_surething_arbs,
)
from atlas.live.math_engine import (  # noqa: E402
    build_remaining_score_matrix,
    devig_shin,
    infer_prematch_lambdas_from_1x2,
)

MIN_EDGE = DEFAULT_MIN_EDGE      # 0.005, hard-coded in the edge-AI prompt
SUM_TOL = 5e-4                   # claimed vs recomputed inverse-price sums
PROFIT_TOL = 1.5e-3              # claimed vs recomputed profit pct points
SHIN_Z_TOL = 5e-3                # shin z agreement (method variants differ)
SHIN_P_TOL = 3e-3                # fair probability agreement
COMPLEMENT_TOL = 0.02            # p_sum tolerance for a true cover


# --------------------------------------------------------------------------
# audit plumbing
# --------------------------------------------------------------------------

class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.hard_failures = 0
        self.nits = 0
        self.missed_arbs: list[str] = []

    def add(self, status: str, check: str, detail: str) -> None:
        self.rows.append((status, check, detail))
        if status == "FAIL":
            self.hard_failures += 1
        elif status == "NIT":
            self.nits += 1

    def print(self) -> None:
        width = max((len(c) for _, c, _ in self.rows), default=10)
        for status, check, detail in self.rows:
            print(f"[{status:<4}] {check:<{width}}  {detail}")
        print()


def parse_leg(label: str) -> tuple[str, float]:
    """'1x2 Home @3.25' -> ('1x2 Home', 3.25)."""
    text, _, odds = label.rpartition("@")
    if not text:
        raise ValueError(f"no odds in leg label: {label!r}")
    return text.strip(), float(odds)


def implied_sum(odds: dict[str, float]) -> float:
    return sum(1.0 / float(o) for o in odds.values())


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_pairs(report: dict, audit: Audit) -> None:
    pairs = report.get("cross_family_pairs_tested", [])
    for i, pair in enumerate(pairs, 1):
        a_label, a_odds = parse_leg(pair["leg_a"])
        b_label, b_odds = parse_leg(pair["leg_b"])
        s = 1.0 / a_odds + 1.0 / b_odds
        claimed_s = float(pair["implied_sum"])
        ok_sum = abs(s - claimed_s) <= SUM_TOL

        profit = (1.0 / s - 1.0) * 100.0
        claimed_profit = float(pair.get("profit_pct", profit))
        ok_profit = abs(profit - claimed_profit) <= PROFIT_TOL

        complementary = abs(float(pair["p_sum"]) - 1.0) <= COMPLEMENT_TOL
        arb = complementary and s < 1.0 - MIN_EDGE
        ok_verdict = arb == bool(pair.get("superedge"))

        name = f"pair {i}: {a_label} + {b_label}"
        if ok_sum and ok_profit and ok_verdict:
            audit.add(
                "PASS",
                name,
                f"implied {s:.6f} == claimed {claimed_s:.6f}; "
                f"guaranteed profit {profit:+.3f}% == claimed {claimed_profit:+.3f}%; "
                f"not an arb (correct)",
            )
        else:
            detail = (
                f"implied {s:.6f} vs claimed {claimed_s:.6f} "
                f"(sum ok={ok_sum}); profit {profit:+.3f}% vs claimed "
                f"{claimed_profit:+.3f}% (ok={ok_profit}); "
                f"recomputed arb={arb} vs claimed {pair.get('superedge')}"
            )
            audit.add("FAIL", name, detail)
            if arb:
                audit.missed_arbs.append(name)


def check_families(side: dict, audit: Audit) -> dict[str, float]:
    computed: dict[str, float] = {}
    for key, claimed in side.get("claimed_booksums", {}).items():
        odds = side.get("families", {}).get(key)
        if not odds:
            continue
        s = implied_sum(odds)
        computed[key] = s
        if abs(s - claimed) <= SUM_TOL:
            audit.add("PASS", f"booksum {key}", f"{s:.6f} == claimed {claimed:.6f}")
        else:
            audit.add(
                "FAIL", f"booksum {key}",
                f"{s:.6f} != claimed {claimed:.6f}",
            )
    not_rebuildable = side.get("claimed_booksums_not_reconstructible", {})
    if not_rebuildable:
        keys = ", ".join(f"{k}={v}" for k, v in not_rebuildable.items())
        audit.add("INFO", "booksums not reconstructible",
                  f"claimed but odds not quoted in report: {keys}")
    return computed


def check_shin(report: dict, side: dict, audit: Audit) -> dict[str, dict[str, float]]:
    fair_by_family: dict[str, dict[str, float]] = {}
    z_claims = report.get("shin_z_by_market", {})
    for fam_key, z_key in side.get("shin_z_key_map", {}).items():
        odds = side["families"][fam_key]
        fair, z = devig_shin(odds)
        fair_by_family[fam_key] = fair
        claimed_z = float(z_claims.get(z_key, float("nan")))
        ok_z = abs(z - claimed_z) <= SHIN_Z_TOL
        audit.add(
            "PASS" if ok_z else "FAIL",
            f"shin z {fam_key}",
            f"ATLAS z={z:.6f} vs claimed {claimed_z:.6f}; "
            f"fair={{{', '.join(f'{k} {v:.4f}' for k, v in fair.items())}}}",
        )
    claimed_fair = side.get("claimed_shin_fair_1x2")
    if claimed_fair and "1x2" in fair_by_family:
        for sel, claimed_p in claimed_fair.items():
            p = fair_by_family["1x2"][sel]
            ok = abs(p - claimed_p) <= SHIN_P_TOL
            audit.add(
                "PASS" if ok else "FAIL",
                f"shin fair 1x2 {sel}",
                f"ATLAS {p:.4f} vs claimed {claimed_p:.4f}",
            )
    return fair_by_family


def btts_no_model_price(odds_1x2: dict[str, float]) -> float:
    """Model price for BTTS No from ATLAS's own 1X2 -> Poisson inversion."""
    lh, la = infer_prematch_lambdas_from_1x2(
        odds_1x2["Home"], odds_1x2["Draw"], odds_1x2["Away"]
    )
    mat = build_remaining_score_matrix(lh, la, rho=-0.06, max_goals=10)
    p_yes = sum(mat[r][c] for r in range(1, 11) for c in range(1, 11))
    return 1.0 - p_yes


def resolve_p(spec: str, fair_by_family: dict[str, dict[str, float]],
              btts_no_p: float) -> tuple[float, bool]:
    """'family:Sel1+Sel2' -> (probability, decision_relevant)."""
    if spec == "model:btts_no":
        return btts_no_p, False
    fam, expr = spec.split(":", 1)
    fair = fair_by_family[fam]
    p = sum(fair[part] for part in expr.split("+"))
    return p, True


def check_ev(side: dict, fair_by_family: dict[str, dict[str, float]],
             audit: Audit) -> None:
    btts_no_p = btts_no_model_price(side["families"]["1x2"])
    for leg in side.get("ev_legs", []):
        p, decision_relevant = resolve_p(leg["p_from"], fair_by_family, btts_no_p)
        ev = p * float(leg["odds"]) - 1.0
        tag = "EV" if decision_relevant else "EV(model-priced)"
        if ev < 0:
            audit.add("PASS", f"{tag} {leg['label']}",
                      f"p={p:.4f} -> EV {ev:+.2%} (negative, consistent)")
        elif decision_relevant:
            audit.add("FAIL", f"{tag} {leg['label']}",
                      f"p={p:.4f} -> EV {ev:+.2%} POSITIVE on same-board fair "
                      f"probs: contradicts a no_edge verdict")
        else:
            audit.add("NIT", f"{tag} {leg['label']}",
                      f"p={p:.4f} -> EV {ev:+.2%} positive only under the "
                      f"model price (advisory, not same-board evidence)")


def build_selections(side: dict, fair_by_family: dict[str, dict[str, float]]) -> list[Selection]:
    sels: list[Selection] = []
    for fam in ("1x2", "1x2_1h", "1x2_2h"):
        for sel, p in fair_by_family[fam].items():
            sels.append(Selection(market_key=fam, selection=sel,
                                  odds=side["families"][fam][sel], model_p=p))
    dc_sources = {
        "double_chance": "1x2",
        "1h_double_chance": "1x2_1h",
        "2h_double_chance": "1x2_2h",
    }
    for dc_fam, src in dc_sources.items():
        fair = fair_by_family[src]
        for sel, odds in side["families"][dc_fam].items():
            p = sum(fair[t] for t in sel.split(" or "))
            sels.append(Selection(market_key=dc_fam, selection=sel,
                                  odds=odds, model_p=p))
    # Phantom family: fed to the scanner on purpose. It must be excluded.
    for sel, odds in side["families"]["1x2_never_down"].items():
        sels.append(Selection(market_key="1x2_never_down", selection=sel,
                              odds=odds, model_p=1.0 / 3.0))
    return sels


def check_atlas_arb_scan(side: dict, fair_by_family: dict[str, dict[str, float]],
                         audit: Audit) -> None:
    sels = build_selections(side, fair_by_family)
    sure = find_surething_arbs(sels, min_edge=MIN_EDGE)
    cross = find_cross_family_arbs(sels, min_edge=MIN_EDGE)
    all_arbs = sure + cross

    report_claims_edge = "superedge" in str(report_verdict)
    if all_arbs:
        for a in all_arbs:
            audit.missed_arbs.append(a.describe())
        audit.add("FAIL", "ATLAS arb scan",
                  f"found {len(all_arbs)} arb(s) the report denied: "
                  + "; ".join(a.describe() for a in all_arbs))
    elif report_claims_edge:
        audit.add("FAIL", "ATLAS arb scan",
                  "report claims a superedge but ATLAS finds none on the "
                  "reconstructible board")
    else:
        audit.add("PASS", "ATLAS arb scan",
                  f"surething=0, cross_family=0 on {len(sels)} selections "
                  f"({len(fair_by_family)} 1X2 families + 3 DC families + "
                  f"phantom never-down) — agrees with report")

    # Phantom guards, demonstrated live:
    nd = implied_sum(side["families"]["1x2_never_down"])
    guarded = "1x2_never_down" not in EXHAUSTIVE_FAMILIES
    audit.add(
        "PASS" if guarded else "FAIL",
        "phantom guard: never-down",
        f"booksum {nd:.4f}; family in EXHAUSTIVE_FAMILIES={not guarded}; "
        f"{'excluded by scanner' if guarded else 'NOT EXCLUDED — BUG'}; "
        f"also above the 1-{MIN_EDGE} arb threshold "
        f"({nd:.4f} >= {1.0 - MIN_EDGE:.3f})",
    )
    dc_sum = sum(
        sum(s.model_p for s in sels if s.market_key == fam)
        for fam in ("double_chance",)
    )
    if abs(dc_sum - 2.0) < 1e-9:
        audit.add(
            "INFO", "latent footgun: double_chance",
            "EXHAUSTIVE_FAMILIES lists double_chance, but DC selections sum "
            "to model_p=2.0 (two legs cash on every result). The model-sum "
            "guard rejects it, so behaviour is correct — but the constant "
            "is misleading and should be split into single-family partitions "
            "vs cross-family complement sources.",
        )


def check_prose(side: dict, audit: Audit) -> None:
    for claim in side.get("prose_claims", []):
        src = claim["computed_from"]
        if "formula" in claim and "booksum" in src:
            computed = 1.0 / float(src["booksum"])
        else:
            computed = sum(1.0 / float(o) for o in src.values())
        claimed = float(claim["claimed"])
        ok = abs(computed - claimed) <= SUM_TOL
        if ok:
            audit.add("PASS", f"prose: {claim['id']}",
                      f"{computed:.6f} == claimed {claimed:.6f}")
        else:
            status = "NIT" if not claim.get("decision_relevant", False) else "FAIL"
            note = claim.get("expected_note", "")
            audit.add(status, f"prose: {claim['id']}",
                      f"claimed {claimed} but recomputes to {computed:.6f} "
                      f"from the quoted odds. {note}".strip())


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

report_verdict = ""


def main(argv: list[str]) -> int:
    global report_verdict
    if len(argv) != 2:
        print(__doc__)
        return 2
    report_path = Path(argv[1])
    report = json.loads(report_path.read_text())
    report_verdict = str(report.get("verdict", ""))
    sidecar_path = report_path.with_name("reconstructed_odds.json")

    print("=" * 78)
    print("ATLAS DETERMINISTIC VERIFICATION OF EDGE-AI REPORT")
    print("=" * 78)
    print(f"report      : {report_path}")
    print(f"match       : {report.get('home_team')} vs {report.get('away_team')}"
          f"  ({report.get('competition')})")
    print(f"kickoff     : {report.get('kickoff_iso')}   observed: {report.get('observed_at')}")
    print(f"claimed     : verdict={report_verdict} confidence={report.get('confidence')}")
    print()

    if not sidecar_path.exists():
        print("INCONCLUSIVE: no reconstructed_odds.json sidecar next to the report.")
        print("The edge-AI report must ship its raw board (or a sidecar of its")
        print("quoted odds) before ATLAS can verify anything deterministically.")
        return 2
    side = json.loads(sidecar_path.read_text())

    audit = Audit()
    check_pairs(report, audit)
    computed_booksums = check_families(side, audit)

    # top-level implied_total should equal the tightest claimed booksum
    top = report.get("implied_total")
    odd_even = side.get("claimed_booksums", {}).get("odd_even")
    if top is not None and odd_even is not None:
        ok = abs(float(top) - float(odd_even)) <= SUM_TOL
        audit.add("PASS" if ok else "FAIL", "top-level implied_total",
                  f"{top} vs tightest claimed booksum (odd_even) {odd_even}")

    fair_by_family = check_shin(report, side, audit)
    check_ev(side, fair_by_family, audit)
    check_atlas_arb_scan(side, fair_by_family, audit)
    check_prose(side, audit)

    audit.print()

    # ---------------------------------------------------------------- verdict
    n_families = len(computed_booksums)
    n_pairs = len(report.get("cross_family_pairs_tested", []))
    print("-" * 78)
    print(f"COVERAGE: {n_pairs} cross-family pairs, {n_families} family booksums, "
          f"{len(fair_by_family)} Shin de-vigs, {len(side.get('ev_legs', []))} EV legs, "
          f"prose claims — all rebuilt from the report's own quoted odds.")
    print("The raw 215-market board was NOT attached, so scraping fidelity "
          "cannot be audited from this report alone.")
    print()

    if audit.missed_arbs:
        print("VERDICT: REFUTED — a true cover prices below the arb threshold:")
        for m in audit.missed_arbs:
            print(f"  - {m}")
        return 1
    if audit.hard_failures:
        print(f"VERDICT: REFUTED — {audit.hard_failures} decision-relevant "
              f"check(s) failed above.")
        return 1
    if audit.nits:
        print(f"VERDICT: CONFIRMED ({audit.nits} non-decision-relevant nit(s) "
              f"flagged) — no edge on every reconstructible claim.")
        return 0
    print("VERDICT: CONFIRMED — every reconstructible claim checks out; no edge.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
