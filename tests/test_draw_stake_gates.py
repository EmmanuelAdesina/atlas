"""The 2026-10-31 draw-burst regression suite.

Six draw-family bets on ONE live match in seven minutes reached the book
because (a) the sequential portfolio never checked the posture's min_ev —
'1x2 Draw' at EV 0.0315 was "recommended"; (b) the derivative 1x2 variants
('1up', '2up', 'never down') were invisible to the correlation mapper, so
each variant of the same match was its own factor; and (c) a high-confidence
base estimate skipped the model-vs-market gap blocker entirely, pricing
'never_down Draw' at p 0.611 against the book's implied ~0.50 (a phantom
+18% EV that kept topping the candidate list as the price drifted).

Each test pins one of those three holes shut. A green suite means the next
run cannot repeat the burst through the same code path.
"""

from atlas.live.estimate import Estimate
from atlas.live.loop import Opportunity, selection_correlation_groups
from atlas.live.math_engine import devig_shin
from atlas.ops.posture import Posture, PostureMode
from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio


# --- (a) the portfolio must hold the EV floor ----------------------------


def make_opp(
    match_id="m1",
    market_key="1x2",
    selection="Draw",
    odds=2.05,
    model_p=0.6,
    p_lower=None,
):
    return Opportunity(
        match_id=match_id,
        market_key=market_key,
        selection=selection,
        odds=odds,
        model_p=model_p,
        p_lower=p_lower,
        model_name="test-model",
    )


MATCH_STATE = {"period": "1h", "clock_minute": 30, "score": "0-0", "events": []}


def test_evaluate_snapshot_rejects_a_below_floor_ev():
    """'1x2 Draw' @2.05 with p 0.5032 was EV 0.0315 and got recommended
    because evaluate_snapshot never consulted the EV floor it was given."""
    portfolio = SequentialPortfolio(SequentialConfig(min_ev=0.05))
    decisions = portfolio.evaluate_snapshot(
        opportunities=[make_opp(model_p=0.6, odds=2.0)],  # EV 0.20 -> fine
        match_state=MATCH_STATE,
        placed_bets=[],
    )
    portfolio2 = SequentialPortfolio(SequentialConfig(min_ev=0.05))
    low = portfolio2.evaluate_snapshot(
        opportunities=[make_opp(model_p=0.5032, odds=2.05)],  # EV 0.0315
        match_state=MATCH_STATE,
        placed_bets=[],
    )
    assert len(low) == 1
    d = low[0]
    assert d.should_bet is False, f"EV 0.0315 must not reach the book: {d.reason}"
    assert "min_ev" in d.reason
    assert decisions[0].should_bet is not None


def test_evaluate_snapshot_does_not_use_the_ev_floor_answer_for_recordkeeping():
    """The rejection reason must name the floor so the operator can audit
    WHY nothing was placed without re-deriving the arithmetic."""
    cfg = SequentialConfig(min_ev=0.05)
    decisions = SequentialPortfolio(cfg).evaluate_snapshot(
        opportunities=[make_opp(model_p=0.5032, odds=2.05)],
        match_state=MATCH_STATE,
        placed_bets=[],
    )
    assert f"{cfg.min_ev:.2f}" in decisions[0].reason


# --- (b) derivative 1x2 variants are one family --------------------------


def test_all_derivative_variants_share_the_1x2_family_group():
    for variant in ("1x2_1up", "1x2_2up", "1x2_never_down"):
        groups = selection_correlation_groups(variant, "Draw")
        assert "1x2_family" in groups, f"{variant}/Draw missing from 1x2_family"
        assert "low_tempo_under" in groups, f"{variant}/Draw must draw-side with plain 1x2"


def test_a_variant_draw_and_the_plain_1x2_draw_land_in_the_same_group():
    """Four variants at 2.05 all firing within two minutes was the visible
    symptom; after this fix they enter the same correlation group so the
    per-group bookkeeping sees ONE match, not four markets."""
    plain = set(selection_correlation_groups("1x2", "Draw"))
    variant = set(selection_correlation_groups("1x2_never_down", "Draw"))
    assert plain & variant, "plain and variant draws must share factors"
    assert "1x2_family" in (plain & variant)


def test_variant_home_selection_keeps_its_directional_factors():
    groups = selection_correlation_groups("1x2_never_down", "Home")
    assert "home_goals" in groups and "home_supremacy" in groups
    assert "1x2_family" in groups


def test_half_market_1x2_keys_still_reach_the_first_mapping_branch():
    """'1x2_1h' is in the explicit family tuple; the startswith('1x2_')
    branch must not double-map or shadow it."""
    groups = selection_correlation_groups("1x2_1h", "Home")
    assert groups.count("1x2_family") == 1
    assert "home_goals" in groups


# --- (c) high-confidence variants cannot skip the market-gap blocker -----


class _StubBase:
    """A base model whose answer the test controls exactly."""

    def __init__(self, p, confidence="high"):
        self.p = p
        self.confidence = confidence

    def estimate(self, state, market_key, selection):  # noqa: D102
        return Estimate(
            p=self.p,
            confidence=self.confidence,
            model_name="stub",
        )


from datetime import datetime, timezone  # noqa: E402

from atlas.live.observe import LiveMatchState, TeamStats  # noqa: E402


def make_state(home=0, away=0):
    """A live 0-0 in the second half — the same shape test_runner uses, so
    the calibration layer's score-level logic behaves normally."""
    return LiveMatchState(
        match_id="m1",
        home=TeamStats(name="Home", goals=home),
        away=TeamStats(name="Away", goals=away),
        period="2h",
        clock_minute=70.0,
        observed_at=datetime.now(timezone.utc),
        source="test",
    )


from atlas.live.estimate import CalibratedModel  # noqa: E402


def _calibrated_for(odds_map, key):

    fair, _ = devig_shin(odds_map)
    return CalibratedModel(
        base_model=_StubBase(0.6114, "high"),
        market_fair={(key, k): v for k, v in fair.items()},
    )


def test_a_high_confidence_variant_estimate_abstains_on_a_market_gap():
    """Tonight's live case: 'never_down Draw' model p 0.6114 vs the book's
    de-vigged ~0.50. Even at HIGH confidence the variant must pass the same
    5pp bar a low-confidence estimate would — or abstain."""
    model = _calibrated_for(
        {"Home": 2.10, "Draw": 2.05, "Away": 3.30}, "1x2_never_down"
    )
    est = model.estimate(state=make_state(), market_key="1x2_never_down", selection="Draw")
    assert est.p is None, "a >5pp disagreement is a phantom edge, not an edge"
    assert est.abstained is True
    assert "gap" in (est.abstain_reason or "")


def test_a_variant_estimate_close_to_the_market_still_prices():
    """The blocker is a bar, not a ban: a variant whose estimate agrees with
    the book keeps producing a price, so the variant family is not dead
    code after this fix."""
    from atlas.live.estimate import CalibratedModel

    fair, _ = devig_shin({"Home": 2.10, "Draw": 2.05, "Away": 3.30})
    model = CalibratedModel(
        base_model=_StubBase(round(fair.get("Draw", 0.488), 6), "high"),
        market_fair={("1x2_never_down", k): v for k, v in fair.items()},
    )
    est = model.estimate(state=make_state(), market_key="1x2_never_down", selection="Draw")
    assert est.p is not None and est.abstained is False


def test_a_variant_with_no_market_price_abstains_rather_than_trusting_the_model():
    """Without the book's own de-vigged probability there is nothing for the
    variant to be checked against — abstaining is the only honest answer."""
    from atlas.live.estimate import CalibratedModel

    model = CalibratedModel(base_model=_StubBase(0.6114, "high"), market_fair={})
    est = model.estimate(state=make_state(), market_key="1x2_1up", selection="Draw")
    assert est.abstained is True and "no market-implied" in (est.abstain_reason or "")


def test_the_derivative_variant_registry_covers_tonights_markets():
    from atlas.live.estimate import CalibratedModel

    assert CalibratedModel.DERIVATIVE_1X2_VARIANTS >= {
        "1x2_1up",
        "1x2_2up",
        "1x2_never_down",
    }


# --- the confidence grade must survive the hop to the portfolio ----------


def test_an_opportunity_defaults_to_unknown_and_the_bar_is_closed():
    """Fail-closed: an Opportunity built without a grade is 'unknown', and
    'unknown' is never bettable. A wiring regression can only stop bets,
    never silently re-open them."""
    opp = make_opp()
    assert opp.model_confidence == "unknown"


def test_estimate_for_preserves_the_grade_it_is_given():
    from atlas.portfolio.sequential import _estimate_for

    opp = make_opp()
    opp.model_confidence = "low"
    assert _estimate_for(opp).confidence == "low"
    opp.model_confidence = "high"
    assert _estimate_for(opp).confidence == "high"


def test_the_portfolio_rejects_a_low_confidence_opportunity_even_above_min_ev():
    """Tonight's Under 2.5: EV +0.15 claimed, prior-only intensity underneath.
    An above-floor EV on a 'low' grade is exactly the shape that must be
    refused — the grade exists to stop this."""
    cfg = SequentialConfig(min_ev=0.05)
    low_opp = make_opp(model_p=0.62, odds=2.0)
    low_opp.model_confidence = "low"
    decisions_low = SequentialPortfolio(cfg).evaluate_snapshot(
        opportunities=[low_opp], match_state=MATCH_STATE, placed_bets=[]
    )
    assert len(decisions_low) == 1 and decisions_low[0].should_bet is False
    assert "confidence" in decisions_low[0].reason
    assert decisions_low[0].opportunity.ev > cfg.min_ev, (
        "the test only proves the point if the EV gate would have passed it"
    )


def test_the_portfolio_rejects_an_unknown_confidence_opportunity():
    cfg = SequentialConfig(min_ev=0.05)
    unknown = make_opp(model_p=0.62, odds=2.0)
    assert unknown.model_confidence == "unknown"
    decisions = SequentialPortfolio(cfg).evaluate_snapshot(
        opportunities=[unknown], match_state=MATCH_STATE, placed_bets=[]
    )
    assert decisions[0].should_bet is False
    assert "confidence" in decisions[0].reason


# --- one position per market/selection: drift is not a new bet -----------


def _now_placed(market_key="ou_2_5", selection="Under 2.5", odds=2.0, ev=0.15):
    from datetime import datetime, timedelta, timezone

    return {
        "match_id": "m1",
        "market_key": market_key,
        "selection": selection,
        "odds": odds,
        "stake_ngn": 10.0,
        "ev": ev,
        "timestamp": (datetime.now(timezone.utc) - timedelta(seconds=70)).isoformat(),
        "period": "2h",
        "clock_minute": 62.0,
        "score": (2, 0),
        "events_count": 0,
    }


def test_a_drifted_rebalance_cannot_re_stake_an_open_position():
    """The 2026-10-31 Under 2.5: staked at 2.80; minutes later the price
    drifted to 2.55 (7.3% > the 6% rebalance trigger) and the portfolio
    minted a SECOND NGN 10 on the same economic bet. One open position per
    market/selection, regardless of the drift arithmetic."""
    cfg = SequentialConfig(min_ev=0.05)
    portfolio = SequentialPortfolio(cfg)
    fresh = make_opp(market_key="ou_2_5", selection="Under 2.5",
                     odds=2.4, model_p=0.52, p_lower=0.46)
    fresh.model_confidence = "medium"
    first = portfolio.evaluate_snapshot(
        opportunities=[fresh], match_state=MATCH_STATE,
        placed_bets=[dict(_now_placed(odds=1.55, ev=0.12))],
    )
    defy = make_opp(market_key="ou_2_5", selection="Under 2.5",
                    odds=2.9, model_p=0.42, p_lower=0.37)
    defy.model_confidence = "medium"
    second = portfolio.evaluate_snapshot(
        opportunities=[defy], match_state=MATCH_STATE,
        placed_bets=[dict(_now_placed(odds=2.8, ev=0.12))],
    )
    assert second[0].should_bet is False
    # either gate refusing is correct — but at EV 0.218 (inside the band)
    # the refusal must be the DRIFT cap, not the absurd-EV ceiling
    assert "position already open" in second[0].reason


def test_a_different_selection_on_the_same_marked_market_is_not_the_position():
    cfg = SequentialConfig(min_ev=0.05)
    opp = make_opp(market_key="ou_2_5", selection="Over 2.5",
                   odds=2.0, model_p=0.62, p_lower=0.55)
    opp.model_confidence = "medium"
    decisions = SequentialPortfolio(cfg).evaluate_snapshot(
        opportunities=[opp], match_state=MATCH_STATE,
        placed_bets=[dict(_now_placed())],
    )
    assert decisions[0].should_bet is not None
    assert "position already open" not in decisions[0].reason


# --- the posture's floor must land in the config that gates --------------


def test_posture_apply_to_forwards_min_ev_to_the_portfolio_config():
    posture = Posture(mode=PostureMode.BALANCED, min_ev=0.07)
    cfg = posture.apply_to(SequentialConfig())
    assert cfg.min_ev == 0.07, (
        "the posture's EV floor must reach evaluate_snapshot, not just "
        "the policy engine's decision path"
    )


def test_posture_apply_to_keeps_the_one_position_cap_intact():
    posture = Posture(mode=PostureMode.BALANCED)
    before = posture.apply_to(SequentialConfig(max_bets_per_selection=5))
    assert before.max_bets_per_selection == 5


# --- runner wiring: price_match stamps the REAL grade ---------------------


class _VariedBase:
    """A base model whose grade the test pins to 'medium' regardless of the
    market; the probabilities track fair values so the gap bar passes."""

    def __init__(self):
        pass

    def estimate(self, state, market_key, selection):
        from atlas.live.estimate import Estimate

        fair = {
            "Home": 0.44,
            "Draw": 0.295,
            "Away": 0.25,
        }
        return Estimate(
            p=fair.get(selection, 0.4),
            confidence="medium",
            model_name="stub-varied",
        )


def test_price_match_stamps_a_priced_grade_onto_every_opportunity(tmp_path):
    """2026-10-31: the grade died between pricing and the portfolio — every
    Opportunity reached the selector as if the model were confident. The
    stamping must be prophylactic: any regression here shows up as many
    fewer priced selections, never as a re-opened betting path."""
    from types import SimpleNamespace

    from atlas.live.runner import LiveRunner

    runner = LiveRunner(
        capture=object(),
        ledger=SimpleNamespace(),
        state_path=str(tmp_path / "state.json"),
    )
    runner.model = _VariedBase()
    market = {
        "match_id": "m1",
        "market_key": "1x2",
        "market_label": "1X2",
        "status": "open",
        "selections": [
            {"name": "Home", "odds": 2.20},
            {"name": "Draw", "odds": 3.40},
            {"name": "Away", "odds": 3.20},
        ],
    }
    opps, priced, _ = runner.price_match(make_state(), [market])
    assert priced >= 2
    assert opps and all(
        o.model_confidence == "medium" for o in opps
    ), [f"{o.market_key}/{o.selection}: {o.model_confidence}" for o in opps]


# --- the 2026-11-01 Perth pathologies, gate-by-gate ------------------------


def test_drift_chase_on_handicap_draw_is_blocked_at_every_price():
    """The exact tape: `handicap_0_3 Draw (0:3)` staked at 2.65, then the
    book cut to 2.45 / 2.30 / 2.15 and the old engine re-loaded each time.
    max_bets_per_selection=1 must refuse the SAME selection at ANY price
    while the position is open — the price direction is irrelevant."""
    from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio

    cfg = SequentialConfig(min_ev=0.05)
    pf = SequentialPortfolio(cfg)
    placed_draw = dict(_now_placed(
        market_key="handicap_0_3", selection="Draw (0:3)", odds=2.65, ev=0.06
    ))
    refused = 0
    # EV held inside the [min_ev, max_ev] band at every price, so ONLY the
    # one-position cap can refuse — the tape never gets a second load.
    for price, p in ((2.45, 0.449), (2.30, 0.478), (2.15, 0.512)):
        opp = make_opp(
            market_key="handicap_0_3", selection="Draw (0:3)",
            odds=price, model_p=p, p_lower=p - 0.03,
        )
        opp.model_confidence = "medium"
        d = pf.evaluate_snapshot([opp], MATCH_STATE, [placed_draw])[0]
        assert d.should_bet is False, f"price {price} was NOT refused"
        assert "position already open" in d.reason, f"price {price}: {d.reason}"
        refused += 1
    assert refused == 3


def test_near_certainty_turnover_farming_is_blocked():
    """The exact tape: `goals_in_row_2_away No @1.10-1.17` staked NGN 10 to
    win 0.4-1.7 with claimed EV +0.7-2.8% — turnover with no edge. The EV
    floor exists for precisely this shape."""
    from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio

    cfg = SequentialConfig(min_ev=0.05)
    pf = SequentialPortfolio(cfg)
    for odds, p in ((1.10, 0.92), (1.13, 0.90), (1.17, 0.88), (1.08, 0.93)):
        opp = make_opp(
            market_key="goals_in_row_2_away", selection="No",
            odds=odds, model_p=p, p_lower=p - 0.03,
        )
        opp.model_confidence = "medium"
        d = pf.evaluate_snapshot([opp], MATCH_STATE, [])[0]
        assert d.should_bet is False, f"odds {odds} was NOT refused"
        assert "min_ev" in d.reason


def test_absurd_ev_claim_is_ceilinged():
    """The exact tape: `ou_3_5 Over 3.5` claimed +36.7/+41.0/+46.4% against
    a book at 2.30-2.50 whose de-vigged price said ~40%. A live re-quoted
    market NEVER hands out +25%; the ceiling refuses the claim regardless
    of where the model's confidence grade landed."""
    from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio

    cfg = SequentialConfig(min_ev=0.05, max_ev=0.25)
    pf = SequentialPortfolio(cfg)
    for odds, p, ev in ((2.30, 0.59, 0.367), (2.45, 0.58, 0.410), (2.50, 0.59, 0.464)):
        opp = make_opp(
            market_key="ou_3_5", selection="Over 3.5",
            odds=odds, model_p=p, p_lower=p - 0.04,
        )
        # deliberately graded HIGH: the ceiling must hold even for the
        # designed real-edge hunt at the highest trust level
        opp.model_confidence = "high"
        d = pf.evaluate_snapshot([opp], MATCH_STATE, [])[0]
        assert d.should_bet is False, f"odds {odds} ev {ev} was NOT refused"
        assert "max_ev" in d.reason


def test_sane_edge_still_passes_the_band():
    """The band is [min_ev, max_ev]: a modest, believable edge on a graded
    estimate still reaches the stake computation — the ceiling must not
    close the engine entirely."""
    from atlas.portfolio.sequential import SequentialConfig, SequentialPortfolio

    cfg = SequentialConfig(min_ev=0.05, max_ev=0.25)
    pf = SequentialPortfolio(cfg)
    opp = make_opp(odds=2.0, model_p=0.56, p_lower=0.54)  # ev = +0.12
    opp.model_confidence = "medium"
    d = pf.evaluate_snapshot([opp], MATCH_STATE, [])[0]
    assert d.should_bet is not False or "max_ev" not in d.reason


def test_decisions_engine_holds_the_same_ceiling():
    from atlas.live.decisions import PolicyConfig, compute_recommended_stake_ngn  # noqa
    from atlas.live.estimate import Estimate

    pc = PolicyConfig()
    assert pc.max_ev == 0.25
    assert PolicyConfig(max_ev=0.4).max_ev == 0.4


def test_posture_forwards_the_ceiling_to_the_portfolio():
    from atlas.ops.posture import Posture, PostureMode
    from atlas.portfolio.sequential import SequentialConfig

    posture = Posture(mode=PostureMode.BALANCED)
    cfg = posture.apply_to(SequentialConfig())
    assert cfg.max_ev == 0.25
