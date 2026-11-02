"""The sizing bankroll is the observed wallet, never a fantasy figure.

2026-11-02: the unit's `--bankroll 100000` was removed AND the argparse
default retired — but default-removal alone was a silent no-op
(argparse would just reinsert 100000 from `default=`). These tests pin
the new contract: 0 means 'the observed account balance is the whole
figure' and no observation + no explicit figure stakes nothing
(fail-closed), never 100k.
"""
import pytest

from atlas.live.runner import effective_bankroll


def test_observed_balance_is_the_whole_figure_when_bankroll_zero():
    """The operator stance: --bankroll 0 -> observed balance governs."""
    assert effective_bankroll(0.0, 49.0) == pytest.approx(49.0)
    assert effective_bankroll(0, 1_337.5) == pytest.approx(1_337.5)


def test_explicit_figure_is_only_a_ceiling():
    """An explicit --bankroll > 0 can only LOWER the observed wallet."""
    assert effective_bankroll(100_000.0, 49.0) == pytest.approx(49.0)
    assert effective_bankroll(40.0, 49.0) == pytest.approx(40.0)


def test_no_observation_and_zero_figures_stakes_nothing():
    """FAIL-CLOSED: no observed balance, no explicit figure -> 0 bankroll
    (the fantasy 100k default is retired; nothing is sizeable)."""
    for none_like in (None, 0.0, -5.0, "49"):
        assert effective_bankroll(0.0, none_like) == pytest.approx(0.0), none_like


def test_no_observation_keeps_explicit_non_zero_figures():
    """No observation yet: an explicit positive figure still sizes (the
    pre-existing behaviour for explicitly-run figures is kept)."""
    assert effective_bankroll(100_000.0, None) == pytest.approx(100_000.0)


def test_argparse_default_is_zero_observed_only():
    """The retire-the-default pin: `run` with NO explicit --bankroll gets
    0 (observed-balance-only), never the fantasy 100_000.0 — removing the
    unit's flag is only a fix if omitted ALSO means observed-only."""
    from atlas.main import build_parser

    args = build_parser().parse_args(["run"])
    assert args.bankroll == 0.0
