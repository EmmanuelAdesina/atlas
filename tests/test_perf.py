"""Performance-grading tests (atlas.ops.perf).

These lock the stake-policy rule: the sample size gates every verdict,
because today's NGN 100 loss (odds 10.0) versus yesterday's three 1.1x
winners proves how violently a small sample lies in both directions.
"""

import pytest

from atlas.ops.perf import settled_stats, verdict


def _bet(ix, odds, status, stake=10.0, pnl=0.0):
    return {
        "bet_id": f"b{ix}", "stake_ngn": stake, "odds": odds,
        "status": status, "pnl_ngn": pnl,
    }


def test_unsetled_ledger_reports_the_honest_unknown():
    stats = settled_stats([_bet(1, 2.0, "placed")])
    assert stats["settled"] == 0
    assert "sample too small" in verdict(0, None)


def test_small_sample_is_never_a_verdict():
    three_wins = [_bet(i, 1.11, "won", 100, 11) for i in range(3)]
    stats = settled_stats(three_wins)
    assert stats["roi"] > 0
    assert "sample too small" in stats["verdict"]


def test_beating_the_book_is_said_plainly():
    # 30 bets at 2.0, hit 16/30 -> pnl = +20, staked 300, ROI = +6.7%
    bets = (
        [_bet(i, 2.0, "won", 10, 10) for i in range(16)]
        + [_bet(i + 100, 2.0, "lost", 10, -10) for i in range(14)]
    )
    stats = settled_stats(bets)
    assert stats["hit_rate"] == pytest.approx(16 / 30, abs=1e-3)
    assert stats["verdict"].startswith("ROI +6.7%")
    assert "beating the book" in stats["verdict"]


def test_loss_streak_counts_consecutive_recent_losses():
    bets = [
        _bet(1, 1.5, "won", 10, 5),
        _bet(2, 2.0, "lost", 10, -10),
        _bet(3, 2.0, "lost", 10, -10),
        _bet(4, 2.5, "lost", 10, -10),
        _bet(5, 2.0, "placed"),
    ]
    assert settled_stats(bets)["streak_losses"] == 3


def test_bands_separate_short_odds_from_longshots():
    bets = [
        _bet(1, 1.20, "won", 10, 2),
        _bet(2, 2.50, "lost", 10, -10),
        _bet(3, 5.00, "won", 10, 40),
    ]
    stats = settled_stats(bets)
    bands = stats["by_band"]
    assert set(bands) == {"short <1.5", "mid 1.5-3", "long >3"}
    assert bands["long >3"]["hit_rate"] == 1.0
    # break-even hit rate for odds 5.00 is 0.2
    assert bands["long >3"]["breakeven_hit_rate"] == pytest.approx(0.2)


def test_void_bets_are_excluded():
    stats = settled_stats([_bet(1, 2.0, "void"), _bet(2, 2.0, "won", 10, 10)])
    assert stats["settled"] == 1
