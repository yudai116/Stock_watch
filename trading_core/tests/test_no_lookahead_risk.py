"""Mandatory no-lookahead + hand-computed tests for the risk layer.

Risk functions are pure functions of point-in-time inputs; the lookahead
test verifies dd_control depends only on the RUNNING peak (past equity),
never on future equity values.
"""

import pytest

from risk import dd_control, portfolio_limits, vol_target_sizer
from risk.confidence_multiplier import confidence, sentiment_alignment

PARAMS = {"risk_per_trade_pct": 1.0, "chandelier_k": 3.0}


def test_vol_target_size_hand_computed():
    # size = (100_000 * 1%) / (3 * ATR 2.0) = 166.66 shares
    q = vol_target_sizer.position_size(100_000, 2.0, PARAMS, side=1)
    assert q == pytest.approx(100_000 * 0.01 / 6.0)


def test_short_size_cap():
    ql = vol_target_sizer.position_size(100_000, 2.0, PARAMS, side=1)
    qs = vol_target_sizer.position_size(100_000, 2.0, PARAMS, side=-1)
    assert qs == pytest.approx(ql * 0.6)      # params.yaml short_size_cap_vs_long


def test_size_guards():
    assert vol_target_sizer.position_size(100_000, 0.0, PARAMS) == 0.0
    assert vol_target_sizer.position_size(0.0, 2.0, PARAMS) == 0.0
    # notional sanity cap: 25% of equity
    q = vol_target_sizer.position_size(100_000, 0.01, PARAMS, price=100.0)
    assert q * 100.0 <= 25_000 + 1e-9


def test_dd_control_uses_running_peak_only():
    # equity path: rises to 120k then falls; multiplier at each t uses the
    # peak SO FAR — extending the future path cannot change earlier values
    path = [100_000, 110_000, 120_000, 112_000, 100_000, 90_000]
    peaks, mults = [], []
    peak = 0.0
    for eq in path:
        peak = max(peak, eq)
        mults.append(dd_control.risk_multiplier(eq, peak))
    assert mults[0] == 1.0 and mults[1] == 1.0 and mults[2] == 1.0
    assert mults[3] == 1.0                        # dd 6.7% < soft 10%
    dd4 = 1 - 100_000 / 120_000                   # 16.67% between soft/hard
    frac = (dd4 - 0.10) / (0.20 - 0.10)
    assert mults[4] == pytest.approx(1 - frac * (1 - 0.3))
    assert mults[5] == 0.0                        # dd 25% >= hard 20%
    # future perturbation: recompute first 4 values with a different tail
    peak2, mults2 = 0.0, []
    for eq in path[:4] + [500_000]:
        peak2 = max(peak2, eq)
        mults2.append(dd_control.risk_multiplier(eq, peak2))
    assert mults2[:4] == mults[:4]


def test_portfolio_limits():
    positions = {"NVDA": object(), "AMD": object(), "MU": object()}
    sectors = {"NVDA": "semis", "AMD": "semis", "MU": "semis", "MSFT": "software",
               "AVGO": "semis"}
    # sector cap (max 3 per sector)
    ok, why = portfolio_limits.can_open(positions, sectors, "AVGO", 1.0, 3.0)
    assert not ok and why == "sector_concentration"
    ok, _ = portfolio_limits.can_open(positions, sectors, "MSFT", 1.0, 3.0)
    assert ok
    # heat cap (6%)
    ok, why = portfolio_limits.can_open(positions, sectors, "MSFT", 1.0, 5.5)
    assert not ok and why == "portfolio_heat"
    # duplicate
    ok, why = portfolio_limits.can_open(positions, sectors, "NVDA", 1.0, 0.0)
    assert not ok and why == "already_open"
    # max positions
    many = {f"S{i}": object() for i in range(8)}
    ok, why = portfolio_limits.can_open(many, {}, "MSFT", 1.0, 0.0)
    assert not ok and why == "max_positions"


def test_confidence_range_fixed():
    # extremes stay inside [0.5, 1.5] (SPEC §6.3)
    assert confidence(1.0, +1, 1.0) == pytest.approx(1.5)
    assert confidence(0.0, +1, -1.0) == pytest.approx(0.5)
    assert confidence(None, +1, None) == 1.0            # no data -> neutral
    mid = confidence(0.5, +1, 0.0)
    assert mid == pytest.approx(1.0)


def test_sentiment_alignment_direction():
    assert sentiment_alignment(+1, 0.8) > 0.5           # long + positive news
    assert sentiment_alignment(-1, 0.8) < 0.5           # short + positive news
    assert sentiment_alignment(-1, -0.8) > 0.5          # short + negative news
    assert sentiment_alignment(+1, None) == 0.5
