"""Phase 3a weekly RS rotation (SPEC_ADDENDUM_v2 H / flow [C]):
top-N selection by trailing return, bull-only exposure, cash in bear,
rebalance cadence, and no-lookahead."""

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CASH, CFD, CostModel, InstrumentCosts
from backtest.engine import run_backtest
from signals.rs_rotation import RotationStrategy

_EXEC = dict(commission_pct=0.0005, commission_min=1.0,
             half_spread_pct=0.0003, slippage_pct=0.0005)
CM = CostModel(profiles={
    CASH: InstrumentCosts(**_EXEC, financing_enabled=False),
    CFD: InstrumentCosts(**_EXEC, financing_enabled=True,
                         benchmark_rate=0.04, long_markup=0.035),
})

PARAMS = {"lookback_days": 63, "n_holdings": 2, "rebalance_days": 5}
N = 300
DRIFTS = {"AAA": 0.0030, "BBB": 0.0015, "CCC": -0.0010, "DDD": 0.0000}


def _bars(drift, seed, n=N):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.008, n))
    ts = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": np.r_[close[0], close[:-1]],
        "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": [1000.0] * n,
    })


def _panel():
    return {s: _bars(d, seed=i) for i, (s, d) in enumerate(DRIFTS.items())}


def _regime(label, bars):
    didx = pd.DatetimeIndex(bars["ts"]) + pd.Timedelta(days=1)
    return pd.Series(label, index=didx)


def test_holds_top_momentum_names_in_bull():
    panel = _panel()
    strat = RotationStrategy(regime=_regime("bull", panel["AAA"]), params=PARAMS)
    res = run_backtest(panel, strat, 100_000.0, CM, close_at_end=True)
    entered = {t.symbol for t in res.trades}
    assert "AAA" in entered                    # strongest momentum always held
    assert "CCC" not in entered                # negative drift never in top 2
    assert all(t.side == 1 and t.instrument == CASH for t in res.trades)
    # any single position stays within the 95% equity budget at all times
    peak = float(res.equity_curve.max())
    for t in res.trades:
        assert t.qty * t.entry_price < 0.95 * peak


def test_bear_regime_goes_to_cash():
    panel = _panel()
    strat = RotationStrategy(regime=_regime("bear", panel["AAA"]), params=PARAMS)
    res = run_backtest(panel, strat, 100_000.0, CM, close_at_end=True)
    assert res.trades == []
    assert res.final_equity == pytest.approx(100_000.0)


def test_bull_to_bear_liquidates():
    panel = _panel()
    didx = pd.DatetimeIndex(panel["AAA"]["ts"]) + pd.Timedelta(days=1)
    labels = pd.Series("bull", index=didx)
    labels.iloc[200:] = "bear"
    strat = RotationStrategy(regime=labels, params=PARAMS)
    res = run_backtest(panel, strat, 100_000.0, CM, close_at_end=True)
    out_reasons = {t.reason for t in res.trades}
    assert "rotate_out" in out_reasons
    # after the bear switch (+ one rebalance window), nothing stays open
    assert not any(t.reason == "end_of_data" for t in res.trades)


def test_rebalance_cadence():
    """Entries may only occur on rebalance days (every 5 trading days)."""
    panel = _panel()
    strat = RotationStrategy(regime=_regime("bull", panel["AAA"]), params=PARAMS)
    res = run_backtest(panel, strat, 100_000.0, CM, close_at_end=True)
    entry_days = sorted({t.entry_ts for t in res.trades})
    assert len(entry_days) >= 1
    # every entry fill is the bar after a rebalance decision; decisions are
    # >= 5 trading days apart, so distinct entry days are >= 5 bdays apart
    for a, b in zip(entry_days[:-1], entry_days[1:]):
        assert np.busday_count(a.date(), b.date()) >= 5


def test_benchmark_symbol_never_traded():
    panel = _panel()
    panel["QQQ"] = _bars(0.001, seed=99)
    strat = RotationStrategy(regime=_regime("bull", panel["AAA"]), params=PARAMS,
                             benchmark_symbols=("QQQ",))
    res = run_backtest(panel, strat, 100_000.0, CM, close_at_end=True)
    assert "QQQ" not in {t.symbol for t in res.trades}


def test_rotation_no_lookahead():
    K = 200
    pa = _panel()
    pb = {s: b.copy() for s, b in pa.items()}
    for b in pb.values():
        b.loc[K + 1:, ["open", "high", "low", "close"]] *= 0.5

    sa = RotationStrategy(regime=_regime("bull", pa["AAA"]), params=PARAMS)
    sb = RotationStrategy(regime=_regime("bull", pb["AAA"]), params=PARAMS)
    ra = run_backtest(pa, sa, 100_000.0, CM, close_at_end=True)
    rb = run_backtest(pb, sb, 100_000.0, CM, close_at_end=True)
    cutoff = pa["AAA"]["ts"].iloc[K]
    ta = [(t.symbol, t.entry_ts) for t in ra.trades if t.entry_ts <= cutoff]
    tb = [(t.symbol, t.entry_ts) for t in rb.trades if t.entry_ts <= cutoff]
    assert sorted(ta) == sorted(tb)
