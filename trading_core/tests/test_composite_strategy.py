"""End-to-end composite strategy on synthetic hourly data:
breakout entry under bull regime, chandelier/time-stop exits, regime blocks,
and the engine-level no-lookahead property.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import run_backtest
from features.hourly_features import build_hourly_frame
from signals.composite import SwingStrategy

CM = CostModel(commission_pct=0.0005, commission_min=1.0,
               half_spread_pct=0.0003, slippage_pct=0.0005,
               benchmark_rate=0.04, long_markup=0.02,
               short_markdown=0.02, borrow_default=0.005, day_count=360)

PARAMS = {
    "donchian_entry_period": 20, "atr_period": 14, "chandelier_k": 3.0,
    "time_stop_days": 15, "partial_take_profit": "none",
    "rs_top_fraction": 1.0, "volume_z_min": -10.0,       # permissive for test
    "mr_enabled": False, "mr_rsi_period": 14, "mr_rsi_entry": 20.0,
    "mr_zscore_entry": -1.5, "short_donchian_period": 20,
    "risk_per_trade_pct": 1.0,
}


def _trending_bars(n=400, seed=9, drift=0.004):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.004, n)
    close = 100 * np.cumprod(1 + ret)
    # 7 bars/day approximates US equity RTH hourly bars
    days = pd.bdate_range("2024-01-02", periods=n // 7 + 1, tz="UTC")
    ts = pd.DatetimeIndex([d + pd.Timedelta(hours=14 + h) for d in days for h in range(7)][:n])
    return pd.DataFrame({
        "ts": ts, "open": np.r_[close[0], close[:-1]],
        "high": close * 1.004, "low": close * 0.996,
        "close": close, "volume": rng.integers(1_000, 5_000, n).astype(float),
    })


def _daily_index(bars):
    return pd.DatetimeIndex(sorted({t.normalize() + pd.Timedelta(hours=21) for t in bars["ts"]}))


def _strategy(bars, regime_label="bull"):
    feats = {"NVDA": build_hourly_frame(bars, PARAMS)}
    didx = _daily_index(bars)
    regime = pd.Series(regime_label, index=didx)
    rs = pd.DataFrame({"NVDA": 1.0}, index=didx)
    return SwingStrategy(features=feats, regime=regime, rs_rank=rs,
                         params=PARAMS, sector_by_symbol={"NVDA": "semis"})


def test_bull_trend_produces_long_trades():
    bars = _trending_bars()
    res = run_backtest({"NVDA": bars}, _strategy(bars), 100_000.0, CM,
                       close_at_end=True)
    assert len(res.trades) >= 1
    assert all(t.side == 1 for t in res.trades)
    # sizing respects vol targeting: no trade risks more than ~2% of equity
    for t in res.trades:
        assert t.qty * t.entry_price < 0.3 * 100_000


def test_bear_regime_blocks_longs():
    bars = _trending_bars()
    res = run_backtest({"NVDA": bars}, _strategy(bars, "bear"), 100_000.0, CM,
                       close_at_end=True)
    assert all(t.side == -1 for t in res.trades)        # shorts only, if any


def test_crisis_regime_no_new_entries():
    bars = _trending_bars()
    res = run_backtest({"NVDA": bars}, _strategy(bars, "crisis"), 100_000.0, CM)
    assert len(res.trades) == 0
    assert res.final_equity == pytest.approx(100_000.0)  # never traded


def test_trailing_stop_fires_on_reversal():
    bars = _trending_bars(drift=0.0035, seed=4)
    # steady decline after bar 140: an open long must hit the chandelier stop
    rev = bars.copy()
    n = len(rev)
    decay = np.cumprod(np.full(n - 140, 0.997))
    for col in ["open", "high", "low", "close"]:
        rev.loc[140:, col] = rev.loc[139, col] * decay
    res = run_backtest({"NVDA": rev}, _strategy(rev), 100_000.0, CM,
                       close_at_end=True)
    assert "stop" in {t.reason for t in res.trades}


def test_time_stop_fires_when_position_stalls():
    """Surgical check of the time-stop exit path: a stalled long past the
    N-business-day limit with <0.5R progress must be closed."""
    from backtest.broker_sim import BrokerSim, Position
    from backtest.engine import MarketSnapshot

    bars = _trending_bars(drift=0.0, seed=2)   # sideways market
    strat = _strategy(bars)
    broker = BrokerSim(100_000.0, CM)
    i = 300                                    # decision at bar 300 close
    entry_i = 180                              # entered ~17 business days earlier
    # entry price == current close: exactly 0R progress, trailing stop not hit
    entry_px = float(bars["close"].iloc[i])
    broker.positions["NVDA"] = Position(
        symbol="NVDA", side=1, qty=100.0,
        entry_price=entry_px, entry_ts=bars["ts"].iloc[entry_i],
        meta={"stop0": entry_px - 1e6, "extreme": entry_px, "r_unit": 1e6},
    )
    snap = MarketSnapshot({"NVDA": bars}, {"NVDA": i + 1},
                          bars["ts"].iloc[i] + pd.Timedelta(hours=1), broker)
    orders = strat.on_bar_close(snap)
    assert any(o.reason == "time_stop" and o.symbol == "NVDA" for o in orders)


def test_strategy_no_lookahead_future_perturbation():
    """Identical history through bar K, divergent after: all decisions made
    up to K's close must match (entries recorded at fill = K+1 open at latest)."""
    K = 250
    a = _trending_bars()
    b = a.copy()
    b.loc[K + 1:, ["open", "high", "low", "close"]] *= 0.5

    ra = run_backtest({"NVDA": a}, _strategy(a), 100_000.0, CM, close_at_end=True)
    rb = run_backtest({"NVDA": b}, _strategy(b), 100_000.0, CM, close_at_end=True)
    cutoff = a["ts"].iloc[K]
    ta = [(t.entry_ts, round(t.entry_price, 9)) for t in ra.trades if t.entry_ts <= cutoff]
    tb = [(t.entry_ts, round(t.entry_price, 9)) for t in rb.trades if t.entry_ts <= cutoff]
    assert ta == tb
    ea = ra.equity_curve[ra.equity_curve.index <= cutoff]
    eb = rb.equity_curve[rb.equity_curve.index <= cutoff]
    pd.testing.assert_series_equal(ea, eb)
