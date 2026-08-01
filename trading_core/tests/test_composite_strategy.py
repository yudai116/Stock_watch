"""End-to-end composite strategy v2 on synthetic DAILY data (R2):
breakout entry under bull regime, intrabar chandelier stops, time stop,
bear-regime index hedge (R1), crisis behaviour, and the engine-level
no-lookahead property.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CASH, CFD, CostModel, InstrumentCosts
from backtest.engine import run_backtest
from features.daily_features import build_signal_frame
from signals.composite import SwingStrategy

_EXEC = dict(commission_pct=0.0005, commission_min=1.0,
             half_spread_pct=0.0003, slippage_pct=0.0005)
CM = CostModel(profiles={
    CASH: InstrumentCosts(**_EXEC, financing_enabled=False),
    CFD: InstrumentCosts(**_EXEC, financing_enabled=True,
                         benchmark_rate=0.04, long_markup=0.035,
                         short_rate=0.0, day_count=360),
})

PARAMS = {
    "donchian_entry_period": 20, "atr_period": 14, "chandelier_k": 3.0,
    "time_stop_days": 15, "partial_take_profit": "none",
    "rs_top_fraction": 1.0, "volume_z_min": -10.0,       # permissive for test
    "mr_enabled": False, "mr_rsi_period": 14, "mr_rsi_entry": 20.0,
    "mr_zscore_entry": -1.5, "risk_per_trade_pct": 1.0,
}
N = 400


def _daily_bars(n=N, seed=9, drift=0.002, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + rng.normal(drift, 0.01, n))
    ts = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": np.r_[close[0], close[:-1]],
        "high": close * 1.012, "low": close * 0.988,
        "close": close, "volume": rng.integers(1_000, 5_000, n).astype(float),
    })


def _day_close_index(bars):
    return pd.DatetimeIndex(bars["ts"]) + pd.Timedelta(days=1)


def _strategy(stock_bars: dict, regime_labels) -> SwingStrategy:
    """``regime_labels``: constant string or Series aligned to day closes."""
    some = next(iter(stock_bars.values()))
    didx = _day_close_index(some)
    regime = (regime_labels if isinstance(regime_labels, pd.Series)
              else pd.Series(regime_labels, index=didx))
    feats = {s: build_signal_frame(b, PARAMS) for s, b in stock_bars.items()}
    rs = pd.DataFrame({s: 1.0 for s in stock_bars}, index=didx)
    return SwingStrategy(features=feats, regime=regime, rs_rank=rs,
                         params=PARAMS,
                         sector_by_symbol={s: "semis" for s in stock_bars})


def test_bull_trend_produces_cash_equity_longs():
    bars = _daily_bars()
    res = run_backtest({"NVDA": bars}, _strategy({"NVDA": bars}, "bull"),
                       100_000.0, CM, close_at_end=True)
    assert len(res.trades) >= 1
    assert all(t.side == 1 and t.instrument == CASH for t in res.trades)
    assert res.total_financing == 0.0            # R1: cash longs, no financing
    for t in res.trades:
        assert t.qty * t.entry_price < 0.3 * 100_000


def test_bear_regime_blocks_new_longs():
    bars = _daily_bars()
    res = run_backtest({"NVDA": bars}, _strategy({"NVDA": bars}, "bear"),
                       100_000.0, CM, close_at_end=True)
    assert all(t.instrument == CFD for t in res.trades)   # no stock trades


def test_bear_switch_opens_index_hedge():
    """Bull -> bear at day 250: an open long book must get a QQQ CFD short."""
    nvda = _daily_bars(seed=9, drift=0.003)
    qqq = _daily_bars(seed=21, drift=0.001, start=300.0)
    didx = _day_close_index(nvda)
    labels = pd.Series("bull", index=didx)
    labels.iloc[250:] = "bear"

    strat = _strategy({"NVDA": nvda}, labels)
    res = run_backtest({"NVDA": nvda, "QQQ": qqq}, strat, 100_000.0, CM,
                       close_at_end=True)
    hedge_trades = [t for t in res.trades if t.symbol == "QQQ"]
    assert hedge_trades, "hedge short was never opened"
    assert all(t.side == -1 and t.instrument == CFD for t in hedge_trades)
    # hedge must not exceed the 0.7 beta-exposure cap vs a ~25k stock book
    ht = hedge_trades[0]
    assert ht.qty * ht.entry_price <= 0.7 * 0.3 * 100_000 * 1.5


def test_crisis_regime_no_new_entries():
    bars = _daily_bars()
    res = run_backtest({"NVDA": bars}, _strategy({"NVDA": bars}, "crisis"),
                       100_000.0, CM, close_at_end=True)
    assert [t for t in res.trades if t.instrument == CASH] == []


def test_trailing_stop_fires_intrabar_on_reversal():
    bars = _daily_bars(drift=0.003, seed=4)
    rev = bars.copy()
    n = len(rev)
    decay = np.cumprod(np.full(n - 250, 0.99))
    for col in ["open", "high", "low", "close"]:
        rev.loc[250:, col] = rev.loc[249, col] * decay
    res = run_backtest({"NVDA": rev}, _strategy({"NVDA": rev}, "bull"),
                       100_000.0, CM, close_at_end=True)
    assert "stop" in {t.reason for t in res.trades}


def test_time_stop_fires_when_position_stalls():
    """Surgical check: a stalled long past the N-business-day limit with
    <0.5R progress must be closed."""
    from backtest.broker_sim import BrokerSim, Position
    from backtest.engine import MarketSnapshot

    bars = _daily_bars(drift=0.0, seed=2)
    strat = _strategy({"NVDA": bars}, "bull")
    broker = BrokerSim(100_000.0, CM)
    i = 300
    entry_i = 275                              # ~25 business days earlier
    entry_px = float(bars["close"].iloc[i - 1])
    broker.positions["NVDA"] = Position(
        symbol="NVDA", side=1, qty=100.0,
        entry_price=entry_px, entry_ts=bars["ts"].iloc[entry_i],
        instrument=CASH,
        meta={"stop0": entry_px - 1e6, "extreme": entry_px, "r_unit": 1e6},
    )
    snap = MarketSnapshot({"NVDA": bars}, {"NVDA": i},
                          bars["ts"].iloc[i - 1] + pd.Timedelta(days=1), broker)
    orders = strat.on_bar_close(snap)
    assert any(o.reason == "time_stop" and o.symbol == "NVDA" for o in orders)


def test_strategy_no_lookahead_future_perturbation():
    K = 250
    a = _daily_bars()
    b = a.copy()
    b.loc[K + 1:, ["open", "high", "low", "close"]] *= 0.5

    ra = run_backtest({"NVDA": a}, _strategy({"NVDA": a}, "bull"),
                      100_000.0, CM, close_at_end=True)
    rb = run_backtest({"NVDA": b}, _strategy({"NVDA": b}, "bull"),
                      100_000.0, CM, close_at_end=True)
    cutoff = a["ts"].iloc[K]
    ta = [(t.entry_ts, round(t.entry_price, 9)) for t in ra.trades if t.entry_ts <= cutoff]
    tb = [(t.entry_ts, round(t.entry_price, 9)) for t in rb.trades if t.entry_ts <= cutoff]
    assert ta == tb
    ea = ra.equity_curve[ra.equity_curve.index <= cutoff]
    eb = rb.equity_curve[rb.equity_curve.index <= cutoff]
    pd.testing.assert_series_equal(ea, eb)
