"""Phase 2 acceptance: MA-cross on synthetic daily bars matches a fully
hand-computed P&L, fills happen at NEXT bar open, and CFD financing accrues
per calendar day (weekend = 3 days).
"""

import pandas as pd
import pytest

from backtest.broker_sim import Order
from backtest.costs import CostModel
from backtest.engine import run_backtest

CLOSES = [100, 101, 102, 103, 102, 100, 98, 97, 100, 103, 105, 106]
OPENS = [100] + CLOSES[:-1]          # open_t = close_{t-1}
QTY = 10.0

CM = CostModel(
    commission_pct=0.001, commission_min=1.0,
    half_spread_pct=0.001, slippage_pct=0.001,   # impact = 0.002
    benchmark_rate=0.05, long_markup=0.02,
    short_markdown=0.02, borrow_default=0.01, day_count=360,
)


def _bars():
    ts = pd.bdate_range("2024-01-01", periods=len(CLOSES), tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": OPENS,
        "high": [max(o, c) + 1 for o, c in zip(OPENS, CLOSES)],
        "low": [min(o, c) - 1 for o, c in zip(OPENS, CLOSES)],
        "close": CLOSES, "volume": [1000] * len(CLOSES),
    })


class MACross:
    """fast MA(2) vs slow MA(3) on closes; long-only, fixed 10 shares."""

    def on_bar_close(self, snap):
        bars = snap.bars("SPY")
        if len(bars) < 3:
            return []
        c = bars["close"]
        fast = c.iloc[-2:].mean()
        slow = c.iloc[-3:].mean()
        pos = snap.position("SPY")
        if fast > slow and pos is None:
            return [Order("SPY", +1, QTY)]
        if fast < slow and pos is not None:
            return [Order("SPY", -1, pos.qty)]
        return []


@pytest.fixture(scope="module")
def result():
    return run_backtest({"SPY": _bars()}, MACross(), initial_cash=100_000.0,
                        cost_model=CM, bar_duration=pd.Timedelta(days=1))


def test_fill_timing_next_bar_open(result):
    # crosses: fast>slow first computable at t2 close -> fill at t3 open (Jan 4)
    # fast<slow at t5 close -> exit at t6 open (Jan 9)
    # fast>slow again at t8 close -> fill at t9 open (Jan 12); open at end
    ts = pd.bdate_range("2024-01-01", periods=12, tz="UTC")
    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.entry_ts == ts[3]
    assert tr.exit_ts == ts[6]


def test_hand_computed_pnl(result):
    # ---- trade 1 (hand computation, independent of engine internals)
    buy_px = 102 * 1.002                      # 102.204
    buy_fee = max(1.0, buy_px * QTY * 0.001)  # 1.02204
    sell_px = 100 * 0.998                     # 99.8
    sell_fee = max(1.0, sell_px * QTY * 0.001)  # min applies: 1.0
    pnl1 = (sell_px - buy_px) * QTY - buy_fee - sell_fee
    assert result.trades[0].pnl == pytest.approx(pnl1)

    # ---- financing, position 1: held over Jan5(1d), Jan8(3d, weekend), Jan9(1d)
    rate = (0.05 + 0.02) / 360
    fin1 = QTY * (103 * 1 + 102 * 3 + 100 * 1) * rate
    # ---- financing, position 2 (opened Jan 12): Jan15(3d), Jan16(1d)
    fin2 = QTY * (103 * 3 + 105 * 1) * rate
    assert result.total_financing == pytest.approx(fin1 + fin2)

    # ---- full equity reconciliation at the last bar close
    buy2_px = 100 * 1.002
    buy2_fee = max(1.0, buy2_px * QTY * 0.001)
    cash = (100_000.0
            - buy_px * QTY - buy_fee
            + sell_px * QTY - sell_fee
            - buy2_px * QTY - buy2_fee
            - fin1 - fin2)
    expected_equity = cash + QTY * 106       # mark last position at close
    assert result.final_equity == pytest.approx(expected_equity, abs=1e-9)
    assert result.total_commission == pytest.approx(buy_fee + sell_fee + buy2_fee)


def test_no_lookahead_at_engine_level():
    """Perturbing bars AFTER t must not change any decision made up to t."""
    bars_a = _bars()
    bars_b = _bars()
    bars_b.loc[8:, ["open", "high", "low", "close"]] += 50  # diverge from t8

    ra = run_backtest({"SPY": bars_a}, MACross(), 100_000.0, CM, pd.Timedelta(days=1))
    rb = run_backtest({"SPY": bars_b}, MACross(), 100_000.0, CM, pd.Timedelta(days=1))
    # trade 1 fully decided/filled before t8: identical in both runs
    assert ra.trades[0].entry_price == rb.trades[0].entry_price
    assert ra.trades[0].pnl == pytest.approx(rb.trades[0].pnl)
    # equity curves identical strictly before the divergence bar's close
    ts = pd.bdate_range("2024-01-01", periods=12, tz="UTC")
    cutoff = ts[8]
    pd.testing.assert_series_equal(
        ra.equity_curve[ra.equity_curve.index <= cutoff],
        rb.equity_curve[rb.equity_curve.index <= cutoff])
