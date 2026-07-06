"""Phase 2 acceptance (kept under v2): MA-cross on synthetic daily bars
matches a fully hand-computed P&L, fills happen at NEXT bar open, and CFD
financing accrues per calendar day (weekend = 3 days) while cash-equity
positions accrue nothing (SPEC_ADDENDUM_v2 R1).
"""

import pandas as pd
import pytest

from backtest.broker_sim import Order
from backtest.costs import CASH, CFD, CostModel, InstrumentCosts
from backtest.engine import run_backtest

CLOSES = [100, 101, 102, 103, 102, 100, 98, 97, 100, 103, 105, 106]
OPENS = [100] + CLOSES[:-1]          # open_t = close_{t-1}
QTY = 10.0

# identical execution frictions for both profiles so the ONLY difference in
# the paired runs below is financing (impact = 0.002, commission 0.1% min $1)
_EXEC = dict(commission_pct=0.001, commission_min=1.0,
             half_spread_pct=0.001, slippage_pct=0.001)
CM = CostModel(profiles={
    CASH: InstrumentCosts(**_EXEC, financing_enabled=False),
    CFD: InstrumentCosts(**_EXEC, financing_enabled=True,
                         benchmark_rate=0.05, long_markup=0.02,
                         short_rate=0.0, day_count=360),
})


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

    def __init__(self, instrument: str):
        self.instrument = instrument

    def on_bar_close(self, snap):
        bars = snap.bars("SPY")
        if len(bars) < 3:
            return []
        c = bars["close"]
        fast = c.iloc[-2:].mean()
        slow = c.iloc[-3:].mean()
        pos = snap.position("SPY")
        if fast > slow and pos is None:
            return [Order("SPY", +1, QTY, instrument=self.instrument)]
        if fast < slow and pos is not None:
            return [Order("SPY", -1, pos.qty, instrument=self.instrument,
                          meta={"close": True})]
        return []


@pytest.fixture(scope="module")
def result_cfd():
    return run_backtest({"SPY": _bars()}, MACross(CFD), initial_cash=100_000.0,
                        cost_model=CM, bar_duration=pd.Timedelta(days=1))


@pytest.fixture(scope="module")
def result_cash():
    return run_backtest({"SPY": _bars()}, MACross(CASH), initial_cash=100_000.0,
                        cost_model=CM, bar_duration=pd.Timedelta(days=1))


def test_fill_timing_next_bar_open(result_cfd):
    # crosses: fast>slow first computable at t2 close -> fill at t3 open (Jan 4)
    # fast<slow at t5 close -> exit at t6 open (Jan 9)
    # fast>slow again at t8 close -> fill at t9 open (Jan 12); open at end
    ts = pd.bdate_range("2024-01-01", periods=12, tz="UTC")
    assert len(result_cfd.trades) == 1
    tr = result_cfd.trades[0]
    assert tr.entry_ts == ts[3]
    assert tr.exit_ts == ts[6]


def test_hand_computed_pnl_cfd(result_cfd):
    # ---- trade 1 (hand computation, independent of engine internals)
    buy_px = 102 * 1.002                      # 102.204
    buy_fee = max(1.0, buy_px * QTY * 0.001)  # 1.02204
    sell_px = 100 * 0.998                     # 99.8
    sell_fee = max(1.0, sell_px * QTY * 0.001)  # min applies: 1.0
    pnl1 = (sell_px - buy_px) * QTY - buy_fee - sell_fee
    assert result_cfd.trades[0].pnl == pytest.approx(pnl1)

    # ---- financing, position 1: held over Jan5(1d), Jan8(3d, weekend), Jan9(1d)
    rate = (0.05 + 0.02) / 360
    fin1 = QTY * (103 * 1 + 102 * 3 + 100 * 1) * rate
    # ---- financing, position 2 (opened Jan 12): Jan15(3d), Jan16(1d)
    fin2 = QTY * (103 * 3 + 105 * 1) * rate
    assert result_cfd.total_financing == pytest.approx(fin1 + fin2)

    # ---- full equity reconciliation at the last bar close
    buy2_px = 100 * 1.002
    buy2_fee = max(1.0, buy2_px * QTY * 0.001)
    cash = (100_000.0
            - buy_px * QTY - buy_fee
            + sell_px * QTY - sell_fee
            - buy2_px * QTY - buy2_fee
            - fin1 - fin2)
    expected_equity = cash + QTY * 106       # mark last position at close
    assert result_cfd.final_equity == pytest.approx(expected_equity, abs=1e-9)
    assert result_cfd.total_commission == pytest.approx(buy_fee + sell_fee + buy2_fee)


def test_cash_equity_run_has_zero_financing(result_cash, result_cfd):
    """R1: same trades, but the cash-equity book pays NO financing —
    final equity differs from the CFD run by exactly the financing sum."""
    assert result_cash.total_financing == 0.0
    assert result_cash.trades[0].entry_ts == result_cfd.trades[0].entry_ts
    assert result_cash.final_equity == pytest.approx(
        result_cfd.final_equity + result_cfd.total_financing, abs=1e-9)


def test_no_lookahead_at_engine_level():
    """Perturbing bars AFTER t must not change any decision made up to t."""
    bars_a = _bars()
    bars_b = _bars()
    bars_b.loc[8:, ["open", "high", "low", "close"]] += 50  # diverge from t8

    ra = run_backtest({"SPY": bars_a}, MACross(CASH), 100_000.0, CM,
                      pd.Timedelta(days=1))
    rb = run_backtest({"SPY": bars_b}, MACross(CASH), 100_000.0, CM,
                      pd.Timedelta(days=1))
    assert ra.trades[0].entry_price == rb.trades[0].entry_price
    assert ra.trades[0].pnl == pytest.approx(rb.trades[0].pnl)
    ts = pd.bdate_range("2024-01-01", periods=12, tz="UTC")
    cutoff = ts[8]
    pd.testing.assert_series_equal(
        ra.equity_curve[ra.equity_curve.index <= cutoff],
        rb.equity_curve[rb.equity_curve.index <= cutoff])


# --------------------------------------------- intrabar stops (R2b, new)

class EntryThenStop:
    """Buy at first opportunity, then rest a stop at a fixed level."""

    def __init__(self, stop_price: float):
        self.stop_price = stop_price
        self.entered = False

    def on_bar_close(self, snap):
        pos = snap.position("SPY")
        orders = []
        if not self.entered and pos is None:
            orders.append(Order("SPY", +1, QTY, instrument=CASH))
            self.entered = True
        if self.entered:
            orders.append(Order("SPY", -1, QTY, instrument=CASH, reason="stop",
                                meta={"close": True, "order_type": "stop",
                                      "stop_price": self.stop_price}))
        return orders


def _stop_bars(lows, opens):
    n = len(lows)
    ts = pd.bdate_range("2024-02-01", periods=n, tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": opens,
        "high": [o + 5 for o in opens], "low": lows,
        "close": opens, "volume": [1000] * n,
    })


def test_stop_triggers_intrabar_at_stop_price():
    # entry fills at bar1 open (100); stop at 95; bar2 low touches 94
    bars = _stop_bars(lows=[99, 99, 94, 99], opens=[100, 100, 100, 100])
    res = run_backtest({"SPY": bars}, EntryThenStop(95.0), 100_000.0, CM,
                       pd.Timedelta(days=1))
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.reason == "stop"
    # fill = stop level with sell impact: 95 * (1 - 0.002)
    assert tr.exit_price == pytest.approx(95.0 * 0.998)


def test_stop_gap_down_fills_at_open():
    # bar2 gaps to open 90 (below the 95 stop): fill at the open, not the stop
    bars = _stop_bars(lows=[99, 99, 88, 99], opens=[100, 100, 90, 100])
    res = run_backtest({"SPY": bars}, EntryThenStop(95.0), 100_000.0, CM,
                       pd.Timedelta(days=1))
    assert len(res.trades) == 1
    assert res.trades[0].exit_price == pytest.approx(90.0 * 0.998)


def test_stop_not_triggered_without_touch():
    bars = _stop_bars(lows=[99, 99, 96, 97], opens=[100, 100, 100, 100])
    res = run_backtest({"SPY": bars}, EntryThenStop(95.0), 100_000.0, CM,
                       pd.Timedelta(days=1))
    assert len(res.trades) == 0            # still open at end (no close_at_end)
