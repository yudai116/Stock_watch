"""Phase 5 live runner: decision->next-open fill contract, idempotency,
regime transitions, kill switch, and halt/reset — all offline."""

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CASH, CFD, CostModel, InstrumentCosts
from data.bitemporal_store import BitemporalStore
from execution.live_runner import LiveState, realized_report, run_once

_EXEC = dict(commission_pct=0.0008, commission_min=1.0,
             half_spread_pct=0.0002, slippage_pct=0.0008)
CM = CostModel(profiles={
    CASH: InstrumentCosts(**_EXEC, financing_enabled=False),
    CFD: InstrumentCosts(**_EXEC, financing_enabled=True,
                         benchmark_rate=0.04, long_markup=0.035),
})


def _bars(n, start_i=0, level=100.0, drift=1.002):
    ts = pd.bdate_range("2024-01-01", periods=start_i + n, tz="UTC")[start_i:]
    close = np.array([level * drift ** i for i in range(start_i, start_i + n)])
    return pd.DataFrame({"ts": ts, "open": np.r_[close[0], close[:-1]],
                         "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": [1e6] * n})


@pytest.fixture()
def env(tmp_path):
    store = BitemporalStore(tmp_path / "ds")
    state = LiveState(tmp_path / "ls")
    return store, state


def test_warmup_guard(env):
    store, state = env
    store.put_bars("QQQ", "1d", _bars(50))
    out = run_once(store, state, CM)
    assert out["status"] == "warmup"


def test_bull_buys_then_fills_next_open(env):
    store, state = env
    store.put_bars("QQQ", "1d", _bars(220))
    out1 = run_once(store, state, CM)
    assert out1["status"] == "ok" and out1["regime"] == "bull"
    assert out1["action"] == "buy" and out1["qty"] == 0.0
    assert len(state.pending_orders()) == 1

    # next trading day arrives -> pending order fills at ITS open + costs
    store.put_bars("QQQ", "1d", _bars(1, start_i=220))
    out2 = run_once(store, state, CM)
    assert out2["status"] == "ok"
    assert out2["fills"] and out2["fills"][0]["side"] == 1
    open_221 = float(_bars(1, start_i=220)["open"].iloc[0])
    assert out2["fills"][0]["price"] == pytest.approx(open_221 * 1.001)  # impact
    qty = state.position_qty("QQQ")
    assert qty > 0
    # invested ~95% of equity, and no immediate re-order (delta < 1%)
    assert out2["action"] == "hold"
    assert qty * open_221 == pytest.approx(0.95 * 100_000, rel=0.05)


def test_idempotent_per_bar_date(env):
    store, state = env
    store.put_bars("QQQ", "1d", _bars(220))
    run_once(store, state, CM)
    out = run_once(store, state, CM)          # same day again
    assert out["status"] == "already_decided"
    assert len(state.pending_orders()) == 1   # no duplicate order


def test_bear_switch_liquidates(env):
    store, state = env
    store.put_bars("QQQ", "1d", _bars(220))
    run_once(store, state, CM)
    store.put_bars("QQQ", "1d", _bars(1, start_i=220))
    run_once(store, state, CM)                # long established
    qty = state.position_qty("QQQ")

    # gentle drift below the 200d MA (small daily moves: no kill switch)
    prev_close = 100 * 1.002 ** 221
    slide = _bars(60, start_i=221,
                  level=prev_close * 0.985 / (0.997 ** 221), drift=0.997)
    store.put_bars("QQQ", "1d", slide)
    out = run_once(store, state, CM)
    assert out["regime"] == "bear"
    assert out["action"] == "sell"
    # next day: exit fills, flat book
    store.put_bars("QQQ", "1d", _bars(1, start_i=281,
                                      level=slide["close"].iloc[-1] / (1.002 ** 281),
                                      drift=1.002))
    out2 = run_once(store, state, CM)
    assert out2["fills"] and out2["fills"][0]["side"] == -1
    assert state.position_qty("QQQ") == pytest.approx(0.0, abs=1e-9)
    assert qty > 0


def test_kill_switch_halts_and_reset_recovers(env):
    store, state = env
    store.put_bars("QQQ", "1d", _bars(220))
    run_once(store, state, CM)
    store.put_bars("QQQ", "1d", _bars(1, start_i=220))
    run_once(store, state, CM)                          # invested

    # overnight catastrophe: -40% gap
    level = float(_bars(1, start_i=220)["close"].iloc[0])
    crash = _bars(1, start_i=221, level=level / (1.002 ** 221) * 0.60)
    store.put_bars("QQQ", "1d", crash)
    out = run_once(store, state, CM)
    assert out["action"] == "KILL_SWITCH_LIQUIDATE"
    assert out["halted"] is True
    assert len(state.pending_orders()) == 1             # forced exit resting

    # next day: exit fills; system stays halted, no re-entry even in "bull"
    store.put_bars("QQQ", "1d", _bars(1, start_i=222,
                                      level=level / (1.002 ** 222) * 0.61))
    out2 = run_once(store, state, CM)
    assert state.position_qty("QQQ") == pytest.approx(0.0, abs=1e-9)
    assert out2["action"] == "halted"
    assert state.pending_orders() == []

    state.reset_killswitch()
    assert state.halted is False


def test_realized_report(env):
    store, state = env
    store.put_bars("QQQ", "1d", _bars(230))
    run_once(store, state, CM)
    for i in range(10):
        store.put_bars("QQQ", "1d", _bars(1, start_i=230 + i))
        run_once(store, state, CM)
    rep = realized_report(state)
    assert rep["status"] == "ok"
    assert rep["days"] >= 10
    assert "sharpe" in rep and "reference_backtest" in rep
    assert rep["halted"] is False
