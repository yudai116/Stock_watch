"""Flow-F [D] overlay: invested in bull, cash in bear, hold in range;
DD compression vs buy & hold on a bear-containing path."""

import numpy as np
import pandas as pd
import pytest

from backtest.costs import CASH, CFD, CostModel, InstrumentCosts
from backtest.engine import run_backtest
from signals.regime_overlay import RegimeOverlayStrategy

_EXEC = dict(commission_pct=0.0005, commission_min=1.0,
             half_spread_pct=0.0003, slippage_pct=0.0005)
CM = CostModel(profiles={
    CASH: InstrumentCosts(**_EXEC, financing_enabled=False),
    CFD: InstrumentCosts(**_EXEC, financing_enabled=True,
                         benchmark_rate=0.04, long_markup=0.035),
})


def _qqq(n=400, seed=5):
    rng = np.random.default_rng(seed)
    # bull, then a -30% grind, then recovery
    ret = np.r_[rng.normal(0.0015, 0.008, 150),
                rng.normal(-0.0035, 0.014, 120),
                rng.normal(0.0015, 0.008, n - 270)]
    close = 300 * np.cumprod(1 + ret)
    ts = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    return pd.DataFrame({"ts": ts, "open": np.r_[close[0], close[:-1]],
                         "high": close * 1.008, "low": close * 0.992,
                         "close": close, "volume": [1e6] * n})


def _labels(bars, spec):
    """spec: list of (n_days, label)."""
    idx = pd.DatetimeIndex(bars["ts"]) + pd.Timedelta(days=1)
    out = pd.Series(index=idx, dtype=object)
    i = 0
    for n, lab in spec:
        out.iloc[i:i + n] = lab
        i += n
    return out.ffill()


def test_bull_enters_bear_exits():
    bars = _qqq()
    labels = _labels(bars, [(150, "bull"), (120, "bear"), (130, "bull")])
    strat = RegimeOverlayStrategy(regime=labels, symbol="QQQ")
    res = run_backtest({"QQQ": bars}, strat, 100_000.0, CM, close_at_end=True)
    reasons = [t.reason for t in res.trades]
    assert "overlay_exit" in reasons               # went to cash in the bear
    assert all(t.instrument == CASH and t.side == 1 for t in res.trades)

    # DD compression: overlay must draw down less than buy & hold
    bh = bars["close"].iloc[-1] / bars["close"].iloc[0]
    curve = res.equity_curve
    dd = ((curve - curve.cummax()) / curve.cummax()).min()
    px = bars["close"]
    bh_dd = ((px - px.cummax()) / px.cummax()).min()
    assert dd > bh_dd + 0.05                       # >=5pt shallower drawdown


def test_range_holds_without_new_buys():
    bars = _qqq(seed=7)
    labels = _labels(bars, [(100, "bull"), (300, "range")])
    strat = RegimeOverlayStrategy(regime=labels, symbol="QQQ")
    res = run_backtest({"QQQ": bars}, strat, 100_000.0, CM, close_at_end=True)
    # exactly one entry (bull), never sold until end_of_data
    assert [t.reason for t in res.trades] == ["end_of_data"]


def test_never_unknown_entry():
    bars = _qqq(seed=8)
    labels = _labels(bars, [(400, "bear")])
    res = run_backtest({"QQQ": bars},
                       RegimeOverlayStrategy(regime=labels, symbol="QQQ"),
                       100_000.0, CM, close_at_end=True)
    assert res.trades == [] and res.final_equity == pytest.approx(100_000.0)


def test_runner_supports_3d(tmp_path):
    from validation.phase3_runner import default_params, make_strategy

    assert default_params("3d") == {}
    inputs = {"regime": pd.Series(dtype=object), "benchmark": "QQQ",
              "features": {}, "rs_rank": pd.DataFrame(), "betas": {}}
    strat = make_strategy("3d", inputs, {})
    assert isinstance(strat, RegimeOverlayStrategy)
    assert strat.symbol == "QQQ"
