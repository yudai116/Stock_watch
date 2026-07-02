"""Mandatory no-lookahead tests for the features layer (CLAUDE.md rule 2).

Method: future perturbation. Compute features on a series, then modify all
bars AFTER index k and recompute — rows <= k must be bit-identical.
"""

import numpy as np
import pandas as pd
import pytest

from features import daily_features as dfeat
from features import hourly_features as hfeat

K = 120
PARAMS = {"atr_period": 14, "donchian_entry_period": 20,
          "short_donchian_period": 20, "mr_rsi_period": 14}


def _hourly_bars(n=200, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    ts = pd.date_range("2024-01-02 14:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1000, 5000, n).astype(float),
    })


def _perturb(bars):
    out = bars.copy()
    out.loc[K + 1:, ["open", "high", "low", "close"]] *= 3.7
    out.loc[K + 1:, "volume"] *= 11
    return out


def test_hourly_features_ignore_future():
    a = hfeat.build_hourly_frame(_hourly_bars(), PARAMS)
    b = hfeat.build_hourly_frame(_perturb(_hourly_bars()), PARAMS)
    pd.testing.assert_frame_equal(a.iloc[: K + 1], b.iloc[: K + 1])


def test_donchian_excludes_current_bar():
    """A breakout must be vs PRIOR bars: current high must not enter the channel."""
    bars = _hourly_bars(60)
    dc = hfeat.donchian(bars, 20)
    i = 50
    prior_high = bars["high"].iloc[i - 20: i].max()
    assert dc["donchian_high"].iloc[i] == pytest.approx(prior_high)
    # even if the current bar spikes, its own channel value is unchanged
    spiked = bars.copy()
    spiked.loc[i, "high"] *= 10
    dc2 = hfeat.donchian(spiked, 20)
    assert dc2["donchian_high"].iloc[i] == dc["donchian_high"].iloc[i]


def test_daily_features_ignore_future():
    n = 300
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.015, n)), index=idx)
    bench = pd.Series(200 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)

    a = dfeat.build_daily_frame(close, bench, bench)
    close2, bench2 = close.copy(), bench.copy()
    close2.iloc[K + 1:] *= 2.5
    bench2.iloc[K + 1:] *= 0.4
    b = dfeat.build_daily_frame(close2, bench2, bench2)
    pd.testing.assert_frame_equal(a.iloc[: K + 1], b.iloc[: K + 1])


def test_rs_rank_and_breadth_ignore_future():
    n = 260
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    closes = {s: pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, n)), index=idx)
              for s in ["A", "B", "C"]}
    bench = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)

    r1 = dfeat.rs_rank(closes, bench, 63)
    b1 = dfeat.breadth(closes, 50)
    closes2 = {s: c.copy() for s, c in closes.items()}
    for c in closes2.values():
        c.iloc[K + 1:] *= rng.uniform(0.2, 4.0)
    r2 = dfeat.rs_rank(closes2, bench, 63)
    b2 = dfeat.breadth(closes2, 50)
    pd.testing.assert_frame_equal(r1.iloc[: K + 1], r2.iloc[: K + 1])
    pd.testing.assert_series_equal(b1.iloc[: K + 1], b2.iloc[: K + 1])
