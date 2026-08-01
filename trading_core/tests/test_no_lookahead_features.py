"""Mandatory no-lookahead tests for the features layer (CLAUDE.md rule 2).

Method: future perturbation. Compute features on a series, then modify all
bars AFTER index k and recompute — rows <= k must be bit-identical.
v2: production signal features are DAILY (SPEC_ADDENDUM_v2 R2).
"""

import numpy as np
import pandas as pd
import pytest

from features import daily_features as dfeat
from features import hourly_features as hfeat
from features.simple_regime import simple_regime

K = 220
PARAMS = {"atr_period": 14, "donchian_entry_period": 20, "mr_rsi_period": 14}


def _daily_bars(n=320, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    ts = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * 1.012, "low": close * 0.988,
        "close": close, "volume": rng.integers(1000, 5000, n).astype(float),
    })


def _perturb(bars):
    out = bars.copy()
    out.loc[K + 1:, ["open", "high", "low", "close"]] *= 3.7
    out.loc[K + 1:, "volume"] *= 11
    return out


def test_daily_signal_frame_ignores_future():
    a = dfeat.build_signal_frame(_daily_bars(), PARAMS)
    b = dfeat.build_signal_frame(_perturb(_daily_bars()), PARAMS)
    pd.testing.assert_frame_equal(a.iloc[: K + 1], b.iloc[: K + 1])


def test_donchian_excludes_current_bar():
    """A breakout must be vs PRIOR bars: current high must not enter the channel."""
    bars = _daily_bars(60)
    dc = hfeat.donchian(bars, 20)
    i = 50
    prior_high = bars["high"].iloc[i - 20: i].max()
    assert dc["donchian_high"].iloc[i] == pytest.approx(prior_high)
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


# ---------------------------------------------- simple regime filter (R4)

def _qqq_and_vix(n=520, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    close = pd.Series(300 * np.cumprod(1 + rng.normal(0.0006, 0.012, n)), index=idx)
    vix = pd.Series(np.clip(rng.normal(18, 5, n), 10, 60), index=idx)
    return close, vix


def test_simple_regime_rule():
    """Definition check. Post-review M2: the VIX used for day t's decision is
    the PREVIOUS session's value (live parity with FRED's publication lag)."""
    close, vix = _qqq_and_vix()
    labels = simple_regime(close, vix)
    ma = close.rolling(200).mean()
    vix_lagged = vix.shift(1)
    for ts in labels.index[::37]:
        if pd.isna(vix_lagged[ts]):
            continue
        if close[ts] > ma[ts] and vix_lagged[ts] < 25.0:
            assert labels[ts] == "bull"
        elif close[ts] < ma[ts]:
            assert labels[ts] == "bear"


def test_simple_regime_uses_previous_session_vix():
    """M2 regression: a VIX spike on day t must NOT affect day t's label —
    only day t+1's (matching what a live decision can actually see)."""
    n = 260
    idx = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    close = pd.Series(np.linspace(100, 140, n), index=idx)   # firmly above MA
    vix = pd.Series(15.0, index=idx)
    spike_day = idx[230]
    vix[spike_day] = 60.0                                     # one-day spike
    labels = simple_regime(close, vix)
    assert labels[spike_day] == "bull"                        # not visible yet
    assert labels[idx[231]] == "range"                        # visible next day
    assert labels[idx[232]] == "bull"                         # spike passed


def test_simple_regime_ignores_future():
    close, vix = _qqq_and_vix()
    a = simple_regime(close, vix)
    close2, vix2 = close.copy(), vix.copy()
    close2.iloc[400:] *= 0.3
    vix2.iloc[400:] = 55.0
    b = simple_regime(close2, vix2)
    cut = close.index[399]
    pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])


def test_simple_regime_vol_proxy_fallback():
    close, _ = _qqq_and_vix()
    labels = simple_regime(close, vix=None)      # realized-vol proxy path
    assert set(labels.unique()) <= {"bull", "bear", "range"}
    assert len(labels) > 0


def test_simple_regime_handles_mismatched_vix_stamps():
    """PRODUCTION regression: store VIX is stamped at 21:00 UTC while the QQQ
    close index sits at next-day midnight. A naive reindex().ffill() yields
    all-NaN vol -> 'bull' never fires -> silent zero-trade backtests."""
    close, vix = _qqq_and_vix()
    # move VIX onto its production stamps: previous day 21:00 UTC
    vix_shifted = pd.Series(vix.values, index=vix.index - pd.Timedelta(hours=3))
    labels = simple_regime(close, vix_shifted)
    labels_exact = simple_regime(close, vix)
    assert (labels == "bull").sum() > 0                  # bulls exist
    # asof alignment must give (nearly) the same answer as exact stamps
    common = labels.index.intersection(labels_exact.index)
    agree = (labels.loc[common] == labels_exact.loc[common]).mean()
    assert agree > 0.95
