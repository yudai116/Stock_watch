"""Index hedge (SPEC_ADDENDUM_v2 R1): sizing, ratio clamp, churn threshold,
and PIT-safe rolling betas."""

import numpy as np
import pandas as pd
import pytest

from signals.index_hedge import (hedge_adjustment_qty, rolling_beta,
                                 target_hedge_notional)


def test_target_notional_hand_computed():
    # book: 60k of beta 1.2 + 40k of beta 0.8 -> beta exposure = 104k
    exposures = {"NVDA": 60_000.0, "MSFT": 40_000.0}
    betas = {"NVDA": 1.2, "MSFT": 0.8}
    target = target_hedge_notional(exposures, betas, hedge_on=True, ratio=0.5)
    assert target == pytest.approx(0.5 * (60_000 * 1.2 + 40_000 * 0.8))


def test_ratio_clamped_to_30_70_band():
    exposures = {"NVDA": 100_000.0}
    betas = {"NVDA": 1.0}
    lo = target_hedge_notional(exposures, betas, True, ratio=0.05)
    hi = target_hedge_notional(exposures, betas, True, ratio=0.95)
    assert lo == pytest.approx(30_000.0)     # min_ratio 0.3
    assert hi == pytest.approx(70_000.0)     # max_ratio 0.7


def test_hedge_off_or_empty_book():
    assert target_hedge_notional({"A": 1e5}, {"A": 1.0}, hedge_on=False) == 0.0
    assert target_hedge_notional({}, {}, hedge_on=True) == 0.0


def test_missing_beta_defaults_to_one():
    target = target_hedge_notional({"XYZ": 50_000.0}, {}, True, ratio=0.5)
    assert target == pytest.approx(25_000.0)


def test_adjustment_ignores_small_drift():
    # current short 100, target 105 qty (5% drift < 10% threshold) -> no trade
    assert hedge_adjustment_qty(100.0, 105.0 * 400.0, 400.0) == 0.0
    # large increase -> negative qty (sell to add short)
    delta = hedge_adjustment_qty(100.0, 200.0 * 400.0, 400.0)
    assert delta == pytest.approx(-100.0)
    # reduction -> positive qty (buy back)
    delta = hedge_adjustment_qty(200.0, 100.0 * 400.0, 400.0)
    assert delta == pytest.approx(100.0)


def test_rolling_beta_recovers_true_beta_and_is_pit():
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    idx_ret = rng.normal(0.0005, 0.01, n)
    stock_ret = 1.5 * idx_ret + rng.normal(0, 0.002, n)   # true beta 1.5
    index_close = pd.Series(100 * np.cumprod(1 + idx_ret), index=idx)
    stock_close = pd.Series(50 * np.cumprod(1 + stock_ret), index=idx)

    beta = rolling_beta(stock_close, index_close, window=60)
    assert beta.iloc[-1] == pytest.approx(1.5, abs=0.2)

    # future perturbation: betas up to K unchanged
    K = 300
    s2, i2 = stock_close.copy(), index_close.copy()
    s2.iloc[K + 1:] *= 0.2
    i2.iloc[K + 1:] *= 3.0
    beta2 = rolling_beta(s2, i2, window=60)
    pd.testing.assert_series_equal(beta.iloc[: K + 1], beta2.iloc[: K + 1])
