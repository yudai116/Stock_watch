"""Bear-regime index hedge (SPEC_ADDENDUM_v2 R1) — replaces short_breakdown.

Instead of shorting single stocks (structurally unfavourable: upward drift,
squeezes, variable borrow), the system shorts an index CFD (QQQ or SMH) to
offset 30-70% of the portfolio's long beta while the regime is bear/crisis.

    hedge notional = hedge_ratio * sum_i(beta_i * long_notional_i)

Betas are estimated from trailing daily returns vs the hedge instrument
(row t uses data <= t only); missing history falls back to beta = 1.0
(conservative for large-cap tech vs QQQ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.config_loader import load_config


def rolling_beta(stock_close: pd.Series, index_close: pd.Series,
                 window: int = 60) -> pd.Series:
    """Trailing OLS beta per day; row t uses returns up to t only."""
    rs = stock_close.pct_change()
    ri = index_close.reindex(stock_close.index).ffill().pct_change()
    cov = rs.rolling(window).cov(ri)
    var = ri.rolling(window).var()
    return cov / var.replace(0, np.nan)


def target_hedge_notional(long_exposures: dict[str, float],
                          betas: dict[str, float],
                          hedge_on: bool,
                          ratio: float | None = None) -> float:
    """Desired SHORT notional of the hedge instrument (>= 0)."""
    cfg = load_config("params")["hedge"]
    if not hedge_on or not long_exposures:
        return 0.0
    r = float(cfg["ratio"]) if ratio is None else float(ratio)
    r = min(max(r, float(cfg["min_ratio"])), float(cfg["max_ratio"]))
    beta_exposure = sum(notional * betas.get(sym, 1.0)
                        for sym, notional in long_exposures.items())
    return max(0.0, r * beta_exposure)


def hedge_adjustment_qty(current_short_qty: float, target_notional: float,
                         index_price: float) -> float:
    """Signed qty change for the hedge position: negative = sell (increase
    short), positive = buy (reduce short). Small drifts are ignored to avoid
    churn (rebalance_threshold of target)."""
    cfg = load_config("params")["hedge"]
    if index_price <= 0:
        return 0.0
    target_qty = target_notional / index_price
    delta = target_qty - current_short_qty
    threshold = float(cfg["rebalance_threshold"]) * max(target_qty, current_short_qty, 1e-9)
    if abs(delta) < threshold:
        return 0.0
    return -delta  # more short needed -> negative (sell)
