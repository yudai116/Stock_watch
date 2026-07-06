"""Volatility-target position sizing (SPEC §6.1).

    size(shares) = (equity * risk_per_trade) / (k * ATR)

so that a k*ATR adverse move loses ~risk_per_trade of equity.

v2 (SPEC_ADDENDUM_v2 R1): stock positions are LONG-ONLY cash equity; the
bear-regime index hedge is sized separately by notional (signals/index_hedge),
so the old short-size cap no longer applies here.
"""

from __future__ import annotations


def position_size(equity: float, atr_value: float, params: dict,
                  confidence_multiplier: float = 1.0,
                  price: float | None = None) -> float:
    """Return share quantity (>=0). ``params`` provides risk_per_trade_pct
    and chandelier_k (grid-searched)."""
    if atr_value is None or atr_value <= 0 or equity <= 0:
        return 0.0
    risk_frac = float(params["risk_per_trade_pct"]) / 100.0
    k = float(params["chandelier_k"])
    qty = (equity * risk_frac * confidence_multiplier) / (k * atr_value)
    if price is not None and price > 0:
        # never exceed 25% of equity notional in one name (sanity cap)
        qty = min(qty, 0.25 * equity / price)
    return max(qty, 0.0)
