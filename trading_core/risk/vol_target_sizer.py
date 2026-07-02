"""Volatility-target position sizing (SPEC §6.1).

    size(shares) = (equity * risk_per_trade) / (k * ATR)

so that a k*ATR adverse move loses ~risk_per_trade of equity. Shorts are
capped at a fraction of the equivalent long size (SPEC §6.4).
"""

from __future__ import annotations

from data.config_loader import load_config


def position_size(equity: float, atr_value: float, params: dict,
                  side: int = 1, confidence_multiplier: float = 1.0,
                  price: float | None = None) -> float:
    """Return share quantity (>=0). ``params`` provides risk_per_trade_pct
    and chandelier_k (GA-searched)."""
    if atr_value is None or atr_value <= 0 or equity <= 0:
        return 0.0
    risk_frac = float(params["risk_per_trade_pct"]) / 100.0
    k = float(params["chandelier_k"])
    qty = (equity * risk_frac * confidence_multiplier) / (k * atr_value)
    if side < 0:
        cap = float(load_config("params")["fixed"]["short_size_cap_vs_long"])
        qty *= cap
    if price is not None and price > 0:
        # never exceed 25% of equity notional in one name (sanity cap)
        qty = min(qty, 0.25 * equity / price)
    return max(qty, 0.0)
