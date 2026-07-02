"""Saxo CFD cost model: commission + spread + slippage + daily financing.

All parameters come from config/costs.yaml (no hardcoding). Sign convention:
returned costs are POSITIVE = charge to the trader.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.config_loader import load_config


@dataclass(frozen=True)
class CostModel:
    commission_pct: float
    commission_min: float
    half_spread_pct: float
    slippage_pct: float
    benchmark_rate: float
    long_markup: float
    short_markdown: float
    borrow_default: float
    day_count: int

    @classmethod
    def from_config(cls, slippage: str = "default") -> "CostModel":
        c = load_config("costs")
        return cls(
            commission_pct=c["commission"]["pct_of_notional"],
            commission_min=c["commission"]["min_usd"],
            half_spread_pct=c["spread"]["half_spread_pct"],
            slippage_pct=c["slippage"][f"{slippage}_pct"],
            benchmark_rate=c["financing"]["benchmark_rate_annual"],
            long_markup=c["financing"]["long_markup_annual"],
            short_markdown=c["financing"]["short_markdown_annual"],
            borrow_default=c["financing"]["borrow_cost_annual_default"],
            day_count=c["financing"]["day_count"],
        )

    # ------------------------------------------------------------- execution

    def fill_price(self, ref_price: float, side: int) -> float:
        """Executable price given reference (next bar open). side=+1 buy, -1 sell."""
        impact = self.half_spread_pct + self.slippage_pct
        return ref_price * (1.0 + side * impact)

    def commission(self, notional: float) -> float:
        return max(self.commission_min, abs(notional) * self.commission_pct)

    # ------------------------------------------------------------- financing

    def financing_cost(self, notional: float, position_side: int, calendar_days: int,
                       borrow_annual: float | None = None) -> float:
        """Overnight CFD financing for ``calendar_days`` (weekend=3 on Fri).

        Long pays benchmark + markup. Short receives benchmark - markdown but
        pays borrow; net credit is returned as a negative cost.
        """
        if calendar_days <= 0:
            return 0.0
        notional = abs(notional)
        if position_side > 0:
            rate = self.benchmark_rate + self.long_markup
        else:
            borrow = self.borrow_default if borrow_annual is None else borrow_annual
            rate = borrow + self.short_markdown - self.benchmark_rate
        return notional * rate * calendar_days / self.day_count
