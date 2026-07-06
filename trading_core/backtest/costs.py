"""Execution cost model, v2 (SPEC_ADDENDUM_v2 R1).

Two instrument profiles routed by ``instrument``:

  * ``cash``  — Saxo cash equity: commission (0.08%, min $1) + spread +
                slippage. NO overnight financing. All stock longs.
  * ``cfd``   — Saxo index CFD (QQQ/SMH hedge): spread + slippage +
                financing. Long CFD pays benchmark+markup daily; short CFD
                is conservatively booked at ZERO financing (no credit).

All parameters come from config/costs.yaml. Sign convention: returned costs
are POSITIVE = charge to the trader.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.config_loader import load_config

CASH = "cash"
CFD = "cfd"


@dataclass(frozen=True)
class InstrumentCosts:
    commission_pct: float
    commission_min: float
    half_spread_pct: float
    slippage_pct: float
    financing_enabled: bool = False
    benchmark_rate: float = 0.0
    long_markup: float = 0.0
    short_rate: float = 0.0          # conservative: 0 credit for hedge shorts
    day_count: int = 360

    def fill_price(self, ref_price: float, side: int) -> float:
        """Executable price given reference (next bar open). side=+1 buy."""
        impact = self.half_spread_pct + self.slippage_pct
        return ref_price * (1.0 + side * impact)

    def commission(self, notional: float) -> float:
        if self.commission_pct == 0.0 and self.commission_min == 0.0:
            return 0.0
        return max(self.commission_min, abs(notional) * self.commission_pct)

    def financing_cost(self, notional: float, position_side: int,
                       calendar_days: int) -> float:
        """Overnight financing for ``calendar_days`` (weekend=3 on Fri)."""
        if not self.financing_enabled or calendar_days <= 0:
            return 0.0
        rate = (self.benchmark_rate + self.long_markup) if position_side > 0 \
            else self.short_rate
        return abs(notional) * rate * calendar_days / self.day_count


@dataclass(frozen=True)
class CostModel:
    """Routes cost math to the right instrument profile."""

    profiles: dict  # instrument name -> InstrumentCosts

    @classmethod
    def from_config(cls, slippage: str = "default") -> "CostModel":
        c = load_config("costs")
        eq = c["cash_equity"]
        cfd = c["index_cfd"]
        return cls(profiles={
            CASH: InstrumentCosts(
                commission_pct=eq["commission"]["pct_of_notional"],
                commission_min=eq["commission"]["min_usd"],
                half_spread_pct=eq["spread"]["half_spread_pct"],
                slippage_pct=eq["slippage"][f"{slippage}_pct"],
                financing_enabled=bool(eq["financing"]["enabled"]),
            ),
            CFD: InstrumentCosts(
                commission_pct=cfd["commission"]["pct_of_notional"],
                commission_min=cfd["commission"]["min_usd"],
                half_spread_pct=cfd["spread"]["half_spread_pct"],
                slippage_pct=cfd["slippage"][f"{slippage}_pct"],
                financing_enabled=bool(cfd["financing"]["enabled"]),
                benchmark_rate=cfd["financing"]["benchmark_rate_annual"],
                long_markup=cfd["financing"]["long_markup_annual"],
                short_rate=cfd["financing"]["short_rate_annual"],
                day_count=cfd["financing"]["day_count"],
            ),
        })

    def profile(self, instrument: str) -> InstrumentCosts:
        try:
            return self.profiles[instrument]
        except KeyError:
            raise ValueError(f"unknown instrument {instrument!r}; "
                             f"known: {sorted(self.profiles)}") from None

    def fill_price(self, ref_price: float, side: int, instrument: str = CASH) -> float:
        return self.profile(instrument).fill_price(ref_price, side)

    def commission(self, notional: float, instrument: str = CASH) -> float:
        return self.profile(instrument).commission(notional)

    def financing_cost(self, notional: float, position_side: int,
                       calendar_days: int, instrument: str = CASH) -> float:
        return self.profile(instrument).financing_cost(notional, position_side,
                                                       calendar_days)
