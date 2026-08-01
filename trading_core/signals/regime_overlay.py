"""Flow-F [D]: risk-management overlay on a core index holding.

The system stops seeking stock-picking alpha; its only job is drawdown
compression on a QQQ (or SMH) core:

    bull  -> fully invested (INVEST_FRACTION of equity, cash equity)
    range -> hold what we have, no new buying
    bear  -> liquidate to cash

Search surface: effectively ZERO strategy parameters (the regime filter's
ma_days / vix_threshold live in params.yaml simple_regime and are shared,
not tuned per-variant) — the maximum defence against overfitting, matching
the evidence base for vol-target/trend-filter DD compression.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.broker_sim import Order
from backtest.costs import CASH
from backtest.engine import MarketSnapshot

INVEST_FRACTION = 0.95


@dataclass
class RegimeOverlayStrategy:
    regime: pd.Series                  # daily labels, index = day-close ts (UTC)
    symbol: str = "QQQ"                # the core asset

    def _regime_now(self, now: pd.Timestamp) -> str:
        s = self.regime.loc[:now]
        return str(s.iloc[-1]) if len(s) else "unknown"

    def on_bar_close(self, snap: MarketSnapshot) -> list[Order]:
        label = self._regime_now(snap.ts)
        pos = snap.position(self.symbol)

        if label == "bull" and pos is None:
            marks = snap.marks()
            px = marks.get(self.symbol)
            if not px or px <= 0:
                return []
            qty = snap.broker.equity(marks) * INVEST_FRACTION / px
            if qty <= 0:
                return []
            return [Order(self.symbol, +1, qty, reason="overlay_enter",
                          instrument=CASH)]

        if label in ("bear", "crisis") and pos is not None:
            return [Order(self.symbol, -1, pos.qty, reason="overlay_exit",
                          instrument=CASH, meta={"close": True})]

        return []   # range/unknown: hold existing, no new buying
