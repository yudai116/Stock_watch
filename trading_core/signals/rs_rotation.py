"""Weekly relative-strength rotation — Phase 3a (SPEC_ADDENDUM_v2 H, flow [C]).

The evidence-richest strategy form (Dual-Momentum family), measured FIRST so
the branch decision needs only one backtest comparison:

  * every ``rebalance_days`` trading days, rank the universe by trailing
    ``lookback_days`` return (data <= decision close only)
  * regime bull  -> hold top ``n_holdings`` names, inverse-vol weighted
    (equal weight fallback), cash-equity longs
  * regime bear/range/crisis/unknown -> go to cash (conservative default;
    a beta hedge variant is a later experiment)

Exactly 3 parameters (R3-compliant): lookback_days, n_holdings, rebalance_days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtest.broker_sim import Order
from backtest.costs import CASH
from backtest.engine import MarketSnapshot

INVEST_FRACTION = 0.95      # keep a cash buffer for costs/slippage
VOL_DAYS = 20               # inverse-vol weighting window


@dataclass
class RotationStrategy:
    regime: pd.Series                          # daily labels, index = day-close ts
    params: dict                               # lookback_days, n_holdings, rebalance_days
    universe_provider: Callable[[pd.Timestamp], set] | None = None
    benchmark_symbols: tuple = ()              # e.g. ("QQQ",): never traded

    def __post_init__(self):
        self._days_since_rebalance = None      # None = never rebalanced

    # ------------------------------------------------------------ helpers

    def _regime_now(self, now: pd.Timestamp) -> str:
        s = self.regime.loc[:now]
        return str(s.iloc[-1]) if len(s) else "unknown"

    def _momentum_and_vol(self, snap: MarketSnapshot, sym: str,
                          lookback: int) -> tuple[float, float] | None:
        bars = snap.bars(sym)
        if len(bars) < lookback + 1:
            return None
        c = bars["close"]
        mom = float(c.iloc[-1] / c.iloc[-1 - lookback] - 1.0)
        ret = c.iloc[-VOL_DAYS - 1:].pct_change().dropna()
        vol = float(ret.std())
        return mom, vol

    # --------------------------------------------------------------- main

    def on_bar_close(self, snap: MarketSnapshot) -> list[Order]:
        p = self.params
        if self._days_since_rebalance is not None:
            self._days_since_rebalance += 1
            if self._days_since_rebalance < int(p["rebalance_days"]):
                return []
        # rebalance day (or first evaluable bar)
        now = snap.ts
        lookback = int(p["lookback_days"])
        n_hold = int(p["n_holdings"])
        regime_label = self._regime_now(now)

        universe = self.universe_provider(now) if self.universe_provider else None
        candidates = [s for s in snap.symbols()
                      if s not in self.benchmark_symbols
                      and (universe is None or s in universe)]

        # target book
        targets: dict[str, float] = {}
        if regime_label == "bull":
            scored = []
            for sym in candidates:
                mv = self._momentum_and_vol(snap, sym, lookback)
                if mv is None:
                    continue
                scored.append((sym, *mv))
            if not scored:
                return []                      # not enough history yet: wait
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:n_hold]
            inv_vol = {s: (1.0 / v if v and v > 0 else 0.0) for s, _, v in top}
            total = sum(inv_vol.values())
            if total <= 0:
                weights = {s: 1.0 / len(top) for s, _, _ in top}
            else:
                weights = {s: iv / total for s, iv in inv_vol.items()}
            marks = snap.marks()
            equity = snap.broker.equity(marks)
            budget = equity * INVEST_FRACTION
            for sym, w in weights.items():
                px = marks.get(sym)
                if px and px > 0:
                    targets[sym] = budget * w / px      # target qty
        # non-bull regimes: targets stays empty -> full cash

        self._days_since_rebalance = 0
        return self._orders_towards(snap, targets)

    def _orders_towards(self, snap: MarketSnapshot,
                        targets: dict[str, float]) -> list[Order]:
        orders: list[Order] = []
        held = {s: pos for s, pos in snap.broker.positions.items()}
        for sym, pos in held.items():
            if sym not in targets:
                orders.append(Order(sym, -1, pos.qty, reason="rotate_out",
                                    instrument=CASH, meta={"close": True}))
        for sym, qty in targets.items():
            if qty <= 0:
                continue
            pos = held.get(sym)
            if pos is None:
                orders.append(Order(sym, +1, qty, reason="rotate_in",
                                    instrument=CASH))
            # held names are kept as-is between rebalances (no resizing churn)
        return orders
