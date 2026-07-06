"""Composite swing strategy, v2 (SPEC_ADDENDUM_v2 R1/R2) — Phase 3b.

Daily decision cycle (R2):
  * decisions at DAILY bar close; market orders fill next day's open
  * protective stops are RESTING stop orders monitored intrabar by the
    engine (R2b) — chandelier trail / initial stop can fire during the day
  * stock positions are LONG-ONLY cash equity (no financing)
  * bear/crisis regime: no new longs + index-CFD hedge (QQQ/SMH) sized to
    offset hedge.ratio of the book's long beta (R1)

Daily inputs (regime label, RS rank, betas) are looked up as "last value
with timestamp <= now"; their timestamps are DAY-CLOSE times, so a decision
never sees a still-forming value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtest.broker_sim import Order
from backtest.costs import CASH, CFD
from backtest.engine import MarketSnapshot
from data.config_loader import load_config
from risk import dd_control, portfolio_limits, vol_target_sizer
from signals import index_hedge, mean_revert, trend_follow
from signals.gates import NO_GATE, GateState
from signals.regime_gate import gate as regime_gate


@dataclass
class SwingStrategy:
    features: dict[str, pd.DataFrame]          # daily signal frames, row-aligned with bars
    regime: pd.Series                          # daily, index = day CLOSE ts (UTC)
    rs_rank: pd.DataFrame                      # daily, index = day CLOSE ts, cols = symbols
    params: dict                               # grid individual + fixed merged
    sector_by_symbol: dict[str, str] = field(default_factory=dict)
    betas: dict[str, pd.Series] = field(default_factory=dict)  # daily rolling beta
    universe_provider: Callable[[pd.Timestamp], set] | None = None
    gate_provider: Callable[[str, pd.Timestamp], GateState] | None = None
    confidence_provider: Callable[[str, int, pd.Timestamp], float] | None = None

    def __post_init__(self):
        self._peak_equity = 0.0
        cfg = load_config("params")
        self._fixed = cfg["fixed"]
        self._hedge_cfg = cfg["hedge"]
        self.hedge_symbol = str(self._hedge_cfg["instrument"])

    # ------------------------------------------------------------ helpers

    def _row(self, snap: MarketSnapshot, sym: str) -> pd.Series | None:
        n = len(snap.bars(sym))
        if n == 0 or sym not in self.features or n > len(self.features[sym]):
            return None
        return self.features[sym].iloc[n - 1]

    def _regime_now(self, now: pd.Timestamp) -> str:
        s = self.regime.loc[:now]
        return str(s.iloc[-1]) if len(s) else "unknown"

    def _rs_now(self, now: pd.Timestamp, sym: str) -> float:
        df = self.rs_rank.loc[:now]
        if df.empty or sym not in df.columns:
            return float("nan")
        return float(df[sym].iloc[-1])

    def _beta_now(self, now: pd.Timestamp, sym: str) -> float:
        s = self.betas.get(sym)
        if s is None:
            return 1.0
        h = s.loc[:now].dropna()
        return float(h.iloc[-1]) if len(h) else 1.0

    def _gate(self, sym: str, now: pd.Timestamp) -> GateState:
        return self.gate_provider(sym, now) if self.gate_provider else NO_GATE

    def _confidence(self, sym: str, side: int, now: pd.Timestamp) -> float:
        return self.confidence_provider(sym, side, now) if self.confidence_provider else 1.0

    # --------------------------------------------------------------- main

    def on_bar_close(self, snap: MarketSnapshot) -> list[Order]:
        now = snap.ts
        p = self.params
        orders: list[Order] = []
        marks = snap.marks()
        equity = snap.broker.equity(marks)
        self._peak_equity = max(self._peak_equity, equity)
        regime_label = self._regime_now(now)
        rg = regime_gate(regime_label, bool(p.get("mr_enabled", False)))

        # ---------- manage open stock positions (exits first)
        open_risk_pct = 0.0
        for sym, pos in list(snap.broker.positions.items()):
            if sym == self.hedge_symbol and pos.instrument == CFD:
                continue                       # hedge managed separately below
            row = self._row(snap, sym)
            if row is None or pd.isna(row["atr"]):
                continue
            m = pos.meta
            close = float(row["close"])
            k = float(p["chandelier_k"]) * self._gate(sym, now).tighten_stop_factor

            m["extreme"] = max(m.get("extreme", close), close)
            trail = m["extreme"] - k * float(row["atr"])
            stop = max(m["stop0"], trail)
            m["stop"] = stop

            # time stop: N business days without progress (< 0.5R unrealized)
            held_bdays = int(np.busday_count(pos.entry_ts.date(), now.date()))
            r_unit = m.get("r_unit", 1e-9)
            unreal_r = (close - pos.entry_price) / r_unit
            stalled = held_bdays >= float(p["time_stop_days"]) and unreal_r < 0.5

            if stalled:
                orders.append(Order(sym, -1, pos.qty, reason="time_stop",
                                    instrument=pos.instrument, meta={"close": True}))
                continue

            # refresh the resting intrabar stop (engine cancel/replace, R2b)
            orders.append(Order(sym, -1, pos.qty, reason="stop",
                                instrument=pos.instrument,
                                meta={"close": True, "order_type": "stop",
                                      "stop_price": stop}))

            # partial take profit (grid choice)
            if (str(p.get("partial_take_profit", "none")) == "half_at_2R"
                    and not m.get("partial_done") and unreal_r >= 2.0):
                orders.append(Order(sym, -1, pos.qty / 2, reason="partial_2R",
                                    instrument=pos.instrument, meta={"close": True}))
                m["partial_done"] = True

            # crisis: scale down existing once
            if rg.scale_down_existing and not m.get("crisis_reduced"):
                orders.append(Order(sym, -1, pos.qty / 2, reason="crisis_reduce",
                                    instrument=pos.instrument, meta={"close": True}))
                m["crisis_reduced"] = True

            open_risk_pct += abs(close - stop) * pos.qty / max(equity, 1e-9) * 100.0

        # ---------- index hedge management (R1)
        orders += self._manage_hedge(snap, rg.hedge_on, marks)

        # ---------- new entries
        dd_mult = dd_control.risk_multiplier(equity, self._peak_equity)
        if dd_mult <= 0.0 or not (rg.allow_long_tf or rg.allow_long_mr):
            return orders

        universe = self.universe_provider(now) if self.universe_provider else None
        for sym in snap.symbols():
            if sym == self.hedge_symbol:
                continue                       # never trade the hedge as a stock
            if universe is not None and sym not in universe:
                continue
            if sym in snap.broker.positions:
                continue
            row = self._row(snap, sym)
            if row is None or pd.isna(row["atr"]) or row["atr"] <= 0:
                continue
            gate_state = self._gate(sym, now)
            if gate_state.blocked_new:
                continue
            rs_pct = self._rs_now(now, sym)

            entry = False
            if rg.allow_long_tf and trend_follow.entry_ok(row, rs_pct, p):
                entry = True
            elif rg.allow_long_mr and mean_revert.entry_ok(row, p):
                entry = True
            if not entry:
                continue

            conf = self._confidence(sym, +1, now)
            atr_v = float(row["atr"])
            close = float(row["close"])
            qty = vol_target_sizer.position_size(
                equity, atr_v, p,
                confidence_multiplier=conf * dd_mult, price=close)
            if qty <= 0:
                continue
            risk_pct = float(p["risk_per_trade_pct"]) * conf * dd_mult
            ok, _why = portfolio_limits.can_open(
                {s: v for s, v in snap.broker.positions.items() if s != self.hedge_symbol},
                self.sector_by_symbol, sym, risk_pct, open_risk_pct)
            if not ok:
                continue
            k = float(p["chandelier_k"])
            stop0 = close - k * atr_v
            orders.append(Order(sym, +1, qty, reason="entry", instrument=CASH, meta={
                "stop0": stop0, "stop": stop0, "extreme": close,
                "r_unit": k * atr_v, "entry_atr": atr_v,
            }))
            # protective stop active from the fill (checked intrabar)
            orders.append(Order(sym, -1, qty, reason="stop", instrument=CASH,
                                meta={"close": True, "order_type": "stop",
                                      "stop_price": stop0}))
            open_risk_pct += risk_pct
        return orders

    # ---------------------------------------------------------------- hedge

    def _manage_hedge(self, snap: MarketSnapshot, hedge_on: bool,
                      marks: dict[str, float]) -> list[Order]:
        hsym = self.hedge_symbol
        pos = snap.broker.positions.get(hsym)
        cur_short_qty = pos.qty if pos is not None and pos.side < 0 else 0.0

        if hsym not in marks:
            return []                          # hedge instrument not tradable yet
        px = marks[hsym]

        if not hedge_on:
            if cur_short_qty > 0:
                return [Order(hsym, +1, cur_short_qty, reason="hedge_off",
                              instrument=CFD, meta={"close": True})]
            return []

        exposures = {s: p_.qty * marks.get(s, p_.entry_price)
                     for s, p_ in snap.broker.positions.items()
                     if p_.side > 0 and p_.instrument == CASH}
        betas = {s: self._beta_now(snap.ts, s) for s in exposures}
        target = index_hedge.target_hedge_notional(exposures, betas, hedge_on=True)
        delta = index_hedge.hedge_adjustment_qty(cur_short_qty, target, px)
        if delta == 0.0:
            return []
        if delta < 0:                          # increase short
            return [Order(hsym, -1, -delta, reason="hedge_add", instrument=CFD)]
        reduce_qty = min(delta, cur_short_qty)
        if reduce_qty <= 0:
            return []
        return [Order(hsym, +1, reduce_qty, reason="hedge_reduce",
                      instrument=CFD, meta={"close": True})]
