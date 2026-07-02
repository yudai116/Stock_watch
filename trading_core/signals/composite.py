"""Composite swing strategy: wires regime gate, entry signals, hard gates,
sizing and the common exit framework (SPEC §5.3) into backtest/engine.py.

Timing: decisions at hourly bar close; fills next bar open (engine contract).
Stops are evaluated on bar CLOSES (no intrabar fills) — a conservative v1
simplification documented for the reviewer.

Daily inputs (regime label, RS rank) are looked up as "last value with
timestamp <= now", where their timestamps are DAY CLOSE times — so an
hourly decision never sees the current day's still-forming daily value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtest.broker_sim import Order
from backtest.engine import MarketSnapshot
from data.config_loader import load_config
from risk import dd_control, portfolio_limits, vol_target_sizer
from signals import mean_revert, short_breakdown, trend_follow
from signals.gates import NO_GATE, GateState
from signals.regime_gate import gate as regime_gate


@dataclass
class SwingStrategy:
    features: dict[str, pd.DataFrame]          # hourly features, row-aligned with engine bars
    regime: pd.Series                          # daily, index = day CLOSE ts (UTC)
    rs_rank: pd.DataFrame                      # daily, index = day CLOSE ts, cols = symbols
    params: dict                               # GA individual + fixed merged
    sector_by_symbol: dict[str, str] = field(default_factory=dict)
    universe_provider: Callable[[pd.Timestamp], set] | None = None
    gate_provider: Callable[[str, pd.Timestamp], GateState] | None = None
    confidence_provider: Callable[[str, int, pd.Timestamp], float] | None = None

    def __post_init__(self):
        self._peak_equity = 0.0
        self._fixed = load_config("params")["fixed"]

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

    def _gate(self, sym: str, now: pd.Timestamp) -> GateState:
        return self.gate_provider(sym, now) if self.gate_provider else NO_GATE

    def _confidence(self, sym: str, side: int, now: pd.Timestamp) -> float:
        return self.confidence_provider(sym, side, now) if self.confidence_provider else 1.0

    # --------------------------------------------------------------- main

    def on_bar_close(self, snap: MarketSnapshot) -> list[Order]:
        now = snap.ts
        p = self.params
        orders: list[Order] = []
        equity = snap.equity()
        self._peak_equity = max(self._peak_equity, equity)
        regime_label = self._regime_now(now)
        rg = regime_gate(regime_label, bool(p.get("mr_enabled", True)))

        # ---------- manage open positions (exits first)
        open_risk_pct = 0.0
        for sym, pos in list(snap.broker.positions.items()):
            row = self._row(snap, sym)
            if row is None or pd.isna(row["atr"]):
                continue
            m = pos.meta
            close = float(row["close"])
            k = float(p["chandelier_k"]) * self._gate(sym, now).tighten_stop_factor

            if pos.side > 0:
                m["extreme"] = max(m.get("extreme", close), close)
                trail = m["extreme"] - k * float(row["atr"])
                stop = max(m["stop0"], trail)
                hit = close <= stop
            else:
                m["extreme"] = min(m.get("extreme", close), close)
                trail = m["extreme"] + k * float(row["atr"])
                stop = min(m["stop0"], trail)
                hit = close >= stop
            m["stop"] = stop

            # time stop: N business days without progress (< 0.5R unrealized)
            held_bdays = int(np.busday_count(pos.entry_ts.date(), now.date()))
            limit = float(p["time_stop_days"])
            if pos.side < 0:
                limit *= float(self._fixed["short_time_stop_factor"])
            r_unit = m.get("r_unit", 1e-9)
            unreal_r = (close - pos.entry_price) * pos.side / r_unit
            stalled = held_bdays >= limit and unreal_r < 0.5

            if hit or stalled:
                orders.append(Order(sym, -pos.side, pos.qty,
                                    reason="stop" if hit else "time_stop",
                                    meta={"close": True}))
                continue

            # partial take profit (GA choice)
            if (str(p.get("partial_take_profit", "none")) == "half_at_2R"
                    and not m.get("partial_done") and unreal_r >= 2.0):
                orders.append(Order(sym, -pos.side, pos.qty / 2,
                                    reason="partial_2R", meta={"close": True}))
                m["partial_done"] = True

            # crisis: scale down existing once
            if rg.scale_down_existing and not m.get("crisis_reduced"):
                orders.append(Order(sym, -pos.side, pos.qty / 2,
                                    reason="crisis_reduce", meta={"close": True}))
                m["crisis_reduced"] = True

            open_risk_pct += abs(close - stop) * pos.qty / max(equity, 1e-9) * 100.0

        # ---------- new entries
        dd_mult = dd_control.risk_multiplier(equity, self._peak_equity)
        if dd_mult <= 0.0 or not (rg.allow_long_tf or rg.allow_long_mr or rg.allow_short):
            return orders

        universe = self.universe_provider(now) if self.universe_provider else None
        for sym in snap.symbols():
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

            side = 0
            if rg.allow_long_tf and trend_follow.entry_ok(row, rs_pct, p):
                side = 1
            elif rg.allow_long_mr and mean_revert.entry_ok(row, p):
                side = 1
            elif rg.allow_short and not gate_state.blocked_short \
                    and short_breakdown.entry_ok(row, rs_pct, p):
                side = -1
            if side == 0:
                continue

            conf = self._confidence(sym, side, now)
            atr_v = float(row["atr"])
            close = float(row["close"])
            qty = vol_target_sizer.position_size(
                equity, atr_v, p, side=side,
                confidence_multiplier=conf * dd_mult, price=close)
            if qty <= 0:
                continue
            risk_pct = float(p["risk_per_trade_pct"]) * conf * dd_mult
            ok, _why = portfolio_limits.can_open(
                snap.broker.positions, self.sector_by_symbol, sym, risk_pct, open_risk_pct)
            if not ok:
                continue
            k = float(p["chandelier_k"])
            stop0 = close - side * k * atr_v
            orders.append(Order(sym, side, qty, reason="entry", meta={
                "stop0": stop0, "stop": stop0, "extreme": close,
                "r_unit": k * atr_v, "entry_atr": atr_v,
            }))
            open_risk_pct += risk_pct
        return orders
