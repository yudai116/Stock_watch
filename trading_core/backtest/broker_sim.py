"""Broker simulator: positions, fills, financing accrual, equity accounting.

Used by backtest/engine.py. All fills happen at prices provided by the
engine (next bar open) transformed by the cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backtest.costs import CostModel


@dataclass
class Order:
    symbol: str
    side: int              # +1 buy, -1 sell (closing a long = sell)
    qty: float
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: int              # +1 long, -1 short
    qty: float
    entry_price: float
    entry_ts: pd.Timestamp
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str
    side: int
    qty: float
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    pnl: float             # net of commissions/spread/slippage on both legs
    reason: str = ""


class BrokerSim:
    def __init__(self, initial_cash: float, cost_model: CostModel):
        self.cash = float(initial_cash)
        self.costs = cost_model
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.total_commission = 0.0
        self.total_financing = 0.0

    # -------------------------------------------------------------- fills

    def execute(self, order: Order, ref_price: float, ts: pd.Timestamp) -> None:
        """Fill an order at cost-adjusted price. Supports open/close/flip=no."""
        pos = self.positions.get(order.symbol)
        if pos is None and order.meta.get("close"):
            return  # stale close order: position already exited

        px = self.costs.fill_price(ref_price, order.side)
        notional = px * order.qty
        fee = self.costs.commission(notional)
        self.total_commission += fee
        self.cash -= fee

        if pos is None:
            # opening a new position (long if buy, short if sell)
            self.cash -= order.side * notional
            self.positions[order.symbol] = Position(
                symbol=order.symbol, side=order.side, qty=order.qty,
                entry_price=px, entry_ts=ts, meta=dict(order.meta),
            )
            # remember entry commission for trade P&L
            self.positions[order.symbol].meta["_entry_fee"] = fee
            return

        if pos.side == order.side:
            raise ValueError(f"pyramiding not supported: {order.symbol}")

        # closing (full or partial)
        close_qty = min(order.qty, pos.qty)
        self.cash -= order.side * px * close_qty
        gross = (px - pos.entry_price) * pos.side * close_qty
        entry_fee_part = pos.meta.get("_entry_fee", 0.0) * (close_qty / pos.qty)
        pnl = gross - fee - entry_fee_part
        self.trades.append(Trade(
            symbol=order.symbol, side=pos.side, qty=close_qty,
            entry_ts=pos.entry_ts, entry_price=pos.entry_price,
            exit_ts=ts, exit_price=px, pnl=pnl, reason=order.reason,
        ))
        pos.meta["_entry_fee"] = pos.meta.get("_entry_fee", 0.0) - entry_fee_part
        pos.qty -= close_qty
        if pos.qty <= 1e-12:
            del self.positions[order.symbol]

    # ---------------------------------------------------------- financing

    def accrue_financing(self, calendar_days: int, mark_prices: dict[str, float]) -> None:
        if calendar_days <= 0:
            return
        for pos in self.positions.values():
            mark = mark_prices.get(pos.symbol, pos.entry_price)
            cost = self.costs.financing_cost(mark * pos.qty, pos.side, calendar_days,
                                             borrow_annual=pos.meta.get("borrow_annual"))
            self.cash -= cost
            self.total_financing += cost

    # ------------------------------------------------------------- equity

    def equity(self, mark_prices: dict[str, float]) -> float:
        eq = self.cash
        for pos in self.positions.values():
            mark = mark_prices.get(pos.symbol, pos.entry_price)
            eq += pos.side * pos.qty * mark
        return eq
