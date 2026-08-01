"""Event-driven backtest engine (SPEC §7, SPEC_ADDENDUM_v2 R2).

Timing contract (the no-lookahead core):
  * strategy decides at bar CLOSE t, seeing bars with close <= t only
  * market orders fill at the NEXT bar's OPEN, at cost-adjusted prices
  * resting STOP orders (meta order_type="stop") are monitored intrabar:
    a long-exit stop triggers when the bar's low touches the stop level and
    fills at the stop, or at the open if the bar gapped through it (R2b:
    intraday stop monitoring)
  * financing accrues per calendar day boundary crossed while holding
    (instrument-aware: cash longs accrue nothing)

The engine is timeframe-agnostic: production signals run on DAILY bars (R2);
the same engine serves the hand-verified acceptance tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from backtest.broker_sim import BrokerSim, Order, Trade
from backtest.costs import CostModel


class MarketSnapshot:
    """What a strategy is allowed to see at decision time (bar close ``ts``).

    ``bars(symbol)`` returns history up to and including the bar closing at
    ``ts`` — a positional slice of pre-validated data, so per-bar filtering
    cost is O(1). The engine guarantees the slice never extends past ts.
    """

    def __init__(self, panel: dict[str, pd.DataFrame], upto: dict[str, int],
                 ts: pd.Timestamp, broker: BrokerSim):
        self._panel = panel
        self._upto = upto
        self.ts = ts
        self.broker = broker

    def bars(self, symbol: str) -> pd.DataFrame:
        i = self._upto.get(symbol, 0)
        return self._panel[symbol].iloc[:i]

    def symbols(self) -> list[str]:
        return [s for s, i in self._upto.items() if i > 0]

    def position(self, symbol: str):
        return self.broker.positions.get(symbol)

    def marks(self) -> dict[str, float]:
        return {s: float(self._panel[s]["close"].iloc[i - 1])
                for s, i in self._upto.items() if i > 0}

    def equity(self) -> float:
        return self.broker.equity(self.marks())


class Strategy(Protocol):
    def on_bar_close(self, snap: MarketSnapshot) -> list[Order]: ...


@dataclass
class BacktestResult:
    equity_curve: pd.Series           # indexed by bar close ts
    trades: list[Trade]
    total_commission: float
    total_financing: float
    final_equity: float
    initial_equity: float

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()


def run_backtest(
    prices: dict[str, pd.DataFrame],
    strategy: Strategy,
    initial_cash: float,
    cost_model: CostModel | None = None,
    bar_duration: pd.Timedelta = pd.Timedelta(days=1),
    close_at_end: bool = False,
) -> BacktestResult:
    """``prices``: symbol -> bars with columns ts/open/high/low/close/volume,
    ts = bar OPEN time, ascending, tz-aware."""
    costs = cost_model or CostModel.from_config()
    broker = BrokerSim(initial_cash, costs)

    panel: dict[str, pd.DataFrame] = {}
    open_ts: dict[str, pd.DatetimeIndex] = {}
    for sym, df in prices.items():
        d = df.reset_index(drop=True)
        if d["ts"].dt.tz is None:
            raise ValueError(f"{sym}: naive timestamps")
        panel[sym] = d
        open_ts[sym] = pd.DatetimeIndex(d["ts"])

    all_opens = sorted(set().union(*[set(ix) for ix in open_ts.values()]))
    if not all_opens:
        raise ValueError("no bars")

    pending: list[Order] = []
    stop_book: dict[str, Order] = {}   # symbol -> resting stop (close) order
    equity_pts: list[tuple[pd.Timestamp, float]] = []
    upto: dict[str, int] = {s: 0 for s in panel}       # bars visible per symbol
    pos_in_ts: dict[str, int] = {s: 0 for s in panel}  # cursor into open_ts
    prev_date = None

    for t_open in all_opens:
        # ---- financing on calendar day change (positions held overnight)
        cur_date = t_open.date()
        if prev_date is not None and cur_date != prev_date:
            days = (cur_date - prev_date).days
            marks = {s: panel[s]["close"].iloc[upto[s] - 1] for s in panel if upto[s] > 0}
            broker.accrue_financing(days, marks)
        prev_date = cur_date

        # ---- fill pending market orders at this bar's open
        if pending:
            still: list[Order] = []
            for od in pending:
                ix = pos_in_ts[od.symbol]
                sym_ts = open_ts[od.symbol]
                if ix < len(sym_ts) and sym_ts[ix] == t_open:
                    ref = panel[od.symbol]["open"].iloc[ix]
                    broker.execute(od, float(ref), t_open)
                else:
                    still.append(od)  # symbol has no bar now (halt); retry next bar
            pending = still

        # ---- intrabar stop monitoring (R2b)
        for sym in list(stop_book):
            pos = broker.positions.get(sym)
            if pos is None:
                del stop_book[sym]          # position gone: stale stop
                continue
            ix = pos_in_ts[sym]
            if ix >= len(open_ts[sym]) or open_ts[sym][ix] != t_open:
                continue                    # no bar for this symbol now
            od = stop_book[sym]
            stop_px = float(od.meta["stop_price"])
            row = panel[sym].iloc[ix]
            if pos.side > 0:                # long exit: sell stop below
                triggered = float(row["low"]) <= stop_px
                base = min(float(row["open"]), stop_px)
            else:                           # short exit: buy stop above
                triggered = float(row["high"]) >= stop_px
                base = max(float(row["open"]), stop_px)
            if triggered:
                fill = costs.fill_price(base, od.side, pos.instrument)
                broker.execute(od, base, t_open, fill_price=fill)
                del stop_book[sym]

        # ---- advance visibility: this bar closes at t_open + duration
        t_close = t_open + bar_duration
        for sym in panel:
            ix = pos_in_ts[sym]
            if ix < len(open_ts[sym]) and open_ts[sym][ix] == t_open:
                pos_in_ts[sym] = ix + 1
                upto[sym] = ix + 1

        # ---- decision at bar close
        snap = MarketSnapshot(panel, dict(upto), t_close, broker)
        for od in strategy.on_bar_close(snap) or []:
            if od.meta.get("order_type") == "stop":
                stop_book[od.symbol] = od   # cancel/replace semantics
            else:
                pending.append(od)

        marks = {s: float(panel[s]["close"].iloc[upto[s] - 1]) for s in panel if upto[s] > 0}
        equity_pts.append((t_close, broker.equity(marks)))

    if close_at_end and broker.positions:
        # liquidate remaining positions at the last visible close so every
        # entry is represented in `trades` (metrics/stats completeness)
        last_ts = equity_pts[-1][0]
        marks = {s: float(panel[s]["close"].iloc[upto[s] - 1]) for s in panel if upto[s] > 0}
        for sym, pos in list(broker.positions.items()):
            broker.execute(Order(sym, -pos.side, pos.qty, reason="end_of_data",
                                 instrument=pos.instrument, meta={"close": True}),
                           marks.get(sym, pos.entry_price), last_ts)
        equity_pts[-1] = (last_ts, broker.equity(marks))

    curve = pd.Series({ts: eq for ts, eq in equity_pts}).sort_index()
    return BacktestResult(
        equity_curve=curve,
        trades=broker.trades,
        total_commission=broker.total_commission,
        total_financing=broker.total_financing,
        final_equity=float(curve.iloc[-1]),
        initial_equity=initial_cash,
    )


# ---------------------------------------------------------------- metrics

def performance_metrics(result: BacktestResult, bars_per_year: float) -> dict:
    r = result.returns
    eq = result.equity_curve
    if len(r) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": 0.0, "calmar": 0.0,
                "win_rate": 0.0, "payoff": 0.0, "n_trades": len(result.trades)}
    mu, sd = r.mean(), r.std()
    sharpe = float(mu / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0
    downside = r[r < 0].std()
    sortino = float(mu / downside * np.sqrt(bars_per_year)) if downside and downside > 0 else 0.0
    years = len(r) / bars_per_year
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1)
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    wins = [t for t in result.trades if t.pnl > 0]
    losses = [t for t in result.trades if t.pnl <= 0]
    win_rate = len(wins) / len(result.trades) if result.trades else 0.0
    avg_w = np.mean([t.pnl for t in wins]) if wins else 0.0
    avg_l = abs(np.mean([t.pnl for t in losses])) if losses else 0.0
    payoff = float(avg_w / avg_l) if avg_l > 0 else 0.0
    return {"sharpe": sharpe, "sortino": sortino, "cagr": cagr, "max_dd": max_dd,
            "calmar": calmar, "win_rate": win_rate, "payoff": payoff,
            "n_trades": len(result.trades)}


def equity_metrics(curve: pd.Series, bars_per_year: float) -> dict:
    """Metrics for a raw equity curve (e.g. QQQ buy&hold benchmark)."""
    fake = BacktestResult(equity_curve=curve, trades=[], total_commission=0.0,
                          total_financing=0.0, final_equity=float(curve.iloc[-1]),
                          initial_equity=float(curve.iloc[0]))
    return performance_metrics(fake, bars_per_year)
