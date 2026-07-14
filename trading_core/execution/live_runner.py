"""Phase 5 daily runner for the ADOPTED production strategy (flow-F [D]: 3d
regime overlay — QQQ core, simple regime filter, zero tuned parameters).

Contract mirrors the backtest exactly:
  * decision AFTER the daily close (regime from data available then)
  * resulting order FILLS at the NEXT day's open (executed on the next run)
  * bull -> invest_fraction in QQQ / range -> hold / bear|crisis -> cash
  * kill switch (addendum G-3): paper/live drawdown >= killswitch_dd_pct
    liquidates everything and HALTS the system until a human review resets it

Modes:
  paper : local paper book (SQLite), fills simulated with the real cost model
  saxo  : additionally sends the order to Saxo (SIM unless SAXO_ENV=live;
          live is refused without --allow-live). Paper book stays the
          authoritative record for the G-gate comparison.

Daily usage (Windows Task Scheduler, after the US close):
  uv run python -m execution.live_runner run --mode paper
Other commands:
  uv run python -m execution.live_runner status
  uv run python -m execution.live_runner report
  uv run python -m execution.live_runner reset-killswitch
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import CASH, CostModel
from data.bitemporal_store import BitemporalStore
from data.config_loader import data_root, load_config, load_dotenv_if_present
from data.vix_ingest import vix_series_asof
from features.simple_regime import simple_regime

UTC = timezone.utc


# ----------------------------------------------------------------- state

class LiveState:
    """Persistent paper book + decision log (append-only where it matters)."""

    def __init__(self, root: Path | None = None):
        cfg = load_config("params")["live"]
        root = Path(root) if root is not None else data_root()
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "live_state.sqlite")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            bar_date TEXT PRIMARY KEY, regime TEXT, action TEXT,
            target_qty REAL, detail TEXT, decided_at TEXT);
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_bar_date TEXT,
            symbol TEXT, side INTEGER, qty REAL, status TEXT,
            fill_price REAL, fill_fee REAL, fill_bar_date TEXT, sent_saxo INTEGER);
        CREATE TABLE IF NOT EXISTS equity (
            bar_date TEXT PRIMARY KEY, equity REAL, cash REAL,
            qty REAL, close REAL, regime TEXT);
        CREATE TABLE IF NOT EXISTS flags (key TEXT PRIMARY KEY, value TEXT);
        """)
        if self._flag("cash") is None:
            self._set_flag("cash", str(float(cfg["paper_initial_cash"])))
        self.db.commit()

    # flags ------------------------------------------------------------
    def _flag(self, key: str):
        row = self.db.execute("SELECT value FROM flags WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_flag(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO flags VALUES (?,?)", (key, value))
        self.db.commit()

    @property
    def cash(self) -> float:
        return float(self._flag("cash"))

    @cash.setter
    def cash(self, v: float) -> None:
        self._set_flag("cash", str(float(v)))

    @property
    def halted(self) -> bool:
        return self._flag("halted") == "1"

    def halt(self, reason: str) -> None:
        self._set_flag("halted", "1")
        self._set_flag("halt_reason", reason)

    def reset_killswitch(self) -> None:
        self._set_flag("halted", "0")
        self._set_flag("halt_reason", "")

    # positions (single core symbol; qty derived from fills) ------------
    def position_qty(self, symbol: str) -> float:
        row = self.db.execute(
            "SELECT COALESCE(SUM(side*qty),0) FROM orders WHERE symbol=? AND status='filled'",
            (symbol,)).fetchone()
        return float(row[0])

    # orders ------------------------------------------------------------
    def pending_orders(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, created_bar_date, symbol, side, qty FROM orders "
            "WHERE status='pending' ORDER BY id").fetchall()
        return [dict(zip(["id", "created_bar_date", "symbol", "side", "qty"], r))
                for r in rows]

    def add_order(self, bar_date: str, symbol: str, side: int, qty: float,
                  sent_saxo: bool = False) -> None:
        self.db.execute(
            "INSERT INTO orders (created_bar_date, symbol, side, qty, status, sent_saxo)"
            " VALUES (?,?,?,?,'pending',?)",
            (bar_date, symbol, side, qty, int(sent_saxo)))
        self.db.commit()

    def fill_order(self, order_id: int, price: float, fee: float, bar_date: str) -> None:
        self.db.execute(
            "UPDATE orders SET status='filled', fill_price=?, fill_fee=?, "
            "fill_bar_date=? WHERE id=?", (price, fee, bar_date, order_id))
        self.db.commit()

    # decisions / equity --------------------------------------------------
    def has_decision(self, bar_date: str) -> bool:
        return self.db.execute("SELECT 1 FROM decisions WHERE bar_date=?",
                               (bar_date,)).fetchone() is not None

    def add_decision(self, bar_date: str, regime: str, action: str,
                     target_qty: float, detail: dict) -> None:
        self.db.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?)",
            (bar_date, regime, action, target_qty,
             json.dumps(detail, default=str), datetime.now(UTC).isoformat()))
        self.db.commit()

    def record_equity(self, bar_date: str, equity: float, cash: float,
                      qty: float, close: float, regime: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?)",
                        (bar_date, equity, cash, qty, close, regime))
        self.db.commit()

    def equity_history(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM equity ORDER BY bar_date", self.db)

    def peak_equity(self) -> float:
        row = self.db.execute("SELECT MAX(equity) FROM equity").fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0


# ------------------------------------------------------------- daily cycle

def run_once(store: BitemporalStore, state: LiveState,
             cost_model: CostModel | None = None,
             saxo_client=None) -> dict:
    """One daily cycle. Idempotent per trading day (bar date)."""
    cfg = load_config("params")["live"]
    costs = (cost_model or CostModel.from_config()).profile(CASH)
    sym = str(cfg["core_symbol"])
    now = datetime.now(UTC)

    bars = store.get_bars_asof(sym, "1d", now)
    if len(bars) < int(cfg["min_history_days"]):
        return {"status": "warmup", "bars": len(bars),
                "needed": int(cfg["min_history_days"])}
    last = bars.iloc[-1]
    bar_date = str(last["ts"].date())
    today_open = float(last["open"])
    close = float(last["close"])

    # ---- 1. fill orders decided on earlier days at TODAY's open
    fills = []
    for od in state.pending_orders():
        if od["created_bar_date"] >= bar_date:
            continue                        # created today: fills tomorrow
        px = costs.fill_price(today_open, od["side"])
        fee = costs.commission(px * od["qty"])
        state.cash = state.cash - od["side"] * px * od["qty"] - fee
        state.fill_order(od["id"], px, fee, bar_date)
        fills.append({"side": od["side"], "qty": od["qty"], "price": px, "fee": fee})

    # ---- 2. idempotency: one decision per bar date
    if state.has_decision(bar_date):
        return {"status": "already_decided", "bar_date": bar_date, "fills": fills}

    # ---- 3. regime from data available now (PIT: previous-day VIX)
    close_series = pd.Series(bars["close"].to_numpy(),
                             index=pd.DatetimeIndex(bars["ts"]) + pd.Timedelta(days=1))
    vix = vix_series_asof(store.view(now))
    labels = simple_regime(close_series, vix if len(vix) else None)
    regime = str(labels.iloc[-1]) if len(labels) else "unknown"

    # ---- 4. mark equity
    qty = state.position_qty(sym)
    equity = state.cash + qty * close
    peak = max(state.peak_equity(), equity)

    # ---- 5. kill switch (addendum G-3)
    ks = float(cfg["killswitch_dd_pct"]) / 100.0
    dd = 1.0 - equity / peak if peak > 0 else 0.0
    action, target_qty = "hold", qty
    if not state.halted and dd >= ks:
        state.halt(f"kill switch: drawdown {dd:.1%} >= {ks:.1%} on {bar_date}")
        if qty > 0:
            state.add_order(bar_date, sym, -1, qty)
        action, target_qty = "KILL_SWITCH_LIQUIDATE", 0.0
    elif state.halted:
        action = "halted"
    else:
        # ---- 6. target book per adopted 3d rules
        pending_delta = sum(o["side"] * o["qty"] for o in state.pending_orders())
        effective_qty = qty + pending_delta      # avoid double-ordering
        if regime == "bull":
            target_qty = equity * float(cfg["invest_fraction"]) / close
        elif regime in ("bear", "crisis"):
            target_qty = 0.0
        else:                                    # range/unknown: hold
            target_qty = effective_qty
        delta = target_qty - effective_qty
        if abs(delta) * close >= float(cfg["min_order_frac"]) * max(equity, 1e-9):
            side = 1 if delta > 0 else -1
            oqty = abs(delta) if delta > 0 else min(abs(delta), max(effective_qty, 0.0))
            if oqty > 0:
                sent = False
                if saxo_client is not None:
                    saxo_client.place_market_order(sym, side, oqty, "Stock")
                    sent = True
                state.add_order(bar_date, sym, side, oqty, sent_saxo=sent)
                action = "buy" if side > 0 else "sell"

    state.add_decision(bar_date, regime, action, target_qty,
                       {"equity": equity, "cash": state.cash, "qty": qty,
                        "close": close, "dd": dd, "fills": fills})
    state.record_equity(bar_date, equity, state.cash, qty, close, regime)
    return {"status": "ok", "bar_date": bar_date, "regime": regime,
            "action": action, "equity": equity, "qty": qty, "dd": dd,
            "halted": state.halted, "fills": fills}


# ------------------------------------------------------------------ report

def realized_report(state: LiveState) -> dict:
    cfg = load_config("params")["live"]
    hist = state.equity_history()
    if len(hist) < 5:
        return {"status": "insufficient_history", "days": len(hist)}
    eq = hist["equity"]
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    years = len(r) / 252
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1)
    peak = eq.cummax()
    max_dd = float(((eq - peak) / peak).min())
    ref = cfg["reference"]
    return {
        "status": "ok", "days": len(hist),
        "period": f"{hist['bar_date'].iloc[0]} .. {hist['bar_date'].iloc[-1]}",
        "equity": float(eq.iloc[-1]),
        "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd,
        "reference_backtest": dict(ref),
        # addendum G-1: expectancy vs backtest (evaluated after >= 3 months)
        "cagr_vs_reference_ratio": (cagr / ref["cagr"]) if ref["cagr"] else None,
        "halted": state.halted,
        "halt_reason": state._flag("halt_reason") or "",
    }


# --------------------------------------------------------------------- cli

def _ingest_latest() -> None:
    """Best-effort daily data refresh (QQQ bars + VIX). Failures are loud but
    non-fatal: the runner then decides on the freshest data already stored."""
    from data.alpaca_ingest import ingest as alpaca_ingest
    from data.vix_ingest import ingest as vix_ingest

    sym = str(load_config("params")["live"]["core_symbol"])
    try:
        alpaca_ingest([sym], years=1, timeframes=("1d",))
    except Exception as e:
        print(f"WARN bar ingest failed ({e}); using last stored bars")
    try:
        vix_ingest()
    except Exception as e:
        print(f"WARN VIX ingest failed ({e}); using last stored VIX")


def main() -> None:
    load_dotenv_if_present()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--mode", choices=["paper", "saxo"], default="paper")
    runp.add_argument("--skip-ingest", action="store_true")
    runp.add_argument("--allow-live", action="store_true")
    runp.add_argument("--no-notify", action="store_true",
                      help="suppress the Discord notification for this run")
    sub.add_parser("status")
    sub.add_parser("report")
    sub.add_parser("reset-killswitch")
    sub.add_parser("test-notify")
    args = p.parse_args()

    store = BitemporalStore(data_root())
    state = LiveState()

    if args.cmd == "run":
        from execution.notifier import format_run_summary, send_discord

        if not args.skip_ingest:
            _ingest_latest()
        saxo = None
        if args.mode == "saxo":
            from execution.saxo_client import SaxoClient
            saxo = SaxoClient()
            if saxo.env == "live" and not args.allow_live:
                raise SystemExit("refusing LIVE trading without --allow-live "
                                 "(paper >= 3 months + G gates first)")
        try:
            out = run_once(store, state, saxo_client=saxo)
        except Exception as e:
            if not args.no_notify:
                send_discord(f"❌ trading_core 実行エラー: {e}\n"
                             f"（判定は行われていません。ログを確認してください）")
            raise
        print(json.dumps(out, indent=2, default=str))
        if not args.no_notify:
            send_discord(format_run_summary(out))
        if out.get("halted"):
            print("!! SYSTEM HALTED (kill switch). Review, then "
                  "`python -m execution.live_runner reset-killswitch`")
    elif args.cmd == "status":
        sym = str(load_config("params")["live"]["core_symbol"])
        print(json.dumps({
            "halted": state.halted,
            "halt_reason": state._flag("halt_reason") or "",
            "cash": state.cash,
            "position_qty": state.position_qty(sym),
            "pending_orders": state.pending_orders(),
            "equity_days": len(state.equity_history()),
        }, indent=2, default=str))
    elif args.cmd == "report":
        print(json.dumps(realized_report(state), indent=2, default=str))
    elif args.cmd == "reset-killswitch":
        state.reset_killswitch()
        print("kill switch cleared. Trading resumes on next run.")
    elif args.cmd == "test-notify":
        from execution.notifier import send_discord
        ok = send_discord("✅ trading_core: Discord通知のテストです。"
                          "この通知が見えていれば設定完了。")
        print("sent OK" if ok else
              "FAILED — .env に DISCORD_WEBHOOK_URL を設定してください")


if __name__ == "__main__":
    main()
