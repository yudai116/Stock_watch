"""Alpaca ingestion: 1h + daily bars into the bitemporal store (D1).

Bars are fetched RAW (unadjusted); split/dividend adjustment is applied
point-in-time by ``data/adjuster.py`` so that at any ``as_of`` only splits
already announced/effective are reflected. Corporate actions are stored as
bitemporal records.

Usage:
    python -m data.alpaca_ingest --symbols NVDA,AMD --years 10
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from data.bitemporal_store import BitemporalStore, Record
from data.config_loader import data_root, load_dotenv_if_present

UTC = timezone.utc

# Processing latency added on top of bar close before a bar becomes usable.
BAR_LATENCY = pd.Timedelta(minutes=1)


def _clients():
    from alpaca.data.historical import StockHistoricalDataClient

    key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET_KEY"]
    return StockHistoricalDataClient(key, secret)


def fetch_bars(
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch raw bars from Alpaca. timeframe in {'1h', '1d'}."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = TimeFrame(1, TimeFrameUnit.Hour) if timeframe == "1h" else TimeFrame(1, TimeFrameUnit.Day)
    client = _clients()
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start,
        end=end,
        adjustment="raw",
        feed="sip",
    )
    barset = client.get_stock_bars(req)
    out: dict[str, pd.DataFrame] = {}
    df = barset.df.reset_index()
    for sym, g in df.groupby("symbol"):
        g = g.rename(columns={"timestamp": "ts"})
        out[str(sym)] = g[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return out


def fetch_corporate_actions(symbols: list[str], start: datetime, end: datetime) -> list[Record]:
    """Fetch splits/dividends. available_ts = ex_date market open (conservative:
    the action is public before its ex_date; using ex_date open never leaks)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCorporateAnnouncementsRequest

    key = os.environ["ALPACA_API_KEY"]
    secret = os.environ["ALPACA_SECRET_KEY"]
    client = TradingClient(key, secret, paper=True)
    records: list[Record] = []
    # API window is limited to 90 days per request
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=90), end)
        req = GetCorporateAnnouncementsRequest(
            ca_types=["split", "dividend"], since=cur.date(), until=chunk_end.date()
        )
        try:
            anns = client.get_corporate_announcements(req)
        except Exception:
            anns = []
        for a in anns:
            sym = getattr(a, "initiating_symbol", None) or getattr(a, "target_symbol", None)
            if not sym or sym not in symbols:
                continue
            ex_date = getattr(a, "ex_date", None)
            if ex_date is None:
                continue
            ex_open = datetime(ex_date.year, ex_date.month, ex_date.day, 13, 30, tzinfo=UTC)
            ca_type = str(getattr(a, "ca_type", ""))
            records.append(
                Record(
                    key=f"{sym}|{ca_type}|{ex_date.isoformat()}",
                    event_ts=ex_open,
                    available_ts=ex_open,
                    payload={
                        "symbol": str(sym),
                        "type": ca_type,
                        "old_rate": float(getattr(a, "old_rate", 1) or 1),
                        "new_rate": float(getattr(a, "new_rate", 1) or 1),
                        "cash": float(getattr(a, "cash", 0) or 0),
                    },
                )
            )
        cur = chunk_end
    return records


def ingest(symbols: list[str], years: int, store: BitemporalStore | None = None) -> None:
    load_dotenv_if_present()
    store = store or BitemporalStore(data_root())
    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25))
    for tf in ("1h", "1d"):
        bars = fetch_bars(symbols, tf, start, end)
        for sym, df in bars.items():
            store.put_bars(sym, tf, df, latency=BAR_LATENCY)
            print(f"ingested {sym} {tf}: {len(df)} bars")
    actions = fetch_corporate_actions(symbols, start, end)
    if actions:
        store.put_records("corporate_actions", actions)
        print(f"ingested {len(actions)} corporate actions")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, help="comma separated")
    p.add_argument("--years", type=int, default=10)
    args = p.parse_args()
    ingest([s.strip().upper() for s in args.symbols.split(",")], args.years)


if __name__ == "__main__":
    main()
