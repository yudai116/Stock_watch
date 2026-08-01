"""Polygon.io ingestion — full 10-year history + delisted coverage (D6).

Drop-in alternative to data/alpaca_ingest for BARS and CORPORATE ACTIONS:
everything downstream reads the bitemporal store, so switching the ingest
source changes nothing else. (News still comes from Alpaca — data/news_ingest.)

Why Polygon here: the paid plan returns ~10y of history and delisted/acquired
tickers, which the free Alpaca IEX feed does not — this is what the PIT
universe and the survivorship-bias guard actually need.

Bitemporal mapping mirrors alpaca_ingest:
  * bars fetched RAW (adjusted=false); PIT split/dividend adjustment is applied
    at read time by data/adjuster.py
  * splits/dividends stored as corporate_actions records, available at ex-date

Usage:
    python -m data.polygon_ingest --symbols QQQ,NVDA,AMD --years 10
    python -m data.polygon_ingest --symbols XLNX,ATVI --years 10 --timeframes 1d
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from data.bitemporal_store import BitemporalStore, Record
from data.config_loader import data_root, load_dotenv_if_present

UTC = timezone.utc
BASE = "https://api.polygon.io"
BAR_LATENCY = pd.Timedelta(minutes=1)

# trading_core timeframe -> Polygon (multiplier, timespan)
_TIMEFRAME = {"1h": (1, "hour"), "1d": (1, "day")}


def _api_key() -> str:
    load_dotenv_if_present()
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "POLYGON_API_KEY not set. Add it to .env "
            "(https://polygon.io/dashboard/api-keys)."
        )
    return key


def _get(client: httpx.Client, url: str, params: dict) -> dict:
    """GET with 429 backoff and a clear 403 (plan history limit) message."""
    while True:
        r = client.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(60)
            continue
        if r.status_code == 403:
            raise PermissionError(
                f"Polygon 403 (plan history/entitlement limit): {r.text[:160]}")
        r.raise_for_status()
        return r.json()


# ------------------------------------------------------------------- bars

def fetch_bars(symbols: list[str], timeframe: str,
               start: datetime, end: datetime | None = None) -> dict[str, pd.DataFrame]:
    """Fetch RAW bars from Polygon. timeframe in {'1h','1d'}."""
    if timeframe not in _TIMEFRAME:
        raise ValueError(f"unsupported timeframe {timeframe}; use {list(_TIMEFRAME)}")
    mult, span = _TIMEFRAME[timeframe]
    key = _api_key()
    end = end or datetime.now(UTC)
    s = pd.Timestamp(start).strftime("%Y-%m-%d")
    e = pd.Timestamp(end).strftime("%Y-%m-%d")

    out: dict[str, pd.DataFrame] = {}
    with httpx.Client() as client:
        for sym in symbols:
            url = f"{BASE}/v2/aggs/ticker/{sym}/range/{mult}/{span}/{s}/{e}"
            params = {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": key}
            rows: list[dict] = []
            try:
                while url:
                    data = _get(client, url, params)
                    rows.extend(data.get("results") or [])
                    nxt = data.get("next_url")
                    url = f"{nxt}&apiKey={key}" if nxt else None
                    params = {}
                    if url:
                        time.sleep(0.05)
            except PermissionError as exc:
                print(f"WARN {sym} {timeframe}: {exc}")
                continue
            if rows:
                out[sym] = _rows_to_frame(rows, span)
    return out


def _rows_to_frame(rows: list[dict], span: str) -> pd.DataFrame:
    """Polygon aggregate rows -> engine bar frame (ts = bar OPEN, tz-aware)."""
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df["t"], unit="ms", utc=True)
    if span == "day":
        ts = ts.dt.normalize()          # daily granularity: floor to UTC date
    return pd.DataFrame({
        "ts": ts,
        "open": df["o"].astype(float),
        "high": df["h"].astype(float),
        "low": df["l"].astype(float),
        "close": df["c"].astype(float),
        "volume": df["v"].astype(float),
    }).sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------- corporate actions

def fetch_corporate_actions(symbols: list[str]) -> list[Record]:
    """Splits + dividends from Polygon reference API, stored bitemporally
    with available_ts = ex-date open (conservative; the action is public
    before its ex-date so this never leaks)."""
    key = _api_key()
    records: list[Record] = []
    with httpx.Client() as client:
        for sym in symbols:
            records += _fetch_splits(client, sym, key)
            records += _fetch_dividends(client, sym, key)
    return records


def _ex_open(date_str: str) -> pd.Timestamp:
    d = pd.Timestamp(date_str, tz="UTC")
    return d.normalize() + pd.Timedelta(hours=13, minutes=30)   # ~US market open


def _paged(client: httpx.Client, url: str, params: dict, key: str) -> list[dict]:
    results: list[dict] = []
    while url:
        data = _get(client, url, params)
        results.extend(data.get("results") or [])
        nxt = data.get("next_url")
        url = f"{nxt}&apiKey={key}" if nxt else None
        params = {}
    return results


def _fetch_splits(client: httpx.Client, sym: str, key: str) -> list[Record]:
    rows = _paged(client, f"{BASE}/v3/reference/splits",
                  {"ticker": sym, "limit": 1000, "apiKey": key}, key)
    out = []
    for r in rows:
        ex = r.get("execution_date")
        if not ex:
            continue
        # split_to-for-split_from (e.g. 4-for-1 -> to=4, from=1 -> ratio 4)
        old = float(r.get("split_from", 1) or 1)
        new = float(r.get("split_to", 1) or 1)
        out.append(Record(
            key=f"{sym}|split|{ex}", event_ts=_ex_open(ex), available_ts=_ex_open(ex),
            payload={"symbol": sym, "type": "split", "old_rate": old, "new_rate": new,
                     "cash": 0.0}))
    return out


def _fetch_dividends(client: httpx.Client, sym: str, key: str) -> list[Record]:
    rows = _paged(client, f"{BASE}/v3/reference/dividends",
                  {"ticker": sym, "limit": 1000, "apiKey": key}, key)
    out = []
    for r in rows:
        ex = r.get("ex_dividend_date")
        cash = float(r.get("cash_amount", 0) or 0)
        if not ex or cash <= 0:
            continue
        out.append(Record(
            key=f"{sym}|dividend|{ex}", event_ts=_ex_open(ex), available_ts=_ex_open(ex),
            payload={"symbol": sym, "type": "dividend", "old_rate": 1.0, "new_rate": 1.0,
                     "cash": cash}))
    return out


# --------------------------------------------------------------- ingest

def ingest(symbols: list[str], years: int, store: BitemporalStore | None = None,
           timeframes: tuple[str, ...] = ("1h", "1d")) -> None:
    store = store or BitemporalStore(data_root())
    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25))
    for tf in timeframes:
        bars = fetch_bars(symbols, tf, start, end)
        for sym, df in bars.items():
            store.put_bars(sym, tf, df, latency=BAR_LATENCY)
            print(f"ingested {sym} {tf}: {len(df)} bars "
                  f"({df['ts'].min().date()}..{df['ts'].max().date()})")
    actions = fetch_corporate_actions(symbols)
    if actions:
        store.put_records("corporate_actions", actions)
        print(f"ingested {len(actions)} corporate actions")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, help="comma separated")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--timeframes", default="1h,1d",
                   help="subset of {1h,1d}; use '1d' for fast Phase-3 prep")
    args = p.parse_args()
    ingest([s.strip().upper() for s in args.symbols.split(",")], args.years,
           timeframes=tuple(t.strip() for t in args.timeframes.split(",")))


if __name__ == "__main__":
    main()
