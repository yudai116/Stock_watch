"""VIX daily close ingestion from FRED (series VIXCLS, no API key needed).

Bitemporal mapping: event_ts = trading day close (21:00 UTC);
available_ts = next calendar day 00:00 UTC (FRED publishes end-of-day with a
lag — conservative one-day availability so a decision on day t uses the
PREVIOUS day's VIX, never the same-day print).

Usage:
    python -m data.vix_ingest
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import httpx
import pandas as pd

from data.bitemporal_store import BitemporalStore, Record
from data.config_loader import data_root

UTC = timezone.utc
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def fetch_vix() -> pd.DataFrame:
    r = httpx.get(FRED_CSV, timeout=30, follow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "vix"]
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    return df


def ingest(store: BitemporalStore | None = None) -> int:
    store = store or BitemporalStore(data_root())
    df = fetch_vix()
    records = [
        Record(
            key="VIXCLS",
            event_ts=row["date"].tz_localize("UTC") + pd.Timedelta(hours=21),
            available_ts=row["date"].tz_localize("UTC") + pd.Timedelta(days=1),
            payload={"value": float(row["vix"])},
        )
        for _, row in df.iterrows()
    ]
    n = store.put_records("vix", records)
    print(f"ingested {n} VIX observations ({df['date'].min().date()} .. {df['date'].max().date()})")
    return n


def vix_series_asof(view) -> pd.Series:
    """VIX indexed by event day-close ts, containing only values available
    at view.as_of."""
    df = view.records("vix", keys=["VIXCLS"])
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series([p["value"] for p in df["payload"]],
                  index=pd.DatetimeIndex(df["event_ts"]))
    return s.sort_index()


if __name__ == "__main__":
    ingest()
