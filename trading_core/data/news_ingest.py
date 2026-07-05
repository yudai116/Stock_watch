"""Alpaca News API ingestion (D1). Stored bitemporally:

  event_ts     = publication time
  available_ts = publication time + feed latency (params.yaml sentiment.news_latency_minutes)

Bar assignment ("usable from the next complete 1h bar", SPEC §3.2) is done by
``altdata/latency_aligner.py`` at read time — the store only guarantees
available_ts correctness.

Usage:
    python -m data.news_ingest --symbols NVDA,AMD --years 10
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from data.bitemporal_store import BitemporalStore, Record
from data.config_loader import (alpaca_credentials, data_root, load_config,
                                load_dotenv_if_present)

UTC = timezone.utc


def fetch_news(symbols: list[str], start: datetime, end: datetime) -> list[Record]:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    latency = pd.Timedelta(minutes=load_config("params")["sentiment"]["news_latency_minutes"])
    client = NewsClient(*alpaca_credentials())
    records: list[Record] = []
    page_token = None
    while True:
        req = NewsRequest(
            symbols=",".join(symbols),
            start=start,
            end=end,
            limit=50,
            include_content=False,
            page_token=page_token,
        )
        resp = client.get_news(req)
        items = resp.data.get("news", []) if hasattr(resp, "data") else resp.news
        for n in items:
            created = pd.Timestamp(n.created_at).tz_convert("UTC")
            for sym in n.symbols:
                if sym not in symbols:
                    continue
                records.append(
                    Record(
                        # key = symbol|article-id so two articles about one
                        # symbol at the same time are never version-collapsed
                        key=f"{sym}|{n.id}",
                        event_ts=created,
                        available_ts=created + latency,
                        payload={
                            "symbol": sym,
                            "id": str(n.id),
                            "headline": n.headline,
                            "summary": getattr(n, "summary", "") or "",
                            "source": getattr(n, "source", "") or "",
                        },
                    )
                )
        page_token = getattr(resp, "next_page_token", None)
        if not page_token:
            break
    return records


def ingest(symbols: list[str], years: int, store: BitemporalStore | None = None) -> int:
    load_dotenv_if_present()
    store = store or BitemporalStore(data_root())
    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25))
    records = fetch_news(symbols, start, end)
    n = store.put_records("news", records)
    print(f"ingested {n} news records for {len(symbols)} symbols")
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    p.add_argument("--years", type=int, default=10)
    args = p.parse_args()
    ingest([s.strip().upper() for s in args.symbols.split(",")], args.years)


if __name__ == "__main__":
    main()
