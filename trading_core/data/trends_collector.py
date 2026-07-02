"""Google Trends live collector (D10: experimental, NOT true PIT for history).

Run daily via Windows Task Scheduler / cron:
    python -m data.trends_collector

Each run snapshots interest-over-time for universe-related terms and stores
them with available_ts = collection time. ONLY live-collected rows are true
PIT; anything backfilled must be flagged payload["live"]=False and excluded
from validation until enough live history accumulates.

Requires optional dependency: pip install pytrends
"""

from __future__ import annotations

from datetime import datetime, timezone

from data.bitemporal_store import BitemporalStore, Record
from data.config_loader import data_root, load_config

UTC = timezone.utc

TERMS = ["semiconductor shortage", "AI stocks", "nuclear fusion", "space stocks", "memory chips"]


def collect(store: BitemporalStore | None = None) -> int:
    try:
        from pytrends.request import TrendReq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install pytrends to enable trends collection") from e

    store = store or BitemporalStore(data_root())
    now = datetime.now(UTC)
    pt = TrendReq(hl="en-US", tz=0)
    records: list[Record] = []
    for term in TERMS:
        pt.build_payload([term], timeframe="now 7-d")
        df = pt.interest_over_time()
        if df.empty:
            continue
        # store only the most recent complete day; levels are relative, so
        # downstream uses CHANGE RATES only (SPEC §3.3)
        latest = df[term].iloc[-2]  # last row is partial
        records.append(
            Record(
                key=term,
                event_ts=df.index[-2].tz_localize("UTC") if df.index[-2].tzinfo is None else df.index[-2],
                available_ts=now,
                payload={"interest": int(latest), "live": True},
            )
        )
    n = store.put_records("trends", records)
    print(f"collected {n} trend snapshots at {now.isoformat()}")
    return n


if __name__ == "__main__":
    collect()
