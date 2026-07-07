"""One-time store maintenance: collapse duplicate DAILY bars caused by mixed
ingest sources with different intraday stamps (Alpaca 04:00/05:00 UTC vs
Polygon normalized 00:00 UTC).

Rule per calendar day: prefer the normalized (00:00 = Polygon) row; if only
non-normalized rows exist, keep the first and normalize its stamp. Idempotent.

    python -m data.dedupe_daily
"""

from __future__ import annotations

import pandas as pd

from data.bitemporal_store import BAR_COLUMNS, BitemporalStore
from data.config_loader import data_root


def dedupe_symbol(store: BitemporalStore, symbol: str) -> tuple[int, int]:
    """Returns (n_before, n_after)."""
    path = store._bar_path(symbol, "1d")
    if not path.exists():
        return (0, 0)
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["available_ts"] = pd.to_datetime(df["available_ts"], utc=True)
    n_before = len(df)

    day = df["ts"].dt.normalize()
    if not day.duplicated().any() and (df["ts"] == day).all():
        return (n_before, n_before)

    df["_day"] = day
    df["_is_norm"] = df["ts"] == day
    df = (df.sort_values(["_day", "_is_norm"], ascending=[True, False])
            .drop_duplicates("_day", keep="first"))
    df["ts"] = df["_day"]
    df = df[BAR_COLUMNS + ["available_ts"]].sort_values("ts").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return (n_before, len(df))


def main() -> None:
    store = BitemporalStore(data_root())
    total_removed = 0
    for sym in store.list_bar_symbols("1d"):
        before, after = dedupe_symbol(store, sym)
        removed = before - after
        if removed:
            print(f"{sym}: {before} -> {after} (removed {removed} duplicate days)")
            total_removed += removed
    print(f"\ndone. removed {total_removed} duplicate daily rows"
          if total_removed else "\nstore is clean: no duplicate daily rows")


if __name__ == "__main__":
    main()
