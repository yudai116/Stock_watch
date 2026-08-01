"""Export all stored bars to plain CSV files (human-readable backup).

The bitemporal store already persists everything as Parquet under
datastore/bars/ — this exporter is for an additional, portable, spreadsheet-
friendly copy (e.g. before cancelling a data subscription).

    python -m data.export_bars                 # 1d bars -> exports/1d/*.csv
    python -m data.export_bars --timeframe 1h  # hourly too, if stored
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.bitemporal_store import BitemporalStore
from data.config_loader import REPO_ROOT, data_root

UTC = timezone.utc


def export_all(store: BitemporalStore, timeframe: str,
               out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or (REPO_ROOT / "exports" / timeframe)
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = datetime.now(UTC)
    manifest = []
    for sym in store.list_bar_symbols(timeframe):
        bars = store.get_bars_asof(sym, timeframe, as_of)
        if bars.empty:
            continue
        path = out_dir / f"{sym}.csv"
        bars.drop(columns=["available_ts"]).to_csv(path, index=False)
        manifest.append({
            "symbol": sym, "n_bars": len(bars),
            "first": bars["ts"].min(), "last": bars["ts"].max(),
            "file": path.name,
        })
    df = pd.DataFrame(manifest)
    if not df.empty:
        df.to_csv(out_dir / "_manifest.csv", index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1d", choices=["1d", "1h"])
    args = p.parse_args()
    store = BitemporalStore(data_root())
    df = export_all(store, args.timeframe)
    if df.empty:
        print(f"no {args.timeframe} bars stored")
    else:
        print(df.to_string(index=False))
        print(f"\nexported {len(df)} symbols "
              f"({int(df['n_bars'].sum()):,} bars) -> exports/{args.timeframe}/")


if __name__ == "__main__":
    main()
