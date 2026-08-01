"""Point-in-time universe, rebalanced quarterly (SPEC §3.4, D5).

At each quarter start Q the universe is built from information available at
Q only (enforced by going through ``store.view(Q)``):

  1. candidates = all symbols with bars stored AND available at Q
     (delisted names naturally drop out once their bars stop; they remain
     tradable in the quarters where they were selected)
  2. screens: min price, min history, 20d median dollar volume
  3. sector caps, then rank by dollar volume; take min_names..max_names

Membership is written to the store as dataset ``universe`` with
event_ts = available_ts = Q, so any later read is PIT-consistent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from data.adjuster import adjusted_bars_asof
from data.bitemporal_store import BitemporalStore, Record
from data.config_loader import data_root, load_config

UTC = timezone.utc


def sector_of(symbol: str, cfg: dict) -> str:
    for sector, spec in cfg["sectors"].items():
        if symbol in (spec.get("seed") or []):
            return sector
    return "other"


def quarter_starts(start: datetime, end: datetime) -> list[pd.Timestamp]:
    qs = pd.date_range(pd.Timestamp(start).tz_convert("UTC").normalize(),
                       pd.Timestamp(end).tz_convert("UTC").normalize(),
                       freq="QS", tz="UTC")
    return list(qs)


def build_universe(store: BitemporalStore, as_of: datetime) -> list[str]:
    """Select 25-40 names using ONLY data available at ``as_of``."""
    cfg = load_config("universe")
    elig = cfg["eligibility"]
    view = store.view(as_of)

    rows = []
    for sym in store.list_bar_symbols("1d"):
        bars = adjusted_bars_asof(view, sym, "1d")
        if len(bars) < elig["min_history_days"]:
            continue
        recent = bars.tail(20)
        # a live listing must have traded recently as of the rebalance date
        if (view.as_of - recent["ts"].iloc[-1]) > pd.Timedelta(days=10):
            continue
        price = float(recent["close"].iloc[-1])
        if price < elig["min_price"]:
            continue
        med_dollar_vol = float((recent["close"] * recent["volume"]).median())
        if med_dollar_vol < elig["min_median_dollar_volume_20d"]:
            continue
        rows.append({"symbol": sym, "sector": sector_of(sym, cfg),
                     "dollar_vol": med_dollar_vol})

    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values("dollar_vol", ascending=False)

    max_names = cfg["rebalance"]["max_names"]
    min_names = cfg["rebalance"]["min_names"]
    max_per_sector = int(max_names * cfg["concentration"]["max_per_sector_fraction"])

    selected: list[str] = []
    sector_counts: dict[str, int] = {}
    for _, r in df.iterrows():
        if len(selected) >= max_names:
            break
        if sector_counts.get(r["sector"], 0) >= max_per_sector:
            continue
        selected.append(r["symbol"])
        sector_counts[r["sector"]] = sector_counts.get(r["sector"], 0) + 1

    if len(selected) < min_names:
        # fill ignoring sector caps rather than trade an undersized universe
        for _, r in df.iterrows():
            if r["symbol"] not in selected:
                selected.append(r["symbol"])
            if len(selected) >= min_names:
                break
    return selected


def rebuild_all(store: BitemporalStore, start: datetime, end: datetime) -> pd.DataFrame:
    out = []
    for q in quarter_starts(start, end):
        members = build_universe(store, q)
        if members:
            store.put_records(
                "universe",
                [Record(key=m, event_ts=q, available_ts=q, payload={"quarter": str(q.date())})
                 for m in members],
            )
        out.append({"quarter": q, "n": len(members), "members": members})
        print(f"{q.date()}: {len(members)} names")
    return pd.DataFrame(out)


def members_asof(view, as_of_quarter_lookback_days: int = 100) -> list[str]:
    """Current universe members as of view.as_of (latest quarter available)."""
    df = view.records("universe")
    if df.empty:
        return []
    latest_q = df["event_ts"].max()
    return sorted(df[df["event_ts"] == latest_q]["key"].unique().tolist())


def membership_provider(store: BitemporalStore):
    """Return ``provider(ts) -> set[str]`` for backtests: the universe as it
    was known AT ts (latest quarterly rebalance with event_ts <= ts).

    Membership records carry available_ts = quarter start, so bisecting on
    event_ts is PIT-correct. Returns None when no universe has been built.
    """
    df = store.get_records_asof("universe", datetime.now(UTC))
    if df.empty:
        return None
    by_q = df.groupby("event_ts")["key"].apply(set).sort_index()
    quarters = by_q.index
    sets = list(by_q.values)

    def provider(ts) -> set:
        i = quarters.searchsorted(pd.Timestamp(ts), side="right") - 1
        return sets[i] if i >= 0 else set()

    provider.n_quarters = len(quarters)          # introspection for reports
    provider.first_quarter = quarters[0]
    return provider


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()
    store = BitemporalStore(data_root())
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
    rebuild_all(store, start, end)


if __name__ == "__main__":
    main()
