"""Bitemporal store: the ONLY data retrieval path for features/signals/risk.

Every record carries two timestamps (SPEC §3.1):
  * ``event_ts``     — when the underlying event happened (bar period, filing
    quarter end, news publication, ...)
  * ``available_ts`` — when the information became usable by a decision
    process (bar close + latency, EDGAR acceptance datetime, ...)

All reads go through ``as_of`` methods that filter ``available_ts <= as_of``.
Revisions (e.g. 10-K/A) are stored as new versions with their own
``available_ts``; nothing is ever overwritten (append-only).

Storage:
  * bars       -> Parquet, one file per (symbol, timeframe)
  * records    -> SQLite (news, fundamentals, universe membership,
                  earnings calendar, trends, ...), payload as JSON
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

UTC = timezone.utc

BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def _ensure_utc(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"naive timestamp forbidden in bitemporal store: {ts!r}")
    return t.tz_convert("UTC")


@dataclass(frozen=True)
class Record:
    """A single bitemporal record.

    IMPORTANT key convention: with ``latest_version_only`` reads, records
    sharing (key, event_ts) are treated as REVISIONS of one fact and only
    the latest available version is returned. The key must therefore
    identify the fact uniquely: e.g. ``NVDA|cash`` for fundamentals,
    ``NVDA|<article-id>`` for news, plain ``NVDA`` for universe membership.
    Put the bare symbol in payload["symbol"] for cross-fact filtering.
    """

    key: str
    event_ts: datetime
    available_ts: datetime
    payload: dict[str, Any] = field(default_factory=dict)


class BitemporalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "meta.sqlite"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                dataset      TEXT NOT NULL,
                key          TEXT NOT NULL,
                event_ts     TEXT NOT NULL,
                available_ts TEXT NOT NULL,
                inserted_at  TEXT NOT NULL,
                payload      TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_records ON records"
            " (dataset, key, event_ts, available_ts)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------ bars

    def _bar_path(self, symbol: str, timeframe: str) -> Path:
        return self.bars_dir / f"{symbol.upper()}__{timeframe}.parquet"

    def put_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: pd.DataFrame,
        latency: pd.Timedelta = pd.Timedelta(0),
    ) -> None:
        """Store bars. ``ts`` is the bar OPEN time (tz-aware).

        ``available_ts`` is computed here as bar close + latency (SPEC §3.2:
        a 1h bar becomes usable at the start of the next bar). Callers may
        instead provide an explicit ``available_ts`` column.
        """
        df = bars.copy()
        missing = [c for c in BAR_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"bars missing columns: {missing}")
        df["ts"] = df["ts"].map(_ensure_utc)
        if "available_ts" not in df.columns:
            df["available_ts"] = df["ts"] + _bar_duration(timeframe) + latency
        else:
            df["available_ts"] = df["available_ts"].map(_ensure_utc)
        df = df[BAR_COLUMNS + ["available_ts"]].sort_values("ts")

        path = self._bar_path(symbol, timeframe)
        if path.exists():
            old = pd.read_parquet(path)
            old["ts"] = pd.to_datetime(old["ts"], utc=True)
            old["available_ts"] = pd.to_datetime(old["available_ts"], utc=True)
            df = (
                pd.concat([old, df])
                .drop_duplicates(subset="ts", keep="last")
                .sort_values("ts")
            )
        df.reset_index(drop=True).to_parquet(path, index=False)

    def get_bars_asof(
        self,
        symbol: str,
        timeframe: str,
        as_of: datetime,
        start: datetime | None = None,
    ) -> pd.DataFrame:
        """Return bars with ``available_ts <= as_of`` (leak-proof)."""
        as_of = _ensure_utc(as_of)
        path = self._bar_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=BAR_COLUMNS + ["available_ts"])
        df = pd.read_parquet(path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["available_ts"] = pd.to_datetime(df["available_ts"], utc=True)
        df = df[df["available_ts"] <= as_of]
        if start is not None:
            df = df[df["ts"] >= _ensure_utc(start)]
        return df.reset_index(drop=True)

    def list_bar_symbols(self, timeframe: str) -> list[str]:
        return sorted(
            p.name.split("__")[0]
            for p in self.bars_dir.glob(f"*__{timeframe}.parquet")
        )

    # --------------------------------------------------------------- records

    def put_records(self, dataset: str, records: Iterable[Record]) -> int:
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                dataset,
                r.key,
                _ensure_utc(r.event_ts).isoformat(),
                _ensure_utc(r.available_ts).isoformat(),
                now,
                json.dumps(r.payload, ensure_ascii=False, default=str),
            )
            for r in records
        ]
        self._conn.executemany(
            "INSERT INTO records (dataset, key, event_ts, available_ts,"
            " inserted_at, payload) VALUES (?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_records_asof(
        self,
        dataset: str,
        as_of: datetime,
        keys: list[str] | None = None,
        event_from: datetime | None = None,
        event_to: datetime | None = None,
        latest_version_only: bool = True,
    ) -> pd.DataFrame:
        """Return records with ``available_ts <= as_of``.

        With ``latest_version_only`` (default) only the most recent revision
        of each (key, event_ts) — as known at ``as_of`` — is returned, i.e.
        a 10-K/A supersedes its 10-K only once its own available_ts passes.
        """
        as_of_s = _ensure_utc(as_of).isoformat()
        q = "SELECT key, event_ts, available_ts, payload FROM records WHERE dataset=? AND available_ts<=?"
        args: list[Any] = [dataset, as_of_s]
        if keys:
            q += f" AND key IN ({','.join('?' * len(keys))})"
            args += list(keys)
        if event_from is not None:
            q += " AND event_ts>=?"
            args.append(_ensure_utc(event_from).isoformat())
        if event_to is not None:
            q += " AND event_ts<=?"
            args.append(_ensure_utc(event_to).isoformat())
        q += " ORDER BY key, event_ts, available_ts"
        rows = self._conn.execute(q, args).fetchall()
        df = pd.DataFrame(rows, columns=["key", "event_ts", "available_ts", "payload"])
        if df.empty:
            return df
        df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
        df["available_ts"] = pd.to_datetime(df["available_ts"], utc=True)
        df["payload"] = df["payload"].map(json.loads)
        if latest_version_only:
            df = df.groupby(["key", "event_ts"], as_index=False).last()
        return df.reset_index(drop=True)

    # ----------------------------------------------------------------- views

    def view(self, as_of: datetime) -> "AsOfView":
        return AsOfView(self, as_of)

    def close(self) -> None:
        self._conn.close()


class AsOfView:
    """A read-only facade frozen at one ``as_of``.

    Features/signals/risk modules must accept an AsOfView (or data derived
    from one), never a raw store — this is the "no other retrieval path"
    guarantee of SPEC §3.1.
    """

    def __init__(self, store: BitemporalStore, as_of: datetime):
        self.as_of = _ensure_utc(as_of)
        self._store = store

    def bars(self, symbol: str, timeframe: str, start: datetime | None = None) -> pd.DataFrame:
        return self._store.get_bars_asof(symbol, timeframe, self.as_of, start=start)

    def records(self, dataset: str, **kw: Any) -> pd.DataFrame:
        return self._store.get_records_asof(dataset, self.as_of, **kw)


def _bar_duration(timeframe: str) -> pd.Timedelta:
    tf = timeframe.lower()
    table = {
        "1h": pd.Timedelta(hours=1),
        "1hour": pd.Timedelta(hours=1),
        "10min": pd.Timedelta(minutes=10),
        "1d": pd.Timedelta(days=1),
        "1day": pd.Timedelta(days=1),
    }
    if tf not in table:
        raise ValueError(f"unknown timeframe: {timeframe}")
    return table[tf]
