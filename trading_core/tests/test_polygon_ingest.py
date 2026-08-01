"""Polygon parsing logic (no network): raw bar frames and corporate-action
records match the bitemporal-store / adjuster contract."""

import pandas as pd
import pytest

from data import polygon_ingest as pg


def test_rows_to_frame_daily_floors_to_utc_date():
    # Polygon `t` = ms since epoch; a daily bar for 2020-01-02 (ET open)
    t = int(pd.Timestamp("2020-01-02 05:00", tz="UTC").timestamp() * 1000)
    rows = [{"t": t, "o": 10.0, "h": 11.0, "l": 9.5, "c": 10.5, "v": 1000}]
    df = pg._rows_to_frame(rows, "day")
    assert df["ts"].iloc[0] == pd.Timestamp("2020-01-02", tz="UTC")
    assert df["ts"].dt.tz is not None
    assert df.loc[0, ["open", "high", "low", "close", "volume"]].tolist() == \
        [10.0, 11.0, 9.5, 10.5, 1000.0]


def test_rows_to_frame_hourly_keeps_time():
    t = int(pd.Timestamp("2020-01-02 14:30", tz="UTC").timestamp() * 1000)
    rows = [{"t": t, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 5}]
    df = pg._rows_to_frame(rows, "hour")
    assert df["ts"].iloc[0] == pd.Timestamp("2020-01-02 14:30", tz="UTC")


def test_rows_to_frame_sorted():
    mk = lambda d: int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)
    rows = [{"t": mk("2020-01-03"), "o": 2, "h": 2, "l": 2, "c": 2, "v": 1},
            {"t": mk("2020-01-02"), "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]
    df = pg._rows_to_frame(rows, "day")
    assert list(df["ts"]) == [pd.Timestamp("2020-01-02", tz="UTC"),
                              pd.Timestamp("2020-01-03", tz="UTC")]


def test_ex_open_is_market_open_utc():
    assert pg._ex_open("2022-07-06") == pd.Timestamp("2022-07-06 13:30", tz="UTC")


def test_split_record_matches_adjuster_contract(monkeypatch):
    import httpx

    def fake_paged(client, url, params, key):
        if "splits" in url:
            return [{"execution_date": "2021-07-20", "split_from": 1, "split_to": 4}]
        return []
    monkeypatch.setattr(pg, "_paged", fake_paged)
    recs = pg._fetch_splits(httpx.Client(), "NVDA", "k")
    assert len(recs) == 1
    r = recs[0]
    assert r.key == "NVDA|split|2021-07-20"
    assert r.payload == {"symbol": "NVDA", "type": "split",
                         "old_rate": 1.0, "new_rate": 4.0, "cash": 0.0}
    assert r.available_ts == pd.Timestamp("2021-07-20 13:30", tz="UTC")


def test_dividend_record_skips_zero_cash(monkeypatch):
    import httpx

    def fake_paged(client, url, params, key):
        return [{"ex_dividend_date": "2023-03-01", "cash_amount": 0.04},
                {"ex_dividend_date": "2023-06-01", "cash_amount": 0.0}]
    monkeypatch.setattr(pg, "_paged", fake_paged)
    recs = pg._fetch_dividends(httpx.Client(), "MSFT", "k")
    assert len(recs) == 1                        # zero-cash dropped
    assert recs[0].payload["type"] == "dividend"
    assert recs[0].payload["cash"] == 0.04


def test_records_feed_adjuster_end_to_end(tmp_path, monkeypatch):
    """A Polygon split record, once stored, drives PIT adjustment correctly."""
    from datetime import datetime, timezone

    from data.adjuster import adjusted_bars_asof
    from data.bitemporal_store import BitemporalStore

    store = BitemporalStore(tmp_path)
    ts = pd.bdate_range("2021-07-14", periods=8, tz="UTC")
    px = [100.0] * 8
    store.put_bars("NVDA", "1d", pd.DataFrame({
        "ts": ts, "open": px, "high": px, "low": px, "close": px,
        "volume": [1000.0] * 8}))

    import httpx
    monkeypatch.setattr(pg, "_paged", lambda c, u, p, k:
                        [{"execution_date": "2021-07-20", "split_from": 1, "split_to": 4}]
                        if "splits" in u else [])
    store.put_records("corporate_actions", pg._fetch_splits(httpx.Client(), "NVDA", "k"))

    view = store.view(datetime(2021, 7, 26, tzinfo=timezone.utc))
    adj = adjusted_bars_asof(view, "NVDA", "1d")
    pre = adj[adj["ts"] < pd.Timestamp("2021-07-20 13:30", tz="UTC")]
    assert (pre["close"].round(4) == 25.0).all()   # pre-split /4
