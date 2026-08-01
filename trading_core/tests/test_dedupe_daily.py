"""Mixed-source duplicate daily bars: the production incident where Alpaca
(04:00 UTC stamps) + Polygon (00:00 UTC) doubled every day for QQQ/SMH."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.bitemporal_store import BitemporalStore
from data.dedupe_daily import dedupe_symbol

UTC = timezone.utc


def _bars(offset_hours: float, close: float, n=5):
    days = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    ts = days + pd.Timedelta(hours=offset_hours)
    return pd.DataFrame({"ts": ts, "open": [close] * n, "high": [close + 1] * n,
                         "low": [close - 1] * n, "close": [close] * n,
                         "volume": [1000.0] * n})


def test_dedupe_prefers_normalized_polygon_rows(tmp_path):
    store = BitemporalStore(tmp_path)
    store.put_bars("QQQ", "1d", _bars(4.0, close=500.0))    # alpaca-style stamps
    store.put_bars("QQQ", "1d", _bars(0.0, close=510.0))    # polygon-style stamps
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    assert len(store.get_bars_asof("QQQ", "1d", as_of)) == 10   # the bug

    before, after = dedupe_symbol(store, "QQQ")
    assert (before, after) == (10, 5)
    bars = store.get_bars_asof("QQQ", "1d", as_of)
    assert len(bars) == 5
    assert (bars["close"] == 510.0).all()                   # polygon rows kept
    assert (bars["ts"] == bars["ts"].dt.normalize()).all()


def test_dedupe_normalizes_alpaca_only_symbol(tmp_path):
    store = BitemporalStore(tmp_path)
    store.put_bars("ONLY", "1d", _bars(4.0, close=100.0))
    before, after = dedupe_symbol(store, "ONLY")
    assert (before, after) == (5, 5)                        # nothing removed
    bars = store.get_bars_asof("ONLY", "1d", datetime(2025, 1, 1, tzinfo=UTC))
    assert (bars["ts"] == bars["ts"].dt.normalize()).all()  # stamps normalized


def test_dedupe_idempotent(tmp_path):
    store = BitemporalStore(tmp_path)
    store.put_bars("QQQ", "1d", _bars(4.0, close=500.0))
    store.put_bars("QQQ", "1d", _bars(0.0, close=510.0))
    dedupe_symbol(store, "QQQ")
    before, after = dedupe_symbol(store, "QQQ")
    assert before == after == 5
