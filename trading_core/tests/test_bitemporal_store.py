"""Phase 1 acceptance (a): the as_of API is leak-proof by construction."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.bitemporal_store import BitemporalStore, Record

UTC = timezone.utc


@pytest.fixture()
def store(tmp_path):
    return BitemporalStore(tmp_path)


def _bars(n=10, start="2024-01-01 14:00", freq="1h"):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": range(100, 100 + n),
        "high": range(101, 101 + n),
        "low": range(99, 99 + n),
        "close": range(100, 100 + n),
        "volume": [1000] * n,
    })


def test_bars_not_visible_before_close(store):
    store.put_bars("NVDA", "1h", _bars())
    # bar opening 14:00 closes at 15:00 -> not visible at 14:30
    df = store.get_bars_asof("NVDA", "1h", datetime(2024, 1, 1, 14, 30, tzinfo=UTC))
    assert len(df) == 0
    # visible exactly at close
    df = store.get_bars_asof("NVDA", "1h", datetime(2024, 1, 1, 15, 0, tzinfo=UTC))
    assert len(df) == 1
    # only 3 bars closed by 17:59
    df = store.get_bars_asof("NVDA", "1h", datetime(2024, 1, 1, 17, 59, tzinfo=UTC))
    assert len(df) == 3


def test_bars_latency_shifts_availability(store):
    store.put_bars("AMD", "1h", _bars(), latency=pd.Timedelta(minutes=5))
    df = store.get_bars_asof("AMD", "1h", datetime(2024, 1, 1, 15, 4, tzinfo=UTC))
    assert len(df) == 0
    df = store.get_bars_asof("AMD", "1h", datetime(2024, 1, 1, 15, 5, tzinfo=UTC))
    assert len(df) == 1


def test_naive_timestamps_rejected(store):
    bad = _bars()
    bad["ts"] = bad["ts"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="naive"):
        store.put_bars("X", "1h", bad)
    with pytest.raises(ValueError, match="naive"):
        store.get_bars_asof("X", "1h", datetime(2024, 1, 2))  # naive as_of


def test_records_asof_filtering(store):
    store.put_records("news", [
        Record("NVDA", datetime(2024, 1, 1, 10, tzinfo=UTC),
               datetime(2024, 1, 1, 10, 5, tzinfo=UTC), {"h": "early"}),
        Record("NVDA", datetime(2024, 1, 2, 10, tzinfo=UTC),
               datetime(2024, 1, 2, 10, 5, tzinfo=UTC), {"h": "late"}),
    ])
    df = store.get_records_asof("news", datetime(2024, 1, 1, 12, tzinfo=UTC))
    assert len(df) == 1 and df.iloc[0]["payload"]["h"] == "early"
    df = store.get_records_asof("news", datetime(2024, 1, 3, tzinfo=UTC))
    assert len(df) == 2


def test_revision_versioning_no_overwrite(store):
    """A 10-K/A supersedes the 10-K only after ITS OWN available_ts."""
    q_end = datetime(2023, 12, 31, tzinfo=UTC)
    store.put_records("fundamentals", [
        Record("NVDA", q_end, datetime(2024, 2, 1, tzinfo=UTC),
               {"metric": "cash", "value": 100}),
        Record("NVDA", q_end, datetime(2024, 5, 1, tzinfo=UTC),   # amendment
               {"metric": "cash", "value": 120}),
    ])
    before = store.get_records_asof("fundamentals", datetime(2024, 3, 1, tzinfo=UTC))
    assert len(before) == 1 and before.iloc[0]["payload"]["value"] == 100
    after = store.get_records_asof("fundamentals", datetime(2024, 6, 1, tzinfo=UTC))
    assert len(after) == 1 and after.iloc[0]["payload"]["value"] == 120
    # raw history keeps both versions (append-only)
    all_versions = store.get_records_asof(
        "fundamentals", datetime(2024, 6, 1, tzinfo=UTC), latest_version_only=False)
    assert len(all_versions) == 2


def test_view_is_frozen(store):
    store.put_bars("NVDA", "1h", _bars())
    view = store.view(datetime(2024, 1, 1, 16, 0, tzinfo=UTC))
    assert len(view.bars("NVDA", "1h")) == 2
    # the same view NEVER returns more, regardless of later writes
    store.put_bars("NVDA", "1h", _bars(20))
    assert len(view.bars("NVDA", "1h")) == 2
