"""PIT split/dividend adjustment: a future split must be invisible at earlier as_of."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.adjuster import adjust_bars, adjusted_bars_asof, extract_actions
from data.bitemporal_store import BitemporalStore, Record

UTC = timezone.utc


def _bars():
    ts = pd.bdate_range("2024-01-01", periods=10, tz="UTC")
    px = [100.0] * 10
    return pd.DataFrame({"ts": ts, "open": px, "high": px, "low": px,
                         "close": px, "volume": [1000.0] * 10})


@pytest.fixture()
def store(tmp_path):
    s = BitemporalStore(tmp_path)
    s.put_bars("NVDA", "1d", _bars())
    ex = datetime(2024, 1, 8, 13, 30, tzinfo=UTC)   # split effective bar 5
    s.put_records("corporate_actions", [
        Record("NVDA", ex, ex, {"type": "split", "old_rate": 1, "new_rate": 4}),
    ])
    return s


def test_split_applied_backward():
    bars = _bars()
    actions = pd.DataFrame([{
        "event_ts": pd.Timestamp("2024-01-08 13:30", tz="UTC"),
        "type": "split", "ratio": 4.0, "cash": 0.0,
    }])
    adj = adjust_bars(bars, actions)
    assert adj["close"].iloc[0] == pytest.approx(25.0)     # pre-split / 4
    assert adj["volume"].iloc[0] == pytest.approx(4000.0)  # volume * 4
    assert adj["close"].iloc[-1] == pytest.approx(100.0)   # post-split unchanged


def test_dividend_total_return_adjustment():
    bars = _bars()
    actions = pd.DataFrame([{
        "event_ts": pd.Timestamp("2024-01-08 13:30", tz="UTC"),
        "type": "dividend", "ratio": 1.0, "cash": 2.0,
    }])
    adj = adjust_bars(bars, actions)
    assert adj["close"].iloc[0] == pytest.approx(98.0)     # * (1 - 2/100)
    assert adj["close"].iloc[-1] == pytest.approx(100.0)


def test_symbol_without_actions_when_store_has_others(store):
    """PRODUCTION regression: the corporate_actions dataset is non-empty
    (NVDA has a split) but the requested symbol has no actions at all —
    must return an empty frame, not KeyError('event_ts')."""
    store.put_bars("AMD", "1d", _bars())
    view = store.view(datetime(2024, 1, 12, 21, 0, tzinfo=UTC))
    actions = extract_actions(view, "AMD")
    assert actions.empty
    adj = adjusted_bars_asof(view, "AMD", "1d")
    assert (adj["close"] == 100.0).all()          # bars pass through unadjusted


def test_future_split_invisible_at_earlier_asof(store):
    # decision BEFORE the split ex-date: raw prices, no adjustment
    view_before = store.view(datetime(2024, 1, 5, 21, 0, tzinfo=UTC))
    adj_before = adjusted_bars_asof(view_before, "NVDA", "1d")
    assert (adj_before["close"] == 100.0).all()
    assert extract_actions(view_before, "NVDA").empty

    # decision AFTER: pre-split bars adjusted
    view_after = store.view(datetime(2024, 1, 12, 21, 0, tzinfo=UTC))
    adj_after = adjusted_bars_asof(view_after, "NVDA", "1d")
    pre = adj_after[adj_after["ts"] < pd.Timestamp("2024-01-08 13:30", tz="UTC")]
    post = adj_after[adj_after["ts"] >= pd.Timestamp("2024-01-08 13:30", tz="UTC")]
    assert np.allclose(pre["close"], 25.0)
    assert np.allclose(post["close"], 100.0)
