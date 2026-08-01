"""Fundamental quality (PIT via AsOfView) + sentiment cache keying."""

from datetime import datetime, timezone

import pytest

from altdata.fundamental_quality import fundamental_quality
from data.bitemporal_store import BitemporalStore, Record

UTC = timezone.utc


def _quarterly(store, symbol, metric, values, first_year=2022):
    """Four quarters; each available ~40 days after quarter end."""
    recs = []
    for i, v in enumerate(values):
        q_end = datetime(first_year + (i // 4), 3 * (i % 4) + 3, 28, tzinfo=UTC)
        avail = datetime(first_year + (i // 4), 3 * (i % 4) + 3, 28, tzinfo=UTC)
        avail = avail.replace(month=min(12, avail.month + 1), day=28)
        recs.append(Record(f"{symbol}|{metric}", q_end, avail,
                           {"symbol": symbol, "metric": metric, "value": float(v)}))
    store.put_records("fundamentals", recs)


@pytest.fixture()
def store(tmp_path):
    s = BitemporalStore(tmp_path)
    _quarterly(s, "GOOD", "cash", [100, 120, 150, 200])
    _quarterly(s, "GOOD", "total_assets", [1000, 1000, 1000, 1000])
    _quarterly(s, "GOOD", "total_debt", [500, 450, 400, 300])       # falling
    _quarterly(s, "BAD", "cash", [200, 150, 100, 50])
    _quarterly(s, "BAD", "total_assets", [1000, 1000, 1000, 1000])
    _quarterly(s, "BAD", "total_debt", [300, 400, 500, 700])        # rising
    return s


def test_quality_separates_good_from_bad(store):
    as_of = datetime(2023, 6, 1, tzinfo=UTC)
    good = fundamental_quality(store.view(as_of), "GOOD", is_semiconductor=False)
    bad = fundamental_quality(store.view(as_of), "BAD", is_semiconductor=False)
    assert good > 0.5 > bad


def test_quality_neutral_without_data(store):
    as_of = datetime(2023, 6, 1, tzinfo=UTC)
    assert fundamental_quality(store.view(as_of), "UNKNOWN", False) == 0.5


def test_quality_is_pit(store):
    """Before filings become available, the score must be neutral."""
    early = datetime(2022, 2, 1, tzinfo=UTC)     # before first availability
    assert fundamental_quality(store.view(early), "GOOD", False) == 0.5


def test_sentiment_cache_keys_on_model_version(tmp_path, monkeypatch):
    import altdata.sentiment_scorer as ss

    cache = ss.SentimentCache(tmp_path)
    cache.put("art1", 0.8)
    assert cache.get("art1") == 0.8
    # different pinned revision -> cache miss (reproducibility guard)
    cache.model_revision = "other-revision"
    assert cache.get("art1") is None
    assert ss.article_id({"id": "abc"}) == "abc"
    h1 = ss.article_id({"headline": "NVDA beats"})
    assert h1 == ss.article_id({"headline": "NVDA beats"})
