"""Mandatory no-lookahead tests for the signals layer + latency alignment."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from altdata.latency_aligner import align_news, news_usable_at, usable_from
from signals import gates
from signals.regime_gate import gate

UTC = timezone.utc


# ------------------------------------------------------------ latency rule

def test_news_usable_from_next_complete_bar():
    """SPEC §3.2 example: published 14:23 (+5m latency) -> usable from 16:00."""
    avail = pd.Series([pd.Timestamp("2024-01-02 14:28", tz="UTC")])
    out = usable_from(avail, pd.Timedelta(hours=1))
    assert out.iloc[0] == pd.Timestamp("2024-01-02 16:00", tz="UTC")


def test_news_on_bar_boundary():
    """Available exactly at 15:00 -> the 15:00-16:00 bar is the next complete
    bar -> usable from its close 16:00."""
    avail = pd.Series([pd.Timestamp("2024-01-02 15:00", tz="UTC")])
    out = usable_from(avail, pd.Timedelta(hours=1))
    assert out.iloc[0] == pd.Timestamp("2024-01-02 16:00", tz="UTC")


def test_news_usable_at_decision_time():
    news = pd.DataFrame({
        "available_ts": [pd.Timestamp("2024-01-02 14:28", tz="UTC"),
                         pd.Timestamp("2024-01-02 15:10", tz="UTC")],
        "score": [-0.9, 0.8],
    })
    at_16 = news_usable_at(news, pd.Timestamp("2024-01-02 16:00", tz="UTC"))
    assert len(at_16) == 1 and at_16.iloc[0]["score"] == -0.9
    at_17 = news_usable_at(news, pd.Timestamp("2024-01-02 17:00", tz="UTC"))
    assert len(at_17) == 2


# ------------------------------------------------------------------ gates

def test_earnings_blackout_window():
    now = pd.Timestamp("2024-03-04 16:00", tz="UTC")     # Monday
    # earnings on Thursday = 3 business days ahead -> blocked (N=3)
    assert gates.earnings_blackout(pd.Timestamp("2024-03-07", tz="UTC"), now)
    # earnings far away -> not blocked
    assert not gates.earnings_blackout(pd.Timestamp("2024-03-20", tz="UTC"), now)
    # unknown -> not blocked
    assert not gates.earnings_blackout(None, now)
    # already past -> not blocked
    assert not gates.earnings_blackout(pd.Timestamp("2024-03-01", tz="UTC"), now)


def test_news_shock_gate_uses_only_usable_news():
    """A future negative article must NOT trigger the gate at an earlier decision."""
    scored = align_news(pd.DataFrame({
        "available_ts": [pd.Timestamp("2024-01-02 18:30", tz="UTC")],
        "score": [-0.95],
    }))
    early = gates.evaluate(pd.Timestamp("2024-01-02 18:00", tz="UTC"), None, scored)
    assert not early.blocked_new                       # not usable yet
    late = gates.evaluate(pd.Timestamp("2024-01-02 20:00", tz="UTC"), None, scored)
    assert late.blocked_new                            # usable from 20:00
    assert late.tighten_stop_factor < 1.0


def test_squeeze_guard_blocks_shorts_only():
    scored = align_news(pd.DataFrame({
        "available_ts": [pd.Timestamp("2024-01-02 12:00", tz="UTC")],
        "score": [0.9],
    }))
    st = gates.evaluate(pd.Timestamp("2024-01-02 15:00", tz="UTC"), None, scored)
    assert st.blocked_short and not st.blocked_new


# ------------------------------------------------------------ regime gate

def test_regime_gate_matrix():
    """v2 (R1): single-stock shorting abolished — bear/crisis switch on the
    index hedge instead."""
    assert gate("bull", True).allow_long_tf
    assert not gate("bull", True).hedge_on
    assert gate("range", True).allow_long_mr
    assert not gate("range", False).allow_long_mr      # grid disabled MR
    bear = gate("bear", True)
    assert bear.hedge_on and not bear.allow_long_tf and not bear.allow_long_mr
    crisis = gate("crisis", True)
    assert crisis.scale_down_existing and crisis.hedge_on
    assert not (crisis.allow_long_tf or crisis.allow_long_mr)
    unknown = gate("warmup", True)
    assert not (unknown.allow_long_tf or unknown.allow_long_mr or unknown.hedge_on)
