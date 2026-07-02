"""Assign irregular events (news) to the bar from which they are usable.

Rule (SPEC §3.2): a news item with available_ts a becomes usable at the close
of the FIRST COMPLETE bar that starts at or after a. Equivalently, a decision
made at bar-close time c may use news with available_ts <= c - bar_duration
... no: usable_from_ts is the close of the first bar fully after a.

Example (1h bars): published 14:23, latency 5min -> available 14:28.
The 14:00-15:00 bar was already in progress, so the first complete bar is
15:00-16:00 and the item is usable from 16:00 (that bar's close/decision).
"""

from __future__ import annotations

import pandas as pd


def usable_from(available_ts: pd.Series, bar_duration: pd.Timedelta) -> pd.Series:
    """Close time of the first complete bar after ``available_ts``."""
    a = pd.to_datetime(available_ts, utc=True)
    next_bar_open = a.dt.ceil(bar_duration)
    # if exactly on a boundary, that bar itself is the next complete bar
    return next_bar_open + bar_duration


def align_news(news: pd.DataFrame, bar_duration: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
    """Add ``usable_from_ts`` to a news record frame (from store.records)."""
    if news.empty:
        out = news.copy()
        out["usable_from_ts"] = pd.Series(dtype="datetime64[ns, UTC]")
        return out
    out = news.copy()
    out["usable_from_ts"] = usable_from(out["available_ts"], bar_duration)
    return out


def news_usable_at(news: pd.DataFrame, decision_ts: pd.Timestamp,
                   bar_duration: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
    """News rows usable at a decision made at ``decision_ts`` (a bar close)."""
    aligned = align_news(news, bar_duration)
    return aligned[aligned["usable_from_ts"] <= pd.Timestamp(decision_ts).tz_convert("UTC")]
