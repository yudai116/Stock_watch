"""Hard gates (SPEC §5.4). Parameters are FIXED (params.yaml `fixed`), never
GA-searched.

  * earnings blackout: no new entries N business days before a scheduled report
  * news shock: strong negative aggregated sentiment -> block new entries in
    that symbol + tighten trailing stops
  * squeeze guard: no new SHORTS right after strongly positive news

All inputs must be point-in-time (earnings calendar known-at dates, news
usable-from bar alignment via altdata/latency_aligner.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.config_loader import load_config


@dataclass(frozen=True)
class GateState:
    blocked_new: bool           # no new entries at all
    blocked_short: bool         # no new shorts
    tighten_stop_factor: float  # 1.0 = unchanged


def earnings_blackout(next_earnings_ts: pd.Timestamp | None, now: pd.Timestamp) -> bool:
    """True if within the blackout window before a KNOWN upcoming report."""
    if next_earnings_ts is None or pd.isna(next_earnings_ts):
        return False
    n_days = int(load_config("params")["fixed"]["earnings_blackout_days"])
    bdays_left = np.busday_count(now.date(), next_earnings_ts.date())
    return 0 <= bdays_left <= n_days


def news_sentiment_window(scored_news: pd.DataFrame, now: pd.Timestamp,
                          hours: int = 24) -> float | None:
    """Mean sentiment of news usable at ``now`` within the trailing window.

    ``scored_news`` needs columns usable_from_ts, score.
    """
    if scored_news.empty:
        return None
    w = scored_news[(scored_news["usable_from_ts"] <= now)
                    & (scored_news["usable_from_ts"] > now - pd.Timedelta(hours=hours))]
    if w.empty:
        return None
    return float(w["score"].mean())


def evaluate(now: pd.Timestamp,
             next_earnings_ts: pd.Timestamp | None,
             scored_news: pd.DataFrame) -> GateState:
    cfg = load_config("params")["fixed"]
    blocked_new = earnings_blackout(next_earnings_ts, now)
    tighten = 1.0
    blocked_short = False

    senti = news_sentiment_window(scored_news, now)
    if senti is not None:
        if senti <= cfg["news_shock_neg_threshold"]:
            blocked_new = True
            tighten = float(cfg["news_shock_stop_tighten_factor"])
        if senti >= cfg["squeeze_guard_pos_threshold"]:
            blocked_short = True
    return GateState(blocked_new=blocked_new, blocked_short=blocked_short,
                     tighten_stop_factor=tighten)


NO_GATE = GateState(False, False, 1.0)
