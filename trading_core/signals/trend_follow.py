"""Long breakout entry (SPEC §5.2).

entry = regime gate AND Donchian(20-55) high break AND relative-strength
top-quantile AND volume confirmation. GA searches periods/thresholds only.

Pure function of the CURRENT feature row (already no-lookahead by
construction in features/hourly_features.py).
"""

from __future__ import annotations

import pandas as pd


def entry_ok(row: pd.Series, rs_pct: float, params: dict) -> bool:
    """``row``: latest hourly feature row; ``rs_pct``: cross-sectional RS
    percentile in [0,1] from the last completed DAY."""
    if pd.isna(row["donchian_high"]) or pd.isna(row["volume_z"]) or pd.isna(rs_pct):
        return False
    breakout = row["close"] > row["donchian_high"]
    rs_top = rs_pct >= 1.0 - float(params["rs_top_fraction"])
    volume_ok = row["volume_z"] >= float(params["volume_z_min"])
    return bool(breakout and rs_top and volume_ok)
