"""Bear-regime short breakdown (SPEC §5.1, asymmetric constraints §6.4).

entry = regime gate(bear) AND Donchian low breakdown AND weak relative
strength (bottom quantile) AND volume confirmation.
"""

from __future__ import annotations

import pandas as pd


def entry_ok(row: pd.Series, rs_pct: float, params: dict) -> bool:
    if pd.isna(row["donchian_low_s"]) or pd.isna(row["volume_z"]) or pd.isna(rs_pct):
        return False
    breakdown = row["close"] < row["donchian_low_s"]
    rs_bottom = rs_pct <= float(params["rs_top_fraction"])   # symmetric bottom quantile
    volume_ok = row["volume_z"] >= float(params["volume_z_min"])
    return bool(breakdown and rs_bottom and volume_ok)
