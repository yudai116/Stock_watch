"""Range-regime mean reversion long (SPEC §5.1/§5.2 table).

entry = regime gate(range) AND oversold (RSI below GA threshold) AND price
z-score below GA threshold AND positive longer-term structure (MA slope not
collapsing). Exits share the common framework (chandelier/time stop).
"""

from __future__ import annotations

import pandas as pd


def entry_ok(row: pd.Series, params: dict) -> bool:
    if pd.isna(row["rsi"]) or pd.isna(row["price_z"]) or pd.isna(row["ma_slope"]):
        return False
    oversold = row["rsi"] <= float(params["mr_rsi_entry"])
    stretched = row["price_z"] <= float(params["mr_zscore_entry"])
    structure_ok = row["ma_slope"] > -0.02
    return bool(oversold and stretched and structure_ok)
