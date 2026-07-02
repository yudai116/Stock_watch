"""Phase 1 acceptance (b): delisted-ticker coverage audit against Alpaca.

For each name in config/universe.yaml ``delisted_audit_list``, query Alpaca
daily bars over the past 10 years and report first/last bar and bar count.
Output: reports/coverage_audit.csv + console summary with a D6 recommendation
(Polygon backfill) if coverage is insufficient.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY.
    python -m data.coverage_audit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data.alpaca_ingest import fetch_bars
from data.config_loader import REPO_ROOT, load_config, load_dotenv_if_present

UTC = timezone.utc


def run_audit() -> pd.DataFrame:
    load_dotenv_if_present()
    cfg = load_config("universe")
    targets = cfg["delisted_audit_list"]
    end = datetime.now(UTC)
    start = end - timedelta(days=int(10 * 365.25))

    rows = []
    for t in targets:
        sym = t["symbol"]
        try:
            bars = fetch_bars([sym], "1d", start, end).get(sym, pd.DataFrame())
        except Exception as e:  # symbol unknown to Alpaca etc.
            bars = pd.DataFrame()
            err = str(e)[:80]
        else:
            err = ""
        rows.append({
            "symbol": sym,
            "note": t["note"],
            "n_daily_bars": len(bars),
            "first_bar": bars["ts"].min() if len(bars) else None,
            "last_bar": bars["ts"].max() if len(bars) else None,
            "covered": len(bars) > 250,
            "error": err,
        })
    rep = pd.DataFrame(rows)

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    rep.to_csv(out_dir / "coverage_audit.csv", index=False)

    n_delisted = sum(1 for t in targets if "still listed" not in t["note"])
    n_covered = int(rep[~rep["note"].str.contains("still listed")]["covered"].sum())
    ratio = n_covered / n_delisted if n_delisted else 0.0
    print(rep.to_string())
    print(f"\ndelisted coverage: {n_covered}/{n_delisted} = {ratio:.0%}")
    if ratio < 0.8:
        print("=> INSUFFICIENT. Per D6, add Polygon (~$30/mo) for PIT universe backfill.")
    else:
        print("=> Sufficient for PIT universe construction.")
    return rep


if __name__ == "__main__":
    run_audit()
