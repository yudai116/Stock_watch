"""Phase 1 acceptance (b): delisted-ticker coverage audit.

For each name in config/universe.yaml ``delisted_audit_list``, query daily
bars over the past 10 years and report first/last bar and bar count.
Output: reports/coverage_audit.csv + console summary.

  # against the free Alpaca IEX feed (limited history, no delisted names):
  python -m data.coverage_audit
  # against the paid Polygon plan (full history + delisted — recommended, D6):
  python -m data.coverage_audit --source polygon
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data.config_loader import REPO_ROOT, load_config, load_dotenv_if_present

UTC = timezone.utc


def _fetch_bars_fn(source: str):
    if source == "polygon":
        from data.polygon_ingest import fetch_bars
    else:
        from data.alpaca_ingest import fetch_bars
    return fetch_bars


def run_audit(source: str = "alpaca") -> pd.DataFrame:
    load_dotenv_if_present()
    fetch_bars = _fetch_bars_fn(source)
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
    print(f"\nsource: {source} | delisted coverage: {n_covered}/{n_delisted} = {ratio:.0%}")
    if ratio < 0.8:
        if source == "polygon":
            print("=> Still INSUFFICIENT on Polygon — check plan entitlements/history limit.")
        else:
            print("=> INSUFFICIENT on Alpaca (expected: IEX has no delisted names). "
                  "Re-run with --source polygon (D6).")
    else:
        print("=> Sufficient for PIT universe construction.")
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["alpaca", "polygon"], default="alpaca")
    args = p.parse_args()
    run_audit(source=args.source)


if __name__ == "__main__":
    main()
