"""Phase 1 acceptance (c): Alpaca News article-count audit per year.

Counts articles per calendar year for a sample of universe symbols to reveal
history depth / coverage cliffs (Alpaca news starts ~2015, thin early years).
Output: reports/news_audit.csv.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY.
    python -m data.news_audit --symbols NVDA,AMD,MSFT,AAPL,MU
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from data.config_loader import REPO_ROOT, load_dotenv_if_present
from data.news_ingest import fetch_news

UTC = timezone.utc


def run_audit(symbols: list[str], start_year: int = 2015) -> pd.DataFrame:
    load_dotenv_if_present()
    rows = []
    now = datetime.now(UTC)
    for year in range(start_year, now.year + 1):
        y0 = datetime(year, 1, 1, tzinfo=UTC)
        y1 = min(datetime(year + 1, 1, 1, tzinfo=UTC), now)
        recs = fetch_news(symbols, y0, y1)
        counts = (pd.Series([r.payload["symbol"] for r in recs]).value_counts().to_dict()
                  if recs else {})
        rows.append({"year": year, "total": len(recs), **{s: counts.get(s, 0) for s in symbols}})
        print(rows[-1])
    rep = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    rep.to_csv(out_dir / "news_audit.csv", index=False)
    return rep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="NVDA,AMD,MSFT,AAPL,MU")
    p.add_argument("--start-year", type=int, default=2015)
    args = p.parse_args()
    run_audit([s.strip().upper() for s in args.symbols.split(",")], args.start_year)
