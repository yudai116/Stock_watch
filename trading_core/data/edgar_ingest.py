"""EDGAR 10-K/10-Q ingestion via the SEC XBRL companyfacts API.

Bitemporal mapping (SPEC §3.2):
  event_ts     = fiscal period end
  available_ts = filing ACCEPTANCE datetime; if accepted after the close,
                 the next business day's market open.
Amendments (10-K/A, 10-Q/A) are appended as new versions — never overwritten.

Usage:
    python -m data.edgar_ingest --symbols NVDA,AMD
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from data.bitemporal_store import BitemporalStore, Record
from data.config_loader import data_root, load_dotenv_if_present

UTC = timezone.utc
SEC_BASE = "https://data.sec.gov"

# XBRL us-gaap tags used by altdata/fundamental_quality.py
FACT_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
    "total_assets": ["Assets"],
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "inventory": ["InventoryNet"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
}

MARKET_CLOSE_UTC_HOUR = 21   # 16:00 ET standard time buffer
MARKET_OPEN_UTC = (14, 30)


def acceptance_to_available(accepted: pd.Timestamp) -> pd.Timestamp:
    """Filing accepted after the close becomes usable at next business day open."""
    accepted = accepted.tz_convert("UTC")
    open_h, open_m = MARKET_OPEN_UTC
    same_day_open = accepted.normalize() + pd.Timedelta(hours=open_h, minutes=open_m)
    close = accepted.normalize() + pd.Timedelta(hours=MARKET_CLOSE_UTC_HOUR)
    if accepted <= same_day_open:
        candidate = same_day_open
    elif accepted < close:
        candidate = accepted  # intraday acceptance: usable immediately
    else:
        candidate = same_day_open + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:  # roll weekends
        candidate += pd.Timedelta(days=1)
    return candidate


def _headers() -> dict[str, str]:
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError("EDGAR_USER_AGENT must be set in .env (SEC requirement)")
    return {"User-Agent": ua, "Accept-Encoding": "gzip"}


def get_cik_map(client: httpx.Client) -> dict[str, str]:
    r = client.get("https://www.sec.gov/files/company_tickers.json", headers=_headers())
    r.raise_for_status()
    return {v["ticker"].upper(): f"{int(v['cik_str']):010d}" for v in r.json().values()}


def fetch_company_facts(client: httpx.Client, cik: str) -> dict:
    r = client.get(f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json", headers=_headers())
    r.raise_for_status()
    return r.json()


def fetch_filing_acceptance(client: httpx.Client, cik: str) -> dict[str, pd.Timestamp]:
    """accession number -> acceptance datetime, from the submissions API."""
    r = client.get(f"{SEC_BASE}/submissions/CIK{cik}.json", headers=_headers())
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    out = {}
    for accn, acc_dt in zip(recent["accessionNumber"], recent["acceptanceDateTime"]):
        out[accn] = pd.Timestamp(acc_dt).tz_convert("UTC") if pd.Timestamp(acc_dt).tzinfo else pd.Timestamp(acc_dt, tz="UTC")
    return out


def facts_to_records(symbol: str, facts: dict, acceptance: dict[str, pd.Timestamp]) -> list[Record]:
    records: list[Record] = []
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for canonical, tags in FACT_TAGS.items():
        for tag in tags:
            units = gaap.get(tag, {}).get("units", {})
            for vals in units.values():
                for v in vals:
                    form = v.get("form", "")
                    if not form.startswith(("10-K", "10-Q")):
                        continue
                    end = v.get("end")
                    accn = v.get("accn")
                    if not end or accn is None:
                        continue
                    accepted = acceptance.get(accn)
                    if accepted is None:
                        # fall back to filed date + 1 business day (conservative)
                        filed = pd.Timestamp(v.get("filed"), tz="UTC")
                        accepted = filed + pd.Timedelta(days=1)
                    available = acceptance_to_available(accepted)
                    records.append(
                        Record(
                            # key identifies the fact: revisions (10-K/A) of the
                            # same (symbol, metric, period) supersede bitemporally
                            key=f"{symbol}|{canonical}",
                            event_ts=pd.Timestamp(end, tz="UTC"),
                            available_ts=available,
                            payload={
                                "symbol": symbol,
                                "metric": canonical,
                                "tag": tag,
                                "value": v.get("val"),
                                "form": form,
                                "accn": accn,
                            },
                        )
                    )
                break  # first tag with data wins
    return records


def ingest(symbols: list[str], store: BitemporalStore | None = None) -> None:
    load_dotenv_if_present()
    store = store or BitemporalStore(data_root())
    with httpx.Client(timeout=30) as client:
        cik_map = get_cik_map(client)
        for sym in symbols:
            cik = cik_map.get(sym)
            if cik is None:
                print(f"WARN no CIK for {sym} (delisted? needs manual mapping)")
                continue
            facts = fetch_company_facts(client, cik)
            acceptance = fetch_filing_acceptance(client, cik)
            recs = facts_to_records(sym, facts, acceptance)
            store.put_records("fundamentals", recs)
            print(f"{sym}: {len(recs)} fundamental facts")
            time.sleep(0.15)  # SEC rate limit: max 10 req/s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True)
    args = p.parse_args()
    ingest([s.strip().upper() for s in args.symbols.split(",")])


if __name__ == "__main__":
    main()
