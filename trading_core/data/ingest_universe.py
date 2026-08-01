"""One-shot candidate-pool ingestion: every seed ticker in universe.yaml
(incl. delisted) + benchmarks, via Polygon.

    python -m data.ingest_universe --years 10 --timeframes 1d

Names Polygon cannot serve are reported at the end — they simply stay out of
the pool (the PIT rebalance only ever sees stored bars).
"""

from __future__ import annotations

import argparse

from data.config_loader import load_config


def universe_candidates() -> list[str]:
    """All seed tickers + audit delisted names + benchmarks, deduplicated."""
    cfg = load_config("universe")
    symbols: set[str] = set()
    for spec in cfg["sectors"].values():
        symbols.update(spec.get("seed") or [])
    symbols.update(t["symbol"] for t in cfg.get("delisted_audit_list", []))
    symbols.update(str(v) for v in cfg["benchmarks"].values())
    return sorted(symbols)


def main() -> None:
    from data.polygon_ingest import ingest

    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--timeframes", default="1d")
    args = p.parse_args()
    symbols = universe_candidates()
    print(f"candidate pool: {len(symbols)} tickers")
    ingest(symbols, args.years,
           timeframes=tuple(t.strip() for t in args.timeframes.split(",")))


if __name__ == "__main__":
    main()
