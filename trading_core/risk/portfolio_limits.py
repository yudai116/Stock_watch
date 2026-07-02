"""Portfolio-level constraints (SPEC §6.2): concurrent positions, sector
concentration, portfolio heat. Parameters live in params.yaml `fixed`.
"""

from __future__ import annotations

from data.config_loader import load_config


def can_open(open_positions: dict, sector_by_symbol: dict[str, str],
             candidate_symbol: str, candidate_risk_pct: float,
             open_risk_pct_total: float) -> tuple[bool, str]:
    """Check all portfolio limits for a prospective entry.

    ``open_risk_pct_total``: sum of open positions' remaining risk (distance
    to stop) as % of equity. Returns (allowed, reason_if_blocked).
    """
    cfg = load_config("params")["fixed"]
    if candidate_symbol in open_positions:
        return False, "already_open"
    if len(open_positions) >= int(cfg["max_concurrent_positions"]):
        return False, "max_positions"
    sector = sector_by_symbol.get(candidate_symbol, "other")
    n_same = sum(1 for s in open_positions if sector_by_symbol.get(s, "other") == sector)
    if n_same >= int(cfg["max_sector_positions"]):
        return False, "sector_concentration"
    if open_risk_pct_total + candidate_risk_pct > float(cfg["portfolio_heat_max_pct"]):
        return False, "portfolio_heat"
    return True, ""
