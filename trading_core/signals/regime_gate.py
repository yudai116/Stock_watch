"""Regime -> strategy allocation (SPEC §5.1, revised by SPEC_ADDENDUM_v2 R1).

Single-stock shorting is ABOLISHED. The bear-regime response is an index-CFD
hedge (signals/index_hedge.py) plus a stop on new longs.

  bull   -> trend_follow enabled (long)
  range  -> mean_revert enabled (GA/grid may disable), no hedge
  bear   -> no new longs, index hedge ON
  crisis -> no new entries, existing positions scaled down, hedge ON
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    allow_long_tf: bool
    allow_long_mr: bool
    hedge_on: bool
    scale_down_existing: bool


def gate(regime: str, mr_enabled: bool) -> GateDecision:
    if regime == "bull":
        return GateDecision(True, False, False, False)
    if regime == "range":
        return GateDecision(False, mr_enabled, False, False)
    if regime == "bear":
        return GateDecision(False, False, True, False)
    if regime == "crisis":
        return GateDecision(False, False, True, True)
    # unknown/warmup: stay out
    return GateDecision(False, False, False, False)
