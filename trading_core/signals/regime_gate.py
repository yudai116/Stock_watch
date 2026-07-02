"""Regime -> strategy allocation (SPEC §5.1).

  bull   -> trend_follow enabled (long)
  range  -> mean_revert enabled (GA may disable via mr_enabled)
  bear   -> short_breakdown enabled, NO new longs
  crisis -> no new entries, existing positions scaled down
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    allow_long_tf: bool
    allow_long_mr: bool
    allow_short: bool
    scale_down_existing: bool


def gate(regime: str, mr_enabled: bool) -> GateDecision:
    if regime == "bull":
        return GateDecision(True, False, False, False)
    if regime == "range":
        return GateDecision(False, mr_enabled, False, False)
    if regime == "bear":
        return GateDecision(False, False, True, False)
    if regime == "crisis":
        return GateDecision(False, False, False, True)
    # unknown/warmup: stay out
    return GateDecision(False, False, False, False)
