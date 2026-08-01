"""Confidence multiplier in the FIXED range 0.5-1.5 (SPEC §6.3).

multiplier = f(fundamental quality score, sentiment alignment with the
signal direction). Component WEIGHTS are the only ablation knob; the output
range never changes.
"""

from __future__ import annotations

from data.config_loader import load_config


def sentiment_alignment(signal_side: int, sector_sentiment: float | None) -> float:
    """[0,1]: 0.5 neutral; 1.0 = sentiment fully agrees with trade direction."""
    if sector_sentiment is None:
        return 0.5
    s = max(-1.0, min(1.0, sector_sentiment))
    return 0.5 + 0.5 * s * (1 if signal_side > 0 else -1)


def confidence(fund_quality: float | None, signal_side: int,
               sector_sentiment: float | None,
               w_fund: float = 1.0, w_senti: float = 1.0) -> float:
    """Blend components (each in [0,1]) then map to the fixed range."""
    cfg = load_config("params")["fixed"]
    lo, hi = float(cfg["confidence_multiplier_min"]), float(cfg["confidence_multiplier_max"])
    parts, weights = [], []
    if fund_quality is not None and w_fund > 0:
        parts.append(fund_quality); weights.append(w_fund)
    if w_senti > 0:
        parts.append(sentiment_alignment(signal_side, sector_sentiment)); weights.append(w_senti)
    if not parts:
        return 1.0
    blended = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
    return lo + (hi - lo) * blended
