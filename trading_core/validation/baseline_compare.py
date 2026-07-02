"""Chart-only baseline vs alt-data variant comparison (SPEC §8.3, D8).

Adoption rule (config params.yaml `adoption`): the alt-data family is
adopted ONLY if OOS Calmar improves by >= +15% AND max drawdown does not
worsen, both measured under identical WFA/CPCV settings.
"""

from __future__ import annotations

from data.config_loader import load_config


def compare(baseline_oos: dict, variant_oos: dict) -> dict:
    """Both inputs are aggregated OOS metric dicts (validation.walk_forward
    .aggregate_oos output: calmar, max_dd, sharpe, ...)."""
    cfg = load_config("params")["adoption"]
    min_impr = float(cfg["calmar_improvement_min"])

    base_calmar = baseline_oos.get("calmar", 0.0)
    var_calmar = variant_oos.get("calmar", 0.0)
    improvement = (var_calmar - base_calmar) / abs(base_calmar) if base_calmar else float("inf") if var_calmar > 0 else 0.0
    dd_ok = variant_oos.get("max_dd", -1.0) >= baseline_oos.get("max_dd", -1.0)  # DD is negative
    adopted = improvement >= min_impr and dd_ok
    return {
        "baseline_calmar": base_calmar,
        "variant_calmar": var_calmar,
        "calmar_improvement": improvement,
        "dd_not_worse": dd_ok,
        "adopted": adopted,
        "rule": f"calmar +{min_impr:.0%} and no DD worsening",
    }
