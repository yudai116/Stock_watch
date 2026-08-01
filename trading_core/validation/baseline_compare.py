"""Baseline comparisons (SPEC §8.3 D8, SPEC_ADDENDUM_v2 R5/F).

  * alt-data adoption rule: OOS Calmar +15% AND no DD worsening
  * QQQ mandatory gate (R5): a strategy must beat QQQ buy&hold on Sharpe
    AND have a SHALLOWER max drawdown, else it fails regardless of other
    numbers
  * branch classification (flow F): [A]/[B]/[C] from OOS Sharpe + gate
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


# ---------------------------------------------------------- R5: QQQ gate

def qqq_gate(strategy_oos: dict, qqq_oos: dict) -> dict:
    """Mandatory gate: beat QQQ B&H Sharpe AND have a shallower MaxDD,
    measured over IDENTICAL OOS windows. max_dd values are negative."""
    sharpe_ok = strategy_oos.get("sharpe", 0.0) > qqq_oos.get("sharpe", 0.0)
    dd_ok = strategy_oos.get("max_dd", -1.0) > qqq_oos.get("max_dd", -1.0)
    return {
        "passed": bool(sharpe_ok and dd_ok),
        "sharpe_ok": bool(sharpe_ok),
        "dd_ok": bool(dd_ok),
        "strategy_sharpe": strategy_oos.get("sharpe", 0.0),
        "qqq_sharpe": qqq_oos.get("sharpe", 0.0),
        "strategy_max_dd": strategy_oos.get("max_dd", 0.0),
        "qqq_max_dd": qqq_oos.get("max_dd", 0.0),
    }


# ------------------------------------------------------ flow F: branches

def classify_branch(oos_sharpe: float, gate_passed: bool) -> str:
    """[A] proceed to Phase 4; [B] simplification ladder; [C] pivot to
    weekly RS rotation. [D] is decided only after [C]'s variant also fails
    the gate — callers handle that transition."""
    cfg = load_config("params")["branching"]
    if oos_sharpe >= float(cfg["sharpe_a"]) and gate_passed:
        return "A"
    if oos_sharpe < float(cfg["sharpe_b_low"]):
        return "C"
    return "B"      # 0.4-0.8 band, or higher Sharpe with a near-miss gate fail


def overfit_alert(is_sharpe: float) -> bool:
    """R5: in-sample Sharpe above the alert level = overfitting alarm."""
    return is_sharpe > float(load_config("params")["targets"]["is_sharpe_overfit_alert"])
