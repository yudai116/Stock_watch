"""Ablation protocol runner (SPEC §8.3, order is FIXED):

    fundamental_quality -> news_gates -> sentiment_hmm -> trends

Each family is evaluated against the current champion under identical
WFA/CPCV settings; adopted only per validation/baseline_compare rules.
Every run is recorded through optimize/trial_logger (DSR denominator).
"""

from __future__ import annotations

from typing import Callable

from data.config_loader import load_config
from optimize.trial_logger import TrialLogger
from validation.baseline_compare import compare


def run_ablation(
    evaluate: Callable[[set[str]], dict],
    logger: TrialLogger,
    run_id: str,
) -> dict:
    """``evaluate(enabled_families)`` -> aggregated OOS metrics dict.

    Starts from the chart-only baseline (empty set) and adds families in the
    configured order, keeping each only if adopted.
    """
    order = load_config("params")["adoption"]["ablation_order"]
    enabled: set[str] = set()
    baseline = evaluate(set(enabled))
    logger.log(run_id=run_id, phase="ablation_baseline", params={"families": []},
               metrics=baseline)
    report = {"baseline": baseline, "steps": []}

    champion = baseline
    for family in order:
        candidate_set = enabled | {family}
        candidate = evaluate(candidate_set)
        decision = compare(champion, candidate)
        logger.log(run_id=run_id, phase=f"ablation_{family}",
                   params={"families": sorted(candidate_set)},
                   metrics={**candidate, "adopted": decision["adopted"]})
        report["steps"].append({"family": family, "metrics": candidate,
                                "decision": decision})
        if decision["adopted"]:
            enabled.add(family)
            champion = candidate
    report["final_families"] = sorted(enabled)
    report["final_metrics"] = champion
    return report
