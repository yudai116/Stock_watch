"""Trial logger (DSR denominator) + GA runner smoke test + ablation order."""

import random

import pytest

from optimize import param_space
from optimize.ga_runner import objective_value, run_ga
from optimize.trial_logger import TrialLogger
from validation.ablation_runner import run_ablation


def test_param_space_sampling_within_bounds():
    rng = random.Random(0)
    for _ in range(50):
        ind = param_space.sample(rng)
        assert 20 <= ind["donchian_entry_period"] <= 55
        assert 2.5 <= ind["chandelier_k"] <= 3.5
        assert 0.5 <= ind["risk_per_trade_pct"] <= 1.0
        m = param_space.mutate(ind, rng, p=1.0)
        assert 20 <= m["donchian_entry_period"] <= 55
        assert 2.5 <= m["chandelier_k"] <= 3.5


def test_merged_params_contains_fixed():
    rng = random.Random(1)
    p = param_space.merged_params(param_space.sample(rng))
    assert "earnings_blackout_days" in p        # fixed, not searchable
    assert "chandelier_k" in p


def test_objective_penalizes_deep_dd():
    good = objective_value({"calmar": 1.0, "max_dd": -0.15})
    bad = objective_value({"calmar": 1.0, "max_dd": -0.30})
    assert good == pytest.approx(1.0)
    assert bad < good


def test_trial_logger_counts_everything(tmp_path):
    log = TrialLogger(tmp_path)
    for i in range(7):
        log.log("run1", "ga", {"x": i}, {"sharpe_per_bar": 0.01 * i}, generation=0)
    log.log("run2", "manual_tweak", {"threshold": 0.5}, {})
    assert log.count_trials("run1") == 7
    assert log.count_trials() == 8
    assert log.sharpe_variance("run1") > 0


def test_ga_runner_logs_all_individuals(tmp_path):
    log = TrialLogger(tmp_path)

    def evaluate(params):
        # toy landscape: prefer high donchian period, shallow dd
        return {"calmar": params["donchian_entry_period"] / 55.0, "max_dd": -0.1}

    out = run_ga(evaluate, n_generations=3, pop_size=8, seed=0,
                 run_id="ga-test", logger=log)
    assert out["n_trials"] == 3 * 8                     # every individual logged
    assert out["best_params"]["donchian_entry_period"] >= 40
    assert out["best_metrics"]["calmar"] > 0.7


def test_ablation_runner_respects_order_and_rule(tmp_path):
    log = TrialLogger(tmp_path)
    # fundamental_quality helps a lot; news_gates hurts; others no-op
    effect = {"fundamental_quality": 0.30, "news_gates": -0.10,
              "sentiment_hmm": 0.05, "trends": 0.0}

    def evaluate(families):
        calmar = 1.0 + sum(effect[f] for f in families)
        return {"calmar": calmar, "max_dd": -0.12, "sharpe": 1.0}

    report = run_ablation(evaluate, log, run_id="abl-test")
    assert report["final_families"] == ["fundamental_quality"]
    steps = [s["family"] for s in report["steps"]]
    assert steps == ["fundamental_quality", "news_gates", "sentiment_hmm", "trends"]
    assert log.count_trials("abl-test") == 5            # baseline + 4 families
