"""Walk-forward folds, CPCV purge/embargo, DSR, Monte Carlo, adoption rule."""

import numpy as np
import pandas as pd
import pytest

from validation.baseline_compare import compare
from validation.cpcv import cpcv_splits, n_paths
from validation.dsr import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe
from validation.monte_carlo import block_bootstrap, summarize
from validation.walk_forward import aggregate_oos, make_folds, wf_efficiency

IDX = pd.bdate_range("2016-01-01", periods=2000, tz="UTC")


# -------------------------------------------------------------------- WFA

def test_wfa_folds_are_anchored_and_disjoint():
    folds = make_folds(IDX)
    assert len(folds) == 4
    for i, f in enumerate(folds):
        assert f.train_start == IDX[0]                 # anchored
        assert f.train_end <= f.test_start             # no overlap
        if i > 0:
            assert f.test_start >= folds[i - 1].test_end  # OOS disjoint
    # every test window is strictly after its train window ends
    assert all(f.test_start >= f.train_end for f in folds)


def test_wf_efficiency():
    assert wf_efficiency(2.0, 1.2) == pytest.approx(0.6)
    assert wf_efficiency(0.0, 1.0) == 0.0


def test_aggregate_oos_takes_worst_dd():
    agg = aggregate_oos([{"sharpe": 1.0, "max_dd": -0.10},
                         {"sharpe": 2.0, "max_dd": -0.25}])
    assert agg["sharpe"] == pytest.approx(1.5)
    assert agg["max_dd"] == pytest.approx(-0.25)


# ------------------------------------------------------------------- CPCV

def test_cpcv_purge_and_embargo():
    splits = list(cpcv_splits(IDX, n_groups=6, n_test_groups=2,
                              purge_days=10, embargo_days=5))
    from math import comb
    assert len(splits) == comb(6, 2)
    purge = pd.Timedelta(days=10)
    embargo = pd.Timedelta(days=5)
    for train_mask, test_mask in splits:
        assert not (train_mask & test_mask).any()
        test_times = IDX[test_mask]
        train_times = IDX[train_mask]
        # no train obs within purge of any test obs (fast check via blocks)
        t0, t1 = test_times.min(), test_times.max()
        # find contiguous test blocks
        blocks = []
        cur = [test_times[0]]
        for a, b in zip(test_times[:-1], test_times[1:]):
            if (b - a) > pd.Timedelta(days=7):
                blocks.append((cur[0], cur[-1])); cur = [b]
            else:
                cur.append(b)
        blocks.append((cur[0], cur[-1]))
        for s, e in blocks:
            bad = (train_times >= s - purge) & (train_times <= e + purge + embargo)
            assert not bad.any()
    assert n_paths(6, 2) == 5


# -------------------------------------------------------------------- DSR

def test_expected_max_sharpe_grows_with_trials():
    v = 0.01
    assert expected_max_sharpe(1, v) == 0.0
    assert expected_max_sharpe(100, v) > expected_max_sharpe(10, v) > 0


def test_psr_hand_direction():
    # strong SR over many obs, normal-ish moments -> probability near 1
    assert probabilistic_sharpe(0.15, 0.0, 2000, 0.0, 3.0) > 0.99
    # SR below benchmark -> below 0.5
    assert probabilistic_sharpe(0.01, 0.05, 2000, 0.0, 3.0) < 0.5


def test_dsr_deflates_with_more_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 1500))       # real edge
    few = deflated_sharpe(r, n_trials=5)
    many = deflated_sharpe(r, n_trials=5000)
    assert few["dsr"] > many["dsr"]
    assert many["sr0"] > few["sr0"]


def test_dsr_rejects_pure_noise_mined_over_many_trials():
    rng = np.random.default_rng(2)
    # best of many noise strategies: SR ~ E[max] -> DSR should NOT be confident
    r = pd.Series(rng.normal(0.0002, 0.01, 1000))
    res = deflated_sharpe(r, n_trials=10000)
    assert res["dsr"] < 0.95


# ------------------------------------------------------------ Monte Carlo

def test_block_bootstrap_summary():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 1200))
    sims = block_bootstrap(r, n_sims=200, avg_block=20)
    s = summarize(sims, bars_per_year=252)
    assert s["sharpe_p05"] <= s["sharpe_p50"] <= s["sharpe_p95"]
    assert -1.0 <= s["max_dd_p95_worst"] <= 0.0
    assert 0.0 <= s["p_dd_breach_hard"] <= 1.0


# --------------------------------------------------------- adoption rule

def test_adoption_rule():
    base = {"calmar": 1.0, "max_dd": -0.15}
    good = {"calmar": 1.2, "max_dd": -0.14}            # +20%, DD better
    bad_dd = {"calmar": 1.5, "max_dd": -0.22}          # DD worse
    small = {"calmar": 1.05, "max_dd": -0.10}          # +5% only
    assert compare(base, good)["adopted"]
    assert not compare(base, bad_dd)["adopted"]
    assert not compare(base, small)["adopted"]
