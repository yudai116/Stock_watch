"""Grid + plateau selection (SPEC_ADDENDUM_v2 R3): plateau beats sharp peak,
6-parameter hard limit, full trial logging."""

import pytest

import optimize.grid_runner as gr
from optimize.trial_logger import TrialLogger


def test_plateau_rejected_sharp_peak():
    """A lone spike must lose to a stable plateau in the top decile."""
    grid = {"x": [1, 2, 3, 4, 5], "y": [1, 2, 3]}
    def obj(p):
        if p == {"x": 5, "y": 3}:
            return 10.0                     # isolated spike (neighbours ~5)
        if p["x"] == 2 and p["y"] == 2:
            return 8.0                      # plateau centre
        if abs(p["x"] - 2) + abs(p["y"] - 2) == 1:
            return 7.9                      # smooth shoulder
        return 5.0
    results = [(p, obj(p)) for p in gr.enumerate_grid(grid)]
    best, best_obj, info = gr.plateau_select(results, grid, top_fraction=0.10)
    assert best == {"x": 2, "y": 2}
    assert best_obj == 8.0
    assert info["neighbourhood_var"] < 0.01


def test_enumerate_grid_size():
    grid = {"a": [1, 2, 3], "b": [10, 20]}
    combos = gr.enumerate_grid(grid)
    assert len(combos) == 6
    assert {"a": 3, "b": 10} in combos


def test_six_param_hard_limit(monkeypatch):
    fake = {"grid_search": {"fat": {f"p{i}": [1, 2] for i in range(7)}}}
    monkeypatch.setattr(gr, "load_config", lambda name: fake)
    with pytest.raises(ValueError, match="R3 limit"):
        gr.grid_for("fat")


def test_run_grid_logs_every_trial(tmp_path):
    log = TrialLogger(tmp_path)

    def evaluate(params):
        # toy landscape favouring long lookback, mild penalty on n_holdings
        return {"calmar": params["lookback_days"] / 189.0
                - 0.01 * params["n_holdings"],
                "max_dd": -0.10, "sharpe": 1.0}

    out = gr.run_grid(evaluate, "rotation", run_id="grid-test", logger=log)
    # rotation grid: 3 lookbacks x 3 holdings x 2 cadences = 18 combos
    assert out["n_trials"] == 18
    assert log.count_trials("grid-test") == 18
    assert out["best_params"]["lookback_days"] in (63, 126, 189)
    assert set(out["best_params"]) == {"lookback_days", "n_holdings",
                                       "rebalance_days"}
    assert "neighbourhood_var" in out["selection_info"]


def test_real_config_grids_respect_limit():
    for name in ("rotation", "composite"):
        grid = gr.grid_for(name)
        assert 1 <= len(grid) <= gr.MAX_PARAMS
        for levels in grid.values():
            assert 2 <= len(levels) <= 5      # coarse grid: 3-5 levels (2 for binary)
