"""Phase 3 runner smoke test on synthetic data: both variants measured under
identical WFA settings, QQQ gate + DSR + branch present in the report."""

import numpy as np
import pandas as pd
import pytest

from validation.phase3_runner import (build_inputs, close_series,
                                      default_params, render_report,
                                      run_phase3)

N = 620


def _bars(drift, seed, start=100.0, n=N):
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + rng.normal(drift, 0.010, n))
    ts = pd.bdate_range("2021-01-04", periods=n, tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": np.r_[close[0], close[:-1]],
        "high": close * 1.012, "low": close * 0.988,
        "close": close, "volume": rng.integers(1000, 9000, n).astype(float),
    })


@pytest.fixture(scope="module")
def panel():
    bars = {
        "AAA": _bars(0.0022, 1), "BBB": _bars(0.0012, 2),
        "CCC": _bars(0.0004, 3), "DDD": _bars(-0.0004, 4),
        "QQQ": _bars(0.0009, 9, start=300.0),
    }
    vix = pd.Series(16.0, index=close_series(bars["QQQ"]).index)
    return bars, vix


def test_run_phase3_produces_full_report(panel):
    bars, vix = panel
    report = run_phase3(bars, vix, use_grid=False, variants=("3a", "3b"))
    for v in ("3a", "3b"):
        r = report["variants"][v]
        assert {"sharpe", "max_dd", "cagr", "calmar"} <= set(r["oos"])
        assert "passed" in r["gate"]
        assert 0.0 <= r["dsr"]["dsr"] <= 1.0
        assert r["branch"] in ("A", "B", "C")
        assert isinstance(r["overfit_alert"], bool)
        assert r["qqq_oos"].get("sharpe") is not None
    text = render_report(report)
    assert "QQQ gate" in text and "Branch" in text


def test_default_params_are_grid_midpoints():
    p3a = default_params("3a")
    assert set(p3a) == {"lookback_days", "n_holdings", "rebalance_days"}
    p3b = default_params("3b")
    assert "donchian_entry_period" in p3b        # grid param
    assert "earnings_blackout_days" in p3b       # fixed merged in


def test_build_inputs_requires_benchmark(panel):
    bars, vix = panel
    with pytest.raises(ValueError, match="benchmark"):
        build_inputs({k: v for k, v in bars.items() if k != "QQQ"}, vix)


def test_holdout_reserved_from_folds(panel):
    """SPEC §8.3-5: no WFA fold may touch the holdout window."""
    bars, vix = panel
    report = run_phase3(bars, vix, use_grid=False, variants=("3a",))
    r = report["variants"]["3a"]
    holdout_start = pd.Timestamp(r["holdout_start"])
    for _start, end in r["folds"]:
        assert pd.Timestamp(end, tz="UTC") <= holdout_start


def test_benchmarks_never_tradable(panel):
    """QQQ/SMH are regime/hedge instruments, not stock candidates."""
    bars, vix = panel
    bars = dict(bars)
    bars["SMH"] = _bars(0.0010, 7, start=200.0)
    inputs = build_inputs(bars, vix, features_params={
        "atr_period": 14, "donchian_entry_period": 20, "mr_rsi_period": 5})
    assert "QQQ" not in inputs["features"] and "SMH" not in inputs["features"]
    assert "QQQ" not in inputs["rs_rank"].columns
    assert "SMH" not in inputs["rs_rank"].columns


def test_holdout_one_shot_eval(panel):
    from validation.phase3_runner import default_params, evaluate_holdout

    bars, vix = panel
    out = evaluate_holdout("3a", bars, vix, default_params("3a"))
    assert "passed" in out["gate"]
    assert out["metrics"].get("sharpe") is not None
    assert out["qqq"].get("sharpe") is not None
