"""Phase 3 runner (SPEC_ADDENDUM_v2 H): measure 3a and 3b under IDENTICAL
WFA settings, apply the QQQ mandatory gate + DSR, and classify the outcome
branch (flow F) in ONE report.

  3a: simple regime + weekly RS rotation (minimal form — the flow-[C] pivot,
      measured up front)
  3b: full composite (Donchian breakout + regime + vol targeting + hedge)

Usage (after daily bars are ingested; QQQ must be among the stored symbols):
    uv run python -m validation.phase3_runner            # both variants
    uv run python -m validation.phase3_runner --grid     # per-fold grid+plateau

Reported numbers are OOS-only, cost-deducted. IS values appear ONLY as the
overfitting alarm and WF-efficiency denominator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest.costs import CostModel
from backtest.engine import equity_metrics, run_backtest
from data.config_loader import REPO_ROOT, load_config
from features.daily_features import build_signal_frame, relative_strength
from features.simple_regime import simple_regime
from optimize import grid_runner
from optimize.param_space import merged_params
from optimize.trial_logger import TrialLogger
from signals.composite import SwingStrategy
from signals.index_hedge import rolling_beta
from signals.rs_rotation import RotationStrategy
from validation.baseline_compare import classify_branch, overfit_alert, qqq_gate
from validation.dsr import deflated_sharpe
from validation.walk_forward import aggregate_oos, make_folds, wf_efficiency

UTC = timezone.utc
BARS_PER_YEAR = 252


# ------------------------------------------------------------ input prep

def day_close_index(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Daily bars: close ts = open ts + 1 day (engine convention)."""
    return pd.DatetimeIndex(bars["ts"]) + pd.Timedelta(days=1)


def close_series(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(bars["close"].to_numpy(), index=day_close_index(bars))


def build_inputs(daily_bars: dict[str, pd.DataFrame],
                 vix: pd.Series | None = None,
                 benchmark: str = "QQQ",
                 features_params: dict | None = None) -> dict:
    """Common point-in-time inputs. ``features_params`` (composite only)
    triggers building the per-symbol daily signal frames."""
    if benchmark not in daily_bars:
        raise ValueError(f"benchmark {benchmark} bars required (hedge + gate)")
    closes = {s: close_series(b) for s, b in daily_bars.items()}
    bench_close = closes[benchmark]

    regime = simple_regime(bench_close, vix)

    cfg = load_config("params")["features"]
    rs_days = int(cfg["rs_rank_days"])
    rs = pd.DataFrame({s: relative_strength(c, bench_close, rs_days)
                       for s, c in closes.items() if s != benchmark})
    rs_rank = rs.rank(axis=1, pct=True)

    beta_win = int(load_config("params")["hedge"]["beta_window_days"])
    betas = {s: rolling_beta(c, bench_close, beta_win)
             for s, c in closes.items() if s != benchmark}

    features = ({s: build_signal_frame(b, features_params)
                 for s, b in daily_bars.items() if s != benchmark}
                if features_params is not None else {})
    return {"regime": regime, "rs_rank": rs_rank, "betas": betas,
            "features": features, "bench_close": bench_close,
            "benchmark": benchmark}


# --------------------------------------------------------- fold slicing

def slice_curve_metrics(curve: pd.Series, start, end) -> dict:
    seg = curve[(curve.index >= start) & (curve.index < end)]
    if len(seg) < 5:
        return {}
    return equity_metrics(seg, BARS_PER_YEAR)


def _run_once(daily_bars: dict[str, pd.DataFrame], strategy) -> pd.Series:
    result = run_backtest(daily_bars, strategy, initial_cash=100_000.0,
                          cost_model=CostModel.from_config(),
                          bar_duration=pd.Timedelta(days=1))
    return result.equity_curve


def make_strategy(variant: str, inputs: dict, params: dict):
    if variant == "3a":
        return RotationStrategy(regime=inputs["regime"], params=params,
                                benchmark_symbols=(inputs["benchmark"],))
    return SwingStrategy(features=inputs["features"], regime=inputs["regime"],
                         rs_rank=inputs["rs_rank"], params=params,
                         betas=inputs["betas"])


def default_params(variant: str) -> dict:
    """Grid mid-point as the fixed baseline parameter set."""
    grid = grid_runner.grid_for("rotation" if variant == "3a" else "composite")
    mid = {k: v[len(v) // 2] for k, v in grid.items()}
    return merged_params(mid) if variant == "3b" else mid


# ------------------------------------------------------------- evaluation

def evaluate_variant(variant: str, daily_bars: dict[str, pd.DataFrame],
                     vix: pd.Series | None, use_grid: bool,
                     logger: TrialLogger, run_id: str) -> dict:
    base_params = default_params(variant)
    inputs = build_inputs(daily_bars, vix,
                          features_params=base_params if variant == "3b" else None)
    all_closes = sorted(set().union(
        *[set(day_close_index(b)) for b in daily_bars.values()]))
    folds = make_folds(pd.DatetimeIndex(all_closes))

    fold_oos, fold_is, qqq_oos_folds, oos_returns = [], [], [], []
    chosen_params_per_fold = []

    for f in folds:
        if use_grid:
            grid_name = "rotation" if variant == "3a" else "composite"

            def eval_train(p):
                pp = merged_params(p) if variant == "3b" else p
                inp = (build_inputs(daily_bars, vix, features_params=pp)
                       if variant == "3b" else inputs)
                curve = _run_once(daily_bars, make_strategy(variant, inp, pp))
                m = slice_curve_metrics(curve, f.train_start, f.train_end)
                return m or {"calmar": -9.0, "max_dd": -1.0, "sharpe": -9.0}

            sel = grid_runner.run_grid(eval_train, grid_name,
                                       run_id=f"{run_id}-f{f.fold}", logger=logger)
            params = (merged_params(sel["best_params"]) if variant == "3b"
                      else sel["best_params"])
        else:
            params = base_params
            logger.log(run_id=run_id, phase=f"baseline_{variant}",
                       params={k: v for k, v in params.items()}, metrics={})

        inp = (inputs if variant == "3a" or params == base_params
               else build_inputs(daily_bars, vix, features_params=params))
        curve = _run_once(daily_bars, make_strategy(variant, inp, params))
        m_oos = slice_curve_metrics(curve, f.test_start, f.test_end)
        m_is = slice_curve_metrics(curve, f.train_start, f.train_end)
        if m_oos:
            fold_oos.append(m_oos)
            seg = curve[(curve.index >= f.test_start) & (curve.index < f.test_end)]
            oos_returns.append(seg.pct_change().dropna())
        if m_is:
            fold_is.append(m_is)
        chosen_params_per_fold.append(params)

        # QQQ benchmark over the SAME OOS window
        bench = inputs["bench_close"]
        b_seg = bench[(bench.index >= f.test_start) & (bench.index < f.test_end)]
        if len(b_seg) >= 5:
            qqq_oos_folds.append(equity_metrics(b_seg, BARS_PER_YEAR))

    oos = aggregate_oos(fold_oos)
    is_agg = aggregate_oos(fold_is)
    qqq_oos = aggregate_oos(qqq_oos_folds)
    gate = qqq_gate(oos, qqq_oos)
    oos_r = pd.concat(oos_returns) if oos_returns else pd.Series(dtype=float)
    dsr = deflated_sharpe(oos_r, n_trials=max(1, logger.count_trials()))
    return {
        "variant": variant,
        "params_per_fold": chosen_params_per_fold,
        "oos": oos, "is": is_agg, "qqq_oos": qqq_oos,
        "gate": gate,
        "dsr": dsr,
        "wf_efficiency": wf_efficiency(is_agg.get("sharpe", 0.0),
                                       oos.get("sharpe", 0.0)),
        "overfit_alert": overfit_alert(is_agg.get("sharpe", 0.0)),
        "branch": classify_branch(oos.get("sharpe", 0.0), gate["passed"]),
    }


def run_phase3(daily_bars: dict[str, pd.DataFrame],
               vix: pd.Series | None = None,
               use_grid: bool = False,
               variants: tuple = ("3a", "3b")) -> dict:
    logger = TrialLogger()
    run_id = f"phase3-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    report = {"run_id": run_id, "use_grid": use_grid, "variants": {}}
    for v in variants:
        report["variants"][v] = evaluate_variant(
            v, daily_bars, vix, use_grid, logger, run_id)
    return report


# --------------------------------------------------------------- report

def render_report(report: dict) -> str:
    t = load_config("params")["targets"]
    lines = [f"# Phase 3 Report — {report['run_id']}",
             f"grid: {report['use_grid']}", ""]
    for v, r in report["variants"].items():
        oos, gate, dsr = r["oos"], r["gate"], r["dsr"]
        lines += [
            f"## Variant {v}",
            f"| metric | OOS | pass line |",
            f"|---|---|---|",
            f"| Sharpe | {oos.get('sharpe', 0):.2f} | >= {t['sharpe_min']} |",
            f"| CAGR | {oos.get('cagr', 0):.1%} | >= {t['cagr_min']:.0%} |",
            f"| MaxDD | {oos.get('max_dd', 0):.1%} | >= -{t['max_dd_pct']}% |",
            f"| Calmar | {oos.get('calmar', 0):.2f} | - |",
            f"| WF efficiency | {r['wf_efficiency']:.2f} | >= {t['wf_efficiency_min']} |",
            f"| DSR | {dsr['dsr']:.3f} (N={dsr['n_trials']}) | > 0.95 |",
            "",
            f"QQQ gate: **{'PASS' if gate['passed'] else 'FAIL'}** "
            f"(strategy Sharpe {gate['strategy_sharpe']:.2f} vs QQQ {gate['qqq_sharpe']:.2f}; "
            f"MaxDD {gate['strategy_max_dd']:.1%} vs {gate['qqq_max_dd']:.1%})",
            f"Overfit alert (IS Sharpe): {'YES' if r['overfit_alert'] else 'no'}",
            f"**Branch: [{r['branch']}]**", "",
        ]
    return "\n".join(lines)


def main() -> None:
    from data.bitemporal_store import BitemporalStore
    from data.config_loader import data_root
    from data.adjuster import adjusted_bars_asof
    from data.vix_ingest import vix_series_asof

    p = argparse.ArgumentParser()
    p.add_argument("--grid", action="store_true",
                   help="per-fold grid + plateau selection (slow)")
    p.add_argument("--variants", default="3a,3b")
    args = p.parse_args()

    store = BitemporalStore(data_root())
    as_of = datetime.now(UTC)
    view = store.view(as_of)
    bars = {}
    for sym in store.list_bar_symbols("1d"):
        b = adjusted_bars_asof(view, sym, "1d")
        if len(b) > 300:
            bars[sym] = b
    if "QQQ" not in bars:
        raise SystemExit("QQQ daily bars are required (ingest them first)")
    vix = vix_series_asof(view)
    report = run_phase3(bars, vix if len(vix) else None, use_grid=args.grid,
                        variants=tuple(args.variants.split(",")))
    text = render_report(report)
    out = REPO_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "phase3_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
