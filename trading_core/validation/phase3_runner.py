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
# benchmark/hedge instruments: used for regime, RS ranking and hedging,
# NEVER tradable as stock candidates
BENCHMARKS = ("QQQ", "SMH")


# ------------------------------------------------------------ input prep

def day_close_index(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Daily bars: close ts = open ts + 1 day (engine convention)."""
    return pd.DatetimeIndex(bars["ts"]) + pd.Timedelta(days=1)


def close_series(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(bars["close"].to_numpy(), index=day_close_index(bars))


def build_inputs(daily_bars: dict[str, pd.DataFrame],
                 vix: pd.Series | None = None,
                 benchmark: str = "QQQ",
                 features_params: dict | None = None,
                 regime_series: pd.Series | None = None) -> dict:
    """Common point-in-time inputs. ``features_params`` (composite only)
    triggers building the per-symbol daily signal frames.
    ``regime_series`` overrides the default simple filter (R4: used to run
    the HMM head-to-head against the simple regime under identical settings)."""
    if benchmark not in daily_bars:
        raise ValueError(f"benchmark {benchmark} bars required (hedge + gate)")
    closes = {s: close_series(b) for s, b in daily_bars.items()}
    bench_close = closes[benchmark]

    regime = (regime_series if regime_series is not None
              else simple_regime(bench_close, vix))

    tradable = [s for s in closes if s != benchmark and s not in BENCHMARKS]

    cfg = load_config("params")["features"]
    rs_days = int(cfg["rs_rank_days"])
    rs = pd.DataFrame({s: relative_strength(closes[s], bench_close, rs_days)
                       for s in tradable})
    rs_rank = rs.rank(axis=1, pct=True)

    beta_win = int(load_config("params")["hedge"]["beta_window_days"])
    betas = {s: rolling_beta(closes[s], bench_close, beta_win) for s in tradable}

    features = ({s: build_signal_frame(daily_bars[s], features_params)
                 for s in tradable}
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


def make_strategy(variant: str, inputs: dict, params: dict,
                  universe_provider=None):
    if variant == "3a":
        return RotationStrategy(regime=inputs["regime"], params=params,
                                benchmark_symbols=BENCHMARKS,
                                universe_provider=universe_provider)
    if variant == "3d":
        from signals.regime_overlay import RegimeOverlayStrategy
        return RegimeOverlayStrategy(regime=inputs["regime"],
                                     symbol=inputs["benchmark"])
    return SwingStrategy(features=inputs["features"], regime=inputs["regime"],
                         rs_rank=inputs["rs_rank"], params=params,
                         betas=inputs["betas"],
                         universe_provider=universe_provider)


def default_params(variant: str) -> dict:
    """Grid mid-point as the fixed baseline parameter set."""
    if variant == "3d":
        return {}                       # flow-F [D]: zero searched parameters
    grid = grid_runner.grid_for("rotation" if variant == "3a" else "composite")
    mid = {k: v[len(v) // 2] for k, v in grid.items()}
    return merged_params(mid) if variant == "3b" else mid


# ------------------------------------------------------------- evaluation

def hmm_regime_series(daily_bars: dict[str, pd.DataFrame],
                      vix: pd.Series | None, benchmark: str = "QQQ") -> pd.Series:
    """R4: walk-forward HMM regimes from market-level inputs only (no news).
    Heavy (monthly refits over the sample); computed once per run."""
    from features.daily_features import breadth as breadth_fn
    from features.regime_hmm import build_market_features, walk_forward_regimes
    from features.simple_regime import realized_vol_proxy

    closes = {s: close_series(b) for s, b in daily_bars.items()}
    bench_close = closes[benchmark]
    tradable = {s: c for s, c in closes.items()
                if s != benchmark and s not in BENCHMARKS}
    br = breadth_fn(tradable, 50)
    vol = (vix.sort_index().reindex(bench_close.index, method="ffill")
           if vix is not None else realized_vol_proxy(bench_close))
    X = build_market_features(bench_close, vol, br)
    return walk_forward_regimes(X)


def evaluate_variant(variant: str, daily_bars: dict[str, pd.DataFrame],
                     vix: pd.Series | None, use_grid: bool,
                     logger: TrialLogger, run_id: str,
                     universe_provider=None,
                     regime_series: pd.Series | None = None,
                     composite_grid: str = "composite") -> dict:
    base_params = default_params(variant)
    inputs = build_inputs(daily_bars, vix,
                          features_params=base_params if variant == "3b" else None,
                          regime_series=regime_series)
    all_closes = pd.DatetimeIndex(sorted(set().union(
        *[set(day_close_index(b)) for b in daily_bars.values()])))
    # SPEC §8.3-5: the most recent holdout_years are UNTOUCHED by every WFA
    # fold / grid selection; they are evaluated exactly once, after all
    # ablations, via evaluate_holdout().
    holdout_years = float(load_config("params")["validation"]["holdout_years"])
    holdout_start = all_closes.max() - pd.Timedelta(days=int(holdout_years * 365.25))
    folds = make_folds(all_closes[all_closes < holdout_start])

    fold_oos, fold_is, qqq_oos_folds, oos_returns = [], [], [], []
    chosen_params_per_fold = []

    # Full-period equity curves are fold-independent (folds only SLICE them),
    # and composite features depend only on the period parameters — cache
    # both so a grid costs one run per unique combo, not per combo per fold.
    curve_cache: dict[tuple, pd.Series] = {}
    inputs_cache: dict[tuple, dict] = {}

    def _inputs_for(pp: dict) -> dict:
        if variant != "3b":
            return inputs
        key = (pp.get("donchian_entry_period"), pp.get("atr_period"),
               pp.get("mr_rsi_period"))
        if key not in inputs_cache:
            inputs_cache[key] = build_inputs(daily_bars, vix, features_params=pp,
                                             regime_series=regime_series)
        return inputs_cache[key]

    def _curve_for(pp: dict) -> pd.Series:
        key = tuple(sorted((k, str(v)) for k, v in pp.items()))
        if key not in curve_cache:
            curve_cache[key] = _run_once(
                daily_bars, make_strategy(variant, _inputs_for(pp), pp,
                                          universe_provider))
        return curve_cache[key]

    for f in folds:
        if use_grid and variant != "3d":     # 3d has nothing to search
            grid_name = "rotation" if variant == "3a" else composite_grid

            def eval_train(p):
                pp = merged_params(p) if variant == "3b" else p
                m = slice_curve_metrics(_curve_for(pp), f.train_start, f.train_end)
                return m or {"calmar": -9.0, "max_dd": -1.0, "sharpe": -9.0}

            sel = grid_runner.run_grid(eval_train, grid_name,
                                       run_id=f"{run_id}-f{f.fold}", logger=logger)
            params = (merged_params(sel["best_params"]) if variant == "3b"
                      else sel["best_params"])
        else:
            params = base_params
            logger.log(run_id=run_id, phase=f"baseline_{variant}",
                       params={k: v for k, v in params.items()}, metrics={})

        curve = _curve_for(params)
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
        "holdout_start": holdout_start,
        "folds": [(str(f.test_start.date()), str(f.test_end.date())) for f in folds],
    }


def evaluate_holdout(variant: str, daily_bars: dict[str, pd.DataFrame],
                     vix: pd.Series | None, params: dict) -> dict:
    """FINAL one-shot holdout evaluation (SPEC §8.3-5). Run ONCE, after all
    ablations are settled — never used for any selection decision."""
    p = merged_params(params) if variant == "3b" else params
    inputs = build_inputs(daily_bars, vix,
                          features_params=p if variant == "3b" else None)
    all_closes = pd.DatetimeIndex(sorted(set().union(
        *[set(day_close_index(b)) for b in daily_bars.values()])))
    holdout_years = float(load_config("params")["validation"]["holdout_years"])
    holdout_start = all_closes.max() - pd.Timedelta(days=int(holdout_years * 365.25))
    curve = _run_once(daily_bars, make_strategy(variant, inputs, p))
    m = slice_curve_metrics(curve, holdout_start, all_closes.max() + pd.Timedelta(days=1))
    bench = inputs["bench_close"]
    b_seg = bench[bench.index >= holdout_start]
    qqq_m = equity_metrics(b_seg, BARS_PER_YEAR) if len(b_seg) >= 5 else {}
    return {"variant": variant, "holdout_start": holdout_start,
            "metrics": m, "qqq": qqq_m,
            "gate": qqq_gate(m, qqq_m) if m and qqq_m else {"passed": False}}


def run_phase3(daily_bars: dict[str, pd.DataFrame],
               vix: pd.Series | None = None,
               use_grid: bool = False,
               variants: tuple = ("3a", "3b"),
               universe_provider=None,
               regime: str = "simple",
               composite_grid: str = "composite") -> dict:
    logger = TrialLogger()
    run_id = f"phase3-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    universe_mode = (f"PIT quarterly ({universe_provider.n_quarters} quarters)"
                     if universe_provider is not None
                     else "STATIC all-stored symbols (SURVIVORSHIP-BIASED)")
    regime_series = None
    if regime == "hmm":
        print("fitting walk-forward HMM regimes (R4)... this takes a while")
        regime_series = hmm_regime_series(daily_bars, vix)
    report = {"run_id": run_id, "use_grid": use_grid,
              "universe_mode": universe_mode, "regime_mode": regime,
              "composite_grid": composite_grid,
              "variants": {}}
    for v in variants:
        report["variants"][v] = evaluate_variant(
            v, daily_bars, vix, use_grid, logger, run_id,
            universe_provider=universe_provider, regime_series=regime_series,
            composite_grid=composite_grid)
    return report


# --------------------------------------------------------------- report

def render_report(report: dict) -> str:
    t = load_config("params")["targets"]
    lines = [f"# Phase 3 Report — {report['run_id']}",
             f"grid: {report['use_grid']}",
             f"universe: {report.get('universe_mode', 'unknown')}",
             f"regime: {report.get('regime_mode', 'simple')}", ""]
    if "SURVIVORSHIP" in report.get("universe_mode", ""):
        lines += ["> **WARNING**: static universe of currently-listed names — "
                  "results are UPWARD-BIASED. Build the PIT universe "
                  "(--rebuild-universe) before trusting these numbers.", ""]
    for v, r in report["variants"].items():
        oos, gate, dsr = r["oos"], r["gate"], r["dsr"]
        lines += [
            f"## Variant {v}",
            f"holdout reserved from: {str(r['holdout_start'].date())} "
            f"(untouched; final one-shot eval only)",
            f"OOS folds: {r['folds']}",
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
            _passline_verdict(oos, r, dsr, t),
            f"**Branch: [{r['branch']}]**", "",
        ]
    return "\n".join(lines)


def _passline_verdict(oos: dict, r: dict, dsr: dict, t: dict) -> str:
    """One-line R5 pass-line summary so a branch label never hides a miss
    (e.g. branch [A] on Sharpe+gate while CAGR is still below minimum)."""
    checks = {
        "sharpe": oos.get("sharpe", 0) >= t["sharpe_min"],
        "cagr": oos.get("cagr", 0) >= t["cagr_min"],
        "max_dd": oos.get("max_dd", -1) >= -t["max_dd_pct"] / 100.0,
        "wf_eff": r["wf_efficiency"] >= t["wf_efficiency_min"],
        "dsr@95": dsr["dsr"] >= 0.95,
    }
    marks = " / ".join(f"{k} {'OK' if v else 'NG'}" for k, v in checks.items())
    return f"pass lines (R5): {marks}"


def main() -> None:
    from data.bitemporal_store import BitemporalStore
    from data.config_loader import data_root
    from data.adjuster import adjusted_bars_asof
    from data.pit_universe import membership_provider, rebuild_all
    from data.vix_ingest import vix_series_asof

    p = argparse.ArgumentParser()
    p.add_argument("--grid", action="store_true",
                   help="per-fold grid + plateau selection (slow)")
    p.add_argument("--variants", default="3a,3b")
    p.add_argument("--rebuild-universe", action="store_true",
                   help="(re)build quarterly PIT universe records from stored "
                        "daily bars before measuring")
    p.add_argument("--static-universe", action="store_true",
                   help="explicitly accept a survivorship-biased static "
                        "universe (results flagged in the report)")
    p.add_argument("--regime", choices=["simple", "hmm"], default="simple",
                   help="R4: run with the simple filter (default) or the "
                        "walk-forward HMM to compare head-to-head")
    p.add_argument("--composite-grid", choices=["composite", "composite_b1"],
                   default="composite_b1",
                   help="grid for 3b: composite_b1 (flow-F ladder B1, 81 combos,"
                        " default) or the full composite grid (486 combos, slow)")
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

    if args.rebuild_universe:
        first = min(b["ts"].min() for b in bars.values())
        rebuild_all(store, first, as_of)
    provider = membership_provider(store)
    if provider is None and not args.static_universe:
        raise SystemExit(
            "No PIT universe in the store. Run with --rebuild-universe "
            "(recommended; ingest delisted names first for full coverage), "
            "or pass --static-universe to knowingly accept survivorship bias.")

    vix = vix_series_asof(view)
    report = run_phase3(bars, vix if len(vix) else None, use_grid=args.grid,
                        variants=tuple(args.variants.split(",")),
                        universe_provider=provider, regime=args.regime,
                        composite_grid=args.composite_grid)
    text = render_report(report)
    out = REPO_ROOT / "reports"
    out.mkdir(exist_ok=True)
    suffix = "" if args.regime == "simple" else f"_{args.regime}"
    grid_suffix = "_grid" if args.grid else ""
    (out / f"phase3_report{suffix}{grid_suffix}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
