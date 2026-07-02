"""Market regime via Gaussian HMM (SPEC §4, §5.1).

Rules enforced here:
  * inputs are MARKET-LEVEL only, 4-6 dims total (params.yaml hmm.inputs_*)
  * the number of states is selected by BIC within n_states_range — never
    hardcoded
  * inference is walk-forward: the regime label for day t comes from a model
    fitted on data up to the last refit boundary <= t, decoded on the window
    ENDING at t (posterior of the last observation). No future data enters.

State labels: bull / range / bear / crisis, assigned from fitted state
moments (return mean x vol mean).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.config_loader import load_config

LABELS = ("bull", "range", "bear", "crisis")


def _fit_hmm(X: np.ndarray, n_states: int, covariance_type: str, seed: int = 7):
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                        n_iter=200, random_state=seed)
    model.fit(X)
    return model


def _bic(model, X: np.ndarray) -> float:
    n, d = X.shape
    k = model.n_components
    if model.covariance_type == "full":
        cov_params = k * d * (d + 1) / 2
    else:
        cov_params = k * d
    p = (k - 1) + k * (k - 1) + k * d + cov_params
    return -2 * model.score(X) + p * np.log(n)


def select_and_fit(X: np.ndarray, seed: int = 7):
    """Fit for each state count in the configured range; pick lowest BIC."""
    cfg = load_config("params")["hmm"]
    lo, hi = cfg["n_states_range"]
    best, best_bic = None, np.inf
    for k in range(int(lo), int(hi) + 1):
        try:
            m = _fit_hmm(X, k, cfg["covariance_type"], seed)
            b = _bic(m, X)
        except Exception:
            continue
        if b < best_bic:
            best, best_bic = m, b
    if best is None:
        raise RuntimeError("HMM fit failed for all state counts")
    return best


def label_states(model, ret_col: int = 0, vol_col: int = 1) -> dict[int, str]:
    """Map each hidden state to bull/range/bear/crisis by its moments."""
    means = model.means_
    k = means.shape[0]
    ret_m = means[:, ret_col]
    vol_m = means[:, vol_col]
    labels: dict[int, str] = {}
    crisis_state = int(np.argmax(vol_m))
    for s in range(k):
        if s == crisis_state and k >= 3:
            labels[s] = "crisis"
        elif ret_m[s] < np.percentile(ret_m, 34):
            labels[s] = "bear"
        elif ret_m[s] > np.percentile(ret_m, 67):
            labels[s] = "bull"
        else:
            labels[s] = "range"
    return labels


def build_market_features(market_close: pd.Series, vix: pd.Series,
                          breadth_series: pd.Series,
                          sentiment_z: pd.Series | None = None,
                          neg_ratio: pd.Series | None = None) -> pd.DataFrame:
    """Assemble the HMM input matrix. Baseline = 4 dims; alt-data adds <= 2.

    All inputs are rolling/point-in-time transforms of series indexed by day.
    """
    cfg = load_config("params")
    vol_days = cfg["features"]["realized_vol_days"]
    ret = np.log(market_close / market_close.shift(1))
    rv = ret.rolling(vol_days).std() * np.sqrt(252)
    X = pd.DataFrame({
        "market_return": ret,
        "realized_vol": rv,
        "vix": vix.reindex(ret.index).ffill(),
        "breadth": breadth_series.reindex(ret.index).ffill(),
    })
    if sentiment_z is not None:
        X["news_sentiment_z"] = sentiment_z.reindex(ret.index)
    if neg_ratio is not None:
        X["negative_article_ratio"] = neg_ratio.reindex(ret.index)
    if X.shape[1] > 6:
        raise ValueError("HMM inputs exceed 6 dims (SPEC §4)")
    return X.dropna()


def walk_forward_regimes(X: pd.DataFrame, seed: int = 7) -> pd.Series:
    """Regime label per day, walk-forward.

    Refit monthly (params.yaml hmm.refit_frequency); the label for day t is
    the argmax posterior of the LAST observation of X[:t] under the model
    fitted on X up to the most recent refit boundary before/at t.
    """
    cfg = load_config("params")["hmm"]
    min_train = int(cfg["min_train_days"])
    if len(X) <= min_train:
        raise ValueError(f"need > {min_train} rows, got {len(X)}")

    scaler_eps = 1e-12
    labels = pd.Series(index=X.index, dtype=object)
    refit_marks = pd.Series(X.index.tz_convert(None) if X.index.tz is not None
                            else X.index, index=X.index).dt.to_period("M")

    model = None
    state_names: dict[int, str] = {}
    mu = sd = None
    last_refit_period = None

    for i in range(min_train, len(X)):
        period = refit_marks.iloc[i]
        if model is None or period != last_refit_period:
            train = X.iloc[:i]
            mu, sd = train.mean(), train.std().replace(0, scaler_eps)
            Z = ((train - mu) / sd).to_numpy()
            model = select_and_fit(Z, seed)
            state_names = label_states(model)
            last_refit_period = period
        window = X.iloc[max(0, i - 252): i + 1]
        Zw = ((window - mu) / sd).to_numpy()
        post = model.predict_proba(Zw)[-1]
        labels.iloc[i] = state_names[int(np.argmax(post))]
    return labels.dropna()
