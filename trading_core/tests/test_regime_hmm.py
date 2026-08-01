"""Regime HMM: BIC state selection within configured range, dimension cap,
sane labelling on synthetic two-regime data, and walk-forward purity
(future perturbation must not change past labels).
"""

import numpy as np
import pandas as pd
import pytest

import features.regime_hmm as rh


def _two_regime_features(n=360, seed=0):
    """Calm bull (positive ret, low vol) then crisis (negative ret, high vol)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    ret = np.r_[rng.normal(0.001, 0.005, half), rng.normal(-0.003, 0.03, half)]
    vol = np.r_[rng.normal(0.10, 0.01, half), rng.normal(0.45, 0.05, half)]
    vix = np.r_[rng.normal(14, 1, half), rng.normal(38, 4, half)]
    breadth = np.r_[rng.normal(0.7, 0.05, half), rng.normal(0.25, 0.05, half)]
    idx = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    return pd.DataFrame({"market_return": ret, "realized_vol": vol,
                         "vix": vix, "breadth": breadth}, index=idx)


@pytest.fixture()
def fast_cfg(monkeypatch):
    """Shrink HMM config so walk-forward tests run in seconds."""
    from data.config_loader import load_config as real

    def patched(name):
        cfg = {k: v for k, v in real(name).items()}
        if name == "params":
            cfg = dict(cfg)
            cfg["hmm"] = dict(cfg["hmm"])
            cfg["hmm"]["n_states_range"] = [2, 3]
            cfg["hmm"]["min_train_days"] = 120
            cfg["hmm"]["covariance_type"] = "diag"
        return cfg

    monkeypatch.setattr(rh, "load_config", patched)
    return patched


def test_market_features_dimensions():
    """Baseline = 4 dims; with both alt-data series exactly 6 (SPEC §4 cap)."""
    n = 80
    idx = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    close = pd.Series(np.linspace(100, 110, n), index=idx)
    aux = pd.Series(0.5, index=idx)

    base = rh.build_market_features(close, aux, aux)
    assert list(base.columns) == ["market_return", "realized_vol", "vix", "breadth"]

    full = rh.build_market_features(close, aux, aux, sentiment_z=aux, neg_ratio=aux)
    assert full.shape[1] == 6


def test_bic_selects_within_range_and_labels_make_sense(fast_cfg):
    X = _two_regime_features()
    Z = ((X - X.mean()) / X.std()).to_numpy()
    model = rh.select_and_fit(Z, seed=3)
    assert 2 <= model.n_components <= 3
    states = model.predict(Z)
    labels = rh.label_states(model)
    first_label = labels[int(pd.Series(states[:100]).mode()[0])]
    second_label = labels[int(pd.Series(states[-100:]).mode()[0])]
    # calm half must not be labelled crisis; turbulent half must not be bull
    assert first_label != "crisis"
    assert second_label in ("crisis", "bear")


def test_walk_forward_labels_ignore_future(fast_cfg):
    X = _two_regime_features(300, seed=1)
    labels_a = rh.walk_forward_regimes(X, seed=5)

    X2 = X.copy()
    X2.iloc[200:] = X2.iloc[200:] * 5.0 + 3.0      # violent future perturbation
    labels_b = rh.walk_forward_regimes(X2, seed=5)

    cut = X.index[199]
    pd.testing.assert_series_equal(labels_a.loc[:cut], labels_b.loc[:cut])
