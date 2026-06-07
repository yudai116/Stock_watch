"""
regime/hmm_model.py — Gaussian HMM によるレジーム検出

設計方針:
  - 状態数は AIC / BIC で自動決定 (2〜6)。ハードコード禁止。
  - GaussianHMM (hmmlearn) を使用。共分散タイプは 'diag'（サンプル数 << 特徴量数²）
  - Viterbi で状態系列を復号し、Forward-Backward で事後確率を計算
  - 出力形式: {timestamp, regime_label, posterior_prob, dominant_prob}
  - レジームラベルは VIX 平均値で昇順ソートして番号付け (0=低ボラ, N-1=高ボラ)

依存: hmmlearn (pip install hmmlearn)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent.parent  # backtest/


def _count_hmm_params(n_components: int, n_features: int, cov_type: str = "diag") -> int:
    """HMM の自由パラメータ数を返す。BIC 計算に使用。"""
    # 遷移行列: n_components * (n_components - 1)
    # 初期分布: n_components - 1
    # 平均ベクトル: n_components * n_features
    # 分散 (diag): n_components * n_features
    trans   = n_components * (n_components - 1)
    init    = n_components - 1
    means   = n_components * n_features
    if cov_type == "diag":
        covars = n_components * n_features
    elif cov_type == "full":
        covars = n_components * n_features * (n_features + 1) // 2
    else:
        covars = n_components * n_features
    return trans + init + means + covars


def _reorder_by_vix_level(
    labels: np.ndarray,
    posteriors: np.ndarray,
    model,
    vix_feature_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    レジームラベルを VIX 関連特徴量の平均値で昇順に再番号付けする。
    (0=低ボラ/リスクオン, max=高ボラ/リスクオフ)
    """
    means = model.means_[:, vix_feature_idx]  # VIX 特徴量の各状態平均
    sort_idx = np.argsort(means)               # VIX 昇順でソート
    old_to_new = {int(old): int(new) for new, old in enumerate(sort_idx)}

    new_labels     = np.array([old_to_new[l] for l in labels])
    new_posteriors = posteriors[:, sort_idx]
    return new_labels, new_posteriors


class RegimeHMM:
    """
    Gaussian HMM ラッパー。AIC/BIC で最適状態数を選択して学習する。

    Parameters
    ----------
    min_states : int
        探索する最小状態数 (デフォルト 2)
    max_states : int
        探索する最大状態数 (デフォルト 6)
    criterion : str
        モデル選択基準 'bic' or 'aic'
    n_iter : int
        EM アルゴリズムの最大反復数
    random_state : int
        乱数シード
    """

    def __init__(
        self,
        min_states: int = 2,
        max_states: int = 6,
        criterion: str = "bic",
        n_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        self.min_states   = min_states
        self.max_states   = max_states
        self.criterion    = criterion
        self.n_iter       = n_iter
        self.random_state = random_state

        self.model_           = None
        self.n_components_    = None
        self.bic_history_: list[tuple[int, float]] = []
        self.aic_history_: list[tuple[int, float]] = []
        self._vix_feature_idx = 0

    # ── 学習 ─────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "RegimeHMM":
        """
        AIC または BIC で最適な状態数を選んで HMM を学習する。

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
            正規化済み観測変数行列 (NaN なし)
        """
        try:
            from hmmlearn import hmm as _hmm
        except ImportError:
            raise ImportError("hmmlearn が必要です: pip install hmmlearn")

        n_samples, n_features = X.shape
        cov_type = "diag"

        best_score = np.inf
        best_n     = self.min_states
        best_model = None

        print(f"[HMM] 状態数選択 ({self.criterion.upper()}) -- {self.min_states}~{self.max_states}")
        for n in range(self.min_states, self.max_states + 1):
            # 複数初期値で試行してベストを採用
            best_ll = -np.inf
            best_m  = None
            for seed in range(5):
                try:
                    m = _hmm.GaussianHMM(
                        n_components=n,
                        covariance_type=cov_type,
                        n_iter=self.n_iter,
                        random_state=self.random_state + seed,
                        tol=1e-4,
                    )
                    m.fit(X)
                    ll = m.score(X)
                    if ll > best_ll:
                        best_ll = ll
                        best_m  = m
                except Exception:
                    continue

            if best_m is None:
                continue

            n_params = _count_hmm_params(n, n_features, cov_type)
            ll       = best_ll
            bic      = -2 * ll + n_params * np.log(n_samples)
            aic      = -2 * ll + 2 * n_params
            self.bic_history_.append((n, round(bic, 2)))
            self.aic_history_.append((n, round(aic, 2)))

            score = bic if self.criterion == "bic" else aic
            print(f"  n={n}: ll={ll:.1f}  BIC={bic:.1f}  AIC={aic:.1f}")

            if score < best_score:
                best_score = score
                best_n     = n
                best_model = best_m

        self.n_components_ = best_n
        self.model_        = best_model
        print(f"[HMM] 最適状態数: {best_n}  ({self.criterion.upper()}={best_score:.1f})")
        return self

    # ── 予測 ─────────────────────────────────────────────────────────────────

    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Viterbi 復号 + Forward-Backward 事後確率を返す。
        レジームラベルは VIX 特徴量平均で昇順ソート済み。

        Returns
        -------
        labels : np.ndarray (T,), int
            Viterbi 状態系列
        posteriors : np.ndarray (T, n_states)
            各時刻の状態事後確率
        """
        if self.model_ is None:
            raise RuntimeError("先に fit() を呼んでください")

        labels     = self.model_.predict(X)         # Viterbi
        posteriors = self.model_.predict_proba(X)   # Forward-Backward

        labels, posteriors = _reorder_by_vix_level(
            labels, posteriors, self.model_, self._vix_feature_idx
        )
        return labels, posteriors

    def regime_signals(
        self,
        X: np.ndarray,
        timestamps: list[str],
    ) -> list[dict]:
        """
        レジーム信号 JSON レコードを生成する。

        Returns
        -------
        list of dict:
            {timestamp, regime_label, posterior_prob, dominant_prob}
        """
        labels, posteriors = self.predict(X)
        records = []
        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp":    ts,
                "regime_label": int(labels[i]),
                "posterior_prob": [round(float(p), 4) for p in posteriors[i]],
                "dominant_prob":  round(float(posteriors[i].max()), 4),
            })
        return records

    # ── 状態解釈ヘルパー ─────────────────────────────────────────────────────

    def summarize_states(self, feature_names: Optional[list[str]] = None) -> str:
        """学習した各状態の平均・標準偏差を表示する。"""
        if self.model_ is None:
            return "未学習"
        lines = [f"[HMM] 状態数={self.n_components_}"]
        for k in range(self.n_components_):
            mean  = self.model_.means_[k]
            std   = np.sqrt(np.diag(self.model_.covars_[k])) if hasattr(self.model_, "covars_") else np.array([])
            if feature_names and len(feature_names) == len(mean):
                desc = {n: f"{m:.3f}±{s:.3f}" for n, m, s in zip(feature_names, mean, std)}
                lines.append(f"  State {k}: {desc}")
            else:
                lines.append(f"  State {k}: mean={mean.round(3)}")
        return "\n".join(lines)

    # ── 保存/読み込み ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """モデルを JSON にシリアライズして保存する。"""
        if self.model_ is None:
            raise RuntimeError("未学習のモデルは保存できません")
        payload = {
            "n_components":   self.n_components_,
            "criterion":      self.criterion,
            "bic_history":    self.bic_history_,
            "aic_history":    self.aic_history_,
            "startprob":      self.model_.startprob_.tolist(),
            "transmat":       self.model_.transmat_.tolist(),
            "means":          self.model_.means_.tolist(),
            "covars":         self.model_.covars_.tolist(),
            "covariance_type": self.model_.covariance_type,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[HMM] モデル保存: {path}")

    @classmethod
    def load(cls, path: Path) -> "RegimeHMM":
        """保存済みモデルを読み込む。"""
        try:
            from hmmlearn import hmm as _hmm
        except ImportError:
            raise ImportError("hmmlearn が必要です: pip install hmmlearn")

        payload = json.loads(path.read_text())
        obj = cls()
        obj.n_components_ = payload["n_components"]
        obj.criterion     = payload["criterion"]
        obj.bic_history_  = payload.get("bic_history", [])
        obj.aic_history_  = payload.get("aic_history", [])

        n = payload["n_components"]
        m = _hmm.GaussianHMM(n_components=n, covariance_type=payload["covariance_type"])
        m.startprob_ = np.array(payload["startprob"])
        m.transmat_  = np.array(payload["transmat"])
        m.means_     = np.array(payload["means"])
        m.covars_    = np.array(payload["covars"])
        m.n_features = len(payload["means"][0])
        obj.model_   = m
        return obj


def run_regime_detection(
    macro_data: dict,
    output_path: Path,
    model_path: Optional[Path] = None,
    min_states: int = 2,
    max_states: int = 6,
    criterion: str = "bic",
    n_iter: int = 500,
    random_seed: int = 42,
) -> list[dict]:
    """
    マクロデータからレジーム信号を生成するメイン関数。

    Parameters
    ----------
    macro_data : dict
        fetch_macro.py の出力
    output_path : Path
        信号 JSON の保存先
    model_path : Path, optional
        学習済みモデルの保存先/読み込み元
    """
    from backtest.regime.features import build_features

    print("[regime] 特徴量構築中...")
    X, dates, feature_names = build_features(macro_data)

    model = RegimeHMM(
        min_states=min_states,
        max_states=max_states,
        criterion=criterion,
        n_iter=n_iter,
        random_state=random_seed,
    )
    model.fit(X)

    print(model.summarize_states(feature_names))

    if model_path is not None:
        model.save(model_path)

    signals = model.regime_signals(X, dates)

    output_path.write_text(json.dumps(signals, ensure_ascii=False, indent=2))
    n_states_count = {}
    for s in signals:
        k = s["regime_label"]
        n_states_count[k] = n_states_count.get(k, 0) + 1
    print(f"[regime] 信号生成完了: {output_path}  {len(signals)}日")
    for k in sorted(n_states_count):
        pct = 100 * n_states_count[k] / len(signals)
        print(f"  レジーム {k}: {n_states_count[k]}日 ({pct:.1f}%)")

    return signals


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent))
    from backtest.data.fetch_macro import load_macro_data
    from backtest.config import MACRO_DATA_PATH, REGIME_SIGNALS_PATH, HMM_MIN_STATES, HMM_MAX_STATES

    macro = load_macro_data(MACRO_DATA_PATH)
    signals = run_regime_detection(
        macro_data=macro,
        output_path=REGIME_SIGNALS_PATH,
        model_path=HERE / "regime_model.json",
        min_states=HMM_MIN_STATES,
        max_states=HMM_MAX_STATES,
    )
    print(f"\n最初の5レコード:")
    for s in signals[:5]:
        print(s)
