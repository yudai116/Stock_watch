"""
risk/position_sizer.py — リスク管理・ポジションサイジング

【Layer 3: リスク層の役割】
  1. ボラティリティターゲティング
     - 目標年率ボラ = 15% を維持するようポジションサイズを調整
     - 各銘柄の N 日ボラ (日次リターンの標準偏差) を使用

  2. HMM レジームフィルター
     - レジーム信号を読み込み、高ボラレジーム (デフォルト: レジーム 2, 3) では取引停止
     - HMM はエントリー信号ではなく「ゲートキーパー」として機能

  3. ポートフォリオレベル上限
     - 1 銘柄当たりの上限: 口座資産の 20%
     - 同時保有銘柄: 最大 10 銘柄

  4. Kelly 基準 (オプション)
     - f* = (edge / odds) のフル Kelly を 0.5 倍に抑制 (Half-Kelly)
     - 負の f* は取引スキップ
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent.parent  # backtest/


def realized_vol(
    returns: np.ndarray,
    window: int = 20,
    annualize: bool = True,
    bars_per_year: int = 252,
) -> float:
    """
    直近 window 本のリターンから実現ボラティリティを計算する。

    Parameters
    ----------
    returns : np.ndarray
        日次 (または各足) リターン系列
    window : int
        ボラ計算ウィンドウ
    annualize : bool
        True の場合、年率換算して返す
    bars_per_year : int

    Returns
    -------
    float : 実現ボラティリティ
    """
    if len(returns) < 2:
        return np.nan
    seg  = returns[-window:] if len(returns) >= window else returns
    seg  = seg[~np.isnan(seg)]
    if len(seg) < 2:
        return np.nan
    vol  = float(np.std(seg))
    if annualize:
        vol *= np.sqrt(bars_per_year)
    return vol


def vol_target_size(
    target_vol: float,
    current_vol: float,
    capital: float,
    price: float,
    max_position_frac: float = 0.20,
) -> int:
    """
    ボラティリティターゲティングによる株数を計算する。

    Parameters
    ----------
    target_vol : float
        目標年率ボラティリティ (例: 0.15 = 15%)
    current_vol : float
        現在の銘柄年率ボラティリティ
    capital : float
        口座資産
    price : float
        エントリー価格
    max_position_frac : float
        1 銘柄の口座比率上限

    Returns
    -------
    int : 購入株数 (整数, 0 以上)
    """
    if current_vol <= 1e-10 or price <= 0:
        return 0
    # ターゲットボラに合わせた名目資産の割合
    vol_scale   = target_vol / current_vol
    max_dollars = capital * min(vol_scale, max_position_frac)
    shares      = int(max_dollars / price)
    return max(shares, 0)


def half_kelly_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    price: float,
    kelly_fraction: float = 0.5,
    max_position_frac: float = 0.20,
) -> int:
    """
    Half-Kelly によるポジションサイズを計算する。

    Parameters
    ----------
    win_rate : float
        勝率 (0〜1)
    avg_win : float
        平均利益率 (正の値, 例: 0.05 = +5%)
    avg_loss : float
        平均損失率 (正の値, 例: 0.03 = 3%)
    capital : float
        口座資産
    price : float
        エントリー価格
    kelly_fraction : float
        Kelly 乗数 (デフォルト 0.5 = Half-Kelly)
    max_position_frac : float
        上限

    Returns
    -------
    int : 購入株数
    """
    if avg_loss <= 1e-10 or avg_win <= 1e-10:
        return 0
    edge  = win_rate * avg_win - (1 - win_rate) * avg_loss
    odds  = avg_win / avg_loss
    f_raw = edge / (odds * avg_loss) if avg_loss > 0 else 0.
    f     = max(0., min(f_raw * kelly_fraction, max_position_frac))
    return int(capital * f / price)


class RegimeFilter:
    """
    HMM レジーム信号を使って取引可否を判断するフィルター。

    Parameters
    ----------
    signals_path : Path
        regime_signals.json のパス
    stop_regimes : set[int]
        このレジームでは取引停止 (デフォルト: {2, 3} = 高ボラ)
    """

    def __init__(
        self,
        signals_path: Path,
        stop_regimes: Optional[set] = None,
    ) -> None:
        self.stop_regimes = stop_regimes or {2, 3}
        self._signals: dict[str, dict] = {}

        if signals_path.exists():
            raw = json.loads(signals_path.read_text())
            for rec in raw:
                ts = rec["timestamp"][:10]  # YYYY-MM-DD
                self._signals[ts] = rec
            print(f"[RegimeFilter] {len(self._signals)} 日分の信号を読み込みました")
        else:
            print(f"[RegimeFilter] 信号ファイルなし: {signals_path}  (全日取引許可)")

    def is_tradeable(self, date_str: str) -> bool:
        """
        指定日が取引可能かどうかを返す。
        信号がない日は保守的に取引許可 (True) を返す。
        """
        rec = self._signals.get(date_str[:10])
        if rec is None:
            return True
        label = rec.get("regime_label", 0)
        return label not in self.stop_regimes

    def regime_on(self, date_str: str) -> int:
        """レジームラベルを返す (不明: -1)"""
        rec = self._signals.get(date_str[:10])
        return rec.get("regime_label", -1) if rec else -1


class PositionSizer:
    """
    ボラターゲティング + HMM フィルターを統合したポジションサイジング。

    Parameters
    ----------
    target_vol : float
        目標年率ボラ (デフォルト: 0.15)
    max_position_frac : float
        1 銘柄の口座比率上限 (デフォルト: 0.20)
    max_positions : int
        同時最大保有銘柄数 (デフォルト: 10)
    regime_filter : RegimeFilter, optional
        HMM フィルター
    method : str
        "vol_target" or "half_kelly"
    bars_per_year : int
        年間バー数
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        max_position_frac: float = 0.20,
        max_positions: int = 10,
        regime_filter: Optional[RegimeFilter] = None,
        method: str = "vol_target",
        bars_per_year: int = 252,
    ) -> None:
        self.target_vol        = target_vol
        self.max_position_frac = max_position_frac
        self.max_positions     = max_positions
        self.regime_filter     = regime_filter
        self.method            = method
        self.bars_per_year     = bars_per_year

    def size(
        self,
        ticker: str,
        date_str: str,
        returns_history: np.ndarray,
        capital: float,
        price: float,
        win_rate: float = 0.5,
        avg_win: float = 0.05,
        avg_loss: float = 0.03,
    ) -> int:
        """
        指定銘柄・日付のポジションサイズ (株数) を返す。

        HMM フィルターが取引停止を示す場合は 0 を返す。
        """
        # HMM フィルター
        if self.regime_filter and not self.regime_filter.is_tradeable(date_str):
            return 0

        cur_vol = realized_vol(
            returns_history, window=20,
            annualize=True, bars_per_year=self.bars_per_year,
        )
        if np.isnan(cur_vol) or cur_vol <= 0:
            return 0

        if self.method == "vol_target":
            shares = vol_target_size(
                self.target_vol, cur_vol, capital, price, self.max_position_frac,
            )
        elif self.method == "half_kelly":
            shares = half_kelly_size(
                win_rate, avg_win, avg_loss, capital, price,
                kelly_fraction=0.5, max_position_frac=self.max_position_frac,
            )
        else:
            raise ValueError(f"method={self.method} は未知です")

        return shares
