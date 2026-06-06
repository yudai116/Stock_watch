"""
strategy/indicators.py — テクニカル指標スコア関数

各関数は numpy 配列を受け取り、[0, 25] 範囲のスコア配列を返す。
スコアが高いほど「買いシグナルとして有望」を意味する。

【スイングモード (日足)】
  RSI, MACD, BB, EMA200, MOM3M, Stoch, CCI, 52WK

【デイトレモード (10min足)】
  RSI, MACD, BB, MA, RVOL (出来高比), VWAP_B, ORB (Opening Range Breakout), MOM3B

スコア哲学（スイング）: トレンド+押し目戦略。上昇トレンド中の軽い調整を買う。
  - 急落・過売り銘柄は低得点（落下ナイフ除外）
  - EMA200 < 現在価格 → フラット設計（上昇トレンド内なら乖離率問わず高得点維持）
  - 中程度の売られ過ぎ (RSI 35〜52) が最高点

スコア哲学（デイトレ）: モメンタム+ブレイクアウト戦略。
  - ORB（30分レンジブレイク）を主要シグナルに
  - RVOL (出来高急増) と VWAP 強気乖離が補完
"""
from __future__ import annotations

import numpy as np

# ── 内部ユーティリティ ────────────────────────────────────────────────────────

def _clip(x: np.ndarray, lo: float = 0.0, hi: float = 25.0) -> np.ndarray:
    return np.clip(x, lo, hi)

def _fill_nan(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.where(np.isnan(x), fill, x)

def _lininterp(val: float | np.ndarray, x0: float, x1: float,
               y0: float, y1: float) -> np.ndarray:
    """val ∈ [x0, x1] → linear map to [y0, y1]"""
    if x1 == x0:
        return np.full_like(np.asarray(val, dtype=float), (y0 + y1) / 2)
    return y0 + (y1 - y0) * (np.asarray(val, dtype=float) - x0) / (x1 - x0)


# ══════════════════════════════════════════════════════════════════════════════
# スイング指標 (8指標)
# ══════════════════════════════════════════════════════════════════════════════

def score_rsi_swing(rsi: np.ndarray) -> np.ndarray:
    """
    RSI (0-100) → スイングスコア [0, 25]
    トレンド内押し目戦略: RSI 35〜52 が最高点 (上昇中の軽い調整)
    """
    rsi = np.asarray(rsi, dtype=float)
    out = np.zeros_like(rsi)
    # 過売り域 (< 25): 落下ナイフ → 低得点
    m1 = rsi < 25
    out = np.where(m1, _lininterp(rsi, 0, 25, 0, 5), out)
    # 押し目ゾーン (25 〜 35): 徐々に上昇
    m2 = (rsi >= 25) & (rsi < 35)
    out = np.where(m2, _lininterp(rsi, 25, 35, 5, 18), out)
    # 最適ゾーン (35 〜 52): 最高点
    m3 = (rsi >= 35) & (rsi < 52)
    out = np.where(m3, _lininterp(rsi, 35, 52, 18, 25), out)
    # ニュートラル (52 〜 65): 下降
    m4 = (rsi >= 52) & (rsi < 65)
    out = np.where(m4, _lininterp(rsi, 52, 65, 25, 10), out)
    # 過買い (>= 65): 低得点
    m5 = rsi >= 65
    out = np.where(m5, _lininterp(rsi, 65, 100, 10, 0), out)
    return _clip(_fill_nan(out))


def score_macd_swing(macd_hist: np.ndarray) -> np.ndarray:
    """
    MACD ヒストグラム → スイングスコア [0, 25]
    GC直後(ヒスト > 0 & 小さい)が最高点、DC後は低得点
    """
    h = np.asarray(macd_hist, dtype=float)
    out = np.zeros_like(h)
    # GC直後: ヒスト 0 〜 0.5 → 最高点
    m1 = (h >= 0) & (h < 0.5)
    out = np.where(m1, _lininterp(h, 0, 0.5, 20, 24), out)
    # GC継続: ヒスト 0.5 〜 2.0 → 中
    m2 = (h >= 0.5) & (h < 2.0)
    out = np.where(m2, _lininterp(h, 0.5, 2.0, 24, 12), out)
    # ヒスト縮小方向 (> 2.0): 低下
    m3 = h >= 2.0
    out = np.where(m3, _lininterp(h, 2.0, 5.0, 12, 5), out)
    # DC後 (< 0): 非常に低い
    m4 = h < 0
    out = np.where(m4, _lininterp(h, -3.0, 0, 0, 5), out)
    return _clip(_fill_nan(out))


def score_bb_swing(bb_pct: np.ndarray) -> np.ndarray:
    """
    BB パーセンタイル (0=下限, 1=上限) → スイングスコア [0, 25]
    下位 10〜30% が押し目ゾーン (最高点)
    """
    p = np.asarray(bb_pct, dtype=float)
    out = np.zeros_like(p)
    # バンド下抜け (< 0): 過売り → 中程度
    m0 = p < 0
    out = np.where(m0, _lininterp(p, -0.5, 0, 10, 18), out)
    # 下位 10% (0 〜 0.10): 押し目ゾーン
    m1 = (p >= 0) & (p < 0.10)
    out = np.where(m1, _lininterp(p, 0, 0.10, 18, 25), out)
    # 10〜30%: 良いゾーン
    m2 = (p >= 0.10) & (p < 0.30)
    out = np.where(m2, _lininterp(p, 0.10, 0.30, 25, 18), out)
    # 30〜70%: ニュートラル
    m3 = (p >= 0.30) & (p < 0.70)
    out = np.where(m3, _lininterp(p, 0.30, 0.70, 18, 8), out)
    # 70%以上: 高い → 低得点
    m4 = p >= 0.70
    out = np.where(m4, _lininterp(p, 0.70, 1.10, 8, 0), out)
    return _clip(_fill_nan(out))


def score_ema200_swing(ema200_pct: np.ndarray) -> np.ndarray:
    """
    EMA200 乖離率 (%) → スイングスコア [0, 25]
    乖離率 = (Close - EMA200) / EMA200 * 100
    正 → Close > EMA200 (上昇トレンド), 負 → Close < EMA200 (下降トレンド)

    上昇トレンド判定フィルターとして使用。
    乖離率が大きくなっても急激にスコアが下がらないフラット設計。
    NVDA/AMD等の常時+30~100%乖離銘柄でもスコアを維持する。
    """
    p = np.asarray(ema200_pct, dtype=float)
    out = np.zeros_like(p)
    # 下降トレンド (< -5%): 0点
    # m0: p < -5 → out remains 0
    # EMA200直下 (-5 to 0%): 線形 0→12pts
    m1 = (p >= -5) & (p < 0)
    out = np.where(m1, _lininterp(p, -5, 0, 0, 12), out)
    # EMA200直上 (0 to +5%): 線形 12→20pts
    m2 = (p >= 0) & (p < 5)
    out = np.where(m2, _lininterp(p, 0, 5, 12, 20), out)
    # 上昇トレンド (5 to 30%): 線形 20→25pts
    m3 = (p >= 5) & (p < 30)
    out = np.where(m3, _lininterp(p, 5, 30, 20, 25), out)
    # 強い上昇トレンド (30 to 80%): 25ptsフラット (NVDA/AMD等を除外しない)
    m4 = (p >= 30) & (p < 80)
    out = np.where(m4, 25, out)
    # 極端な過熱 (>= 80%): 線形 25→18pts (緩やかな低下)
    m5 = p >= 80
    out = np.where(m5, _lininterp(p, 80, 150, 25, 18), out)
    return _clip(_fill_nan(out))


def score_mom3m_swing(mom3m_pct: np.ndarray) -> np.ndarray:
    """
    3ヶ月モメンタム (63日ROC, %) → スイングスコア [0, 25]
    健全な調整 (−15〜−3%) が最高点。急落 (< −25%) は落下ナイフ。
    """
    m = np.asarray(mom3m_pct, dtype=float)
    out = np.zeros_like(m)
    # 急落 (< -25%): 危険 → 低得点
    m0 = m < -25
    out = np.where(m0, _lininterp(m, -40, -25, 0, 5), out)
    # 深い調整 (-25 〜 -15%)
    m1 = (m >= -25) & (m < -15)
    out = np.where(m1, _lininterp(m, -25, -15, 5, 15), out)
    # 健全な押し目 (-15 〜 -3%): 最高点
    m2 = (m >= -15) & (m < -3)
    out = np.where(m2, _lininterp(m, -15, -3, 15, 25), out)
    # 横ばい付近 (-3 〜 5%)
    m3 = (m >= -3) & (m < 5)
    out = np.where(m3, _lininterp(m, -3, 5, 25, 18), out)
    # 上昇中 (5 〜 20%)
    m4 = (m >= 5) & (m < 20)
    out = np.where(m4, _lininterp(m, 5, 20, 18, 10), out)
    # 急騰 (>= 20%): 過熱
    m5 = m >= 20
    out = np.where(m5, _lininterp(m, 20, 50, 10, 3), out)
    return _clip(_fill_nan(out))


def score_stoch_swing(stoch_k: np.ndarray) -> np.ndarray:
    """
    Stochastic %K (0-100) → スイングスコア [0, 25]
    売られ過ぎ圏 (<20) からのGC候補が最高点
    """
    k = np.asarray(stoch_k, dtype=float)
    out = np.zeros_like(k)
    m1 = k < 10
    out = np.where(m1, _lininterp(k, 0, 10, 18, 25), out)
    m2 = (k >= 10) & (k < 20)
    out = np.where(m2, _lininterp(k, 10, 20, 25, 22), out)
    m3 = (k >= 20) & (k < 40)
    out = np.where(m3, _lininterp(k, 20, 40, 22, 14), out)
    m4 = (k >= 40) & (k < 60)
    out = np.where(m4, _lininterp(k, 40, 60, 14, 8), out)
    m5 = (k >= 60) & (k < 80)
    out = np.where(m5, _lininterp(k, 60, 80, 8, 3), out)
    m6 = k >= 80
    out = np.where(m6, _lininterp(k, 80, 100, 3, 0), out)
    return _clip(_fill_nan(out))


def score_cci_swing(cci: np.ndarray) -> np.ndarray:
    """
    CCI (任意範囲) → スイングスコア [0, 25]
    −200 以下が最高点 (深い過売り)
    """
    c = np.asarray(cci, dtype=float)
    out = np.zeros_like(c)
    m1 = c < -200
    out = np.where(m1, _lininterp(c, -300, -200, 25, 22), out)
    m2 = (c >= -200) & (c < -100)
    out = np.where(m2, _lininterp(c, -200, -100, 22, 14), out)
    m3 = (c >= -100) & (c < 0)
    out = np.where(m3, _lininterp(c, -100, 0, 14, 8), out)
    m4 = (c >= 0) & (c < 100)
    out = np.where(m4, _lininterp(c, 0, 100, 8, 3), out)
    m5 = c >= 100
    out = np.where(m5, _lininterp(c, 100, 200, 3, 0), out)
    return _clip(_fill_nan(out))


def score_52wk_swing(pct_52wk: np.ndarray) -> np.ndarray:
    """
    52週レンジ内位置 (0=年間安値, 1=年間高値) → スイングスコア [0, 25]
    中間帯 (35〜65%) が最高点 (上昇後の押し目)
    """
    p = np.asarray(pct_52wk, dtype=float)
    out = np.zeros_like(p)
    # 安値圏 (< 15%): リスク大
    m1 = p < 0.15
    out = np.where(m1, _lininterp(p, 0, 0.15, 3, 10), out)
    # 下位圏 (15〜35%): 押し目候補
    m2 = (p >= 0.15) & (p < 0.35)
    out = np.where(m2, _lininterp(p, 0.15, 0.35, 10, 20), out)
    # 中間帯 (35〜65%): 最高点
    m3 = (p >= 0.35) & (p < 0.65)
    out = np.where(m3, _lininterp(p, 0.35, 0.65, 20, 25), out)
    # 上位圏 (65〜85%): 過熱
    m4 = (p >= 0.65) & (p < 0.85)
    out = np.where(m4, _lininterp(p, 0.65, 0.85, 25, 12), out)
    # 高値圏 (>= 85%): リスク
    m5 = p >= 0.85
    out = np.where(m5, _lininterp(p, 0.85, 1.0, 12, 5), out)
    return _clip(_fill_nan(out))


# スイング指標名と関数のマッピング
SWING_INDICATORS = {
    "RSI":    score_rsi_swing,
    "MACD":   score_macd_swing,
    "BB":     score_bb_swing,
    "EMA200": score_ema200_swing,
    "MOM3M":  score_mom3m_swing,
    "Stoch":  score_stoch_swing,
    "CCI":    score_cci_swing,
    "52WK":   score_52wk_swing,
}
IND_NAMES_SWING = list(SWING_INDICATORS.keys())  # 順序固定


# ══════════════════════════════════════════════════════════════════════════════
# デイトレ指標 (8指標)
# ══════════════════════════════════════════════════════════════════════════════

def score_rsi_day(rsi: np.ndarray) -> np.ndarray:
    """
    RSI (0-100) → デイトレスコア [0, 25]
    短期モメンタム: RSI 50〜65 (上昇方向) が最高点
    """
    rsi = np.asarray(rsi, dtype=float)
    out = np.zeros_like(rsi)
    m1 = rsi < 40
    out = np.where(m1, _lininterp(rsi, 0, 40, 0, 8), out)
    m2 = (rsi >= 40) & (rsi < 50)
    out = np.where(m2, _lininterp(rsi, 40, 50, 8, 15), out)
    m3 = (rsi >= 50) & (rsi < 65)
    out = np.where(m3, _lininterp(rsi, 50, 65, 15, 25), out)
    m4 = (rsi >= 65) & (rsi < 80)
    out = np.where(m4, _lininterp(rsi, 65, 80, 25, 15), out)
    m5 = rsi >= 80
    out = np.where(m5, _lininterp(rsi, 80, 100, 15, 5), out)
    return _clip(_fill_nan(out))


def score_macd_day(macd_hist: np.ndarray) -> np.ndarray:
    """MACD ヒストグラム → デイトレスコア (ゼロクロス直後が最高点)"""
    h = np.asarray(macd_hist, dtype=float)
    out = np.zeros_like(h)
    m1 = (h >= 0) & (h < 0.3)
    out = np.where(m1, _lininterp(h, 0, 0.3, 20, 25), out)
    m2 = (h >= 0.3) & (h < 1.0)
    out = np.where(m2, _lininterp(h, 0.3, 1.0, 25, 15), out)
    m3 = h >= 1.0
    out = np.where(m3, _lininterp(h, 1.0, 3.0, 15, 5), out)
    m4 = (h < 0) & (h >= -0.5)
    out = np.where(m4, _lininterp(h, -0.5, 0, 5, 10), out)
    m5 = h < -0.5
    out = np.where(m5, _lininterp(h, -2.0, -0.5, 0, 5), out)
    return _clip(_fill_nan(out))


def score_bb_day(bb_pct: np.ndarray) -> np.ndarray:
    """BB パーセンタイル → デイトレスコア (上位ブレイクアウト狙い)"""
    p = np.asarray(bb_pct, dtype=float)
    out = np.zeros_like(p)
    m1 = p < 0.4
    out = np.where(m1, _lininterp(p, 0, 0.4, 3, 10), out)
    m2 = (p >= 0.4) & (p < 0.7)
    out = np.where(m2, _lininterp(p, 0.4, 0.7, 10, 18), out)
    m3 = (p >= 0.7) & (p < 0.9)
    out = np.where(m3, _lininterp(p, 0.7, 0.9, 18, 25), out)
    m4 = (p >= 0.9) & (p <= 1.0)
    out = np.where(m4, _lininterp(p, 0.9, 1.0, 25, 22), out)
    m5 = p > 1.0
    out = np.where(m5, _lininterp(p, 1.0, 1.3, 22, 15), out)
    return _clip(_fill_nan(out))


def score_ma_day(ma_pct: np.ndarray) -> np.ndarray:
    """
    移動平均乖離率 (%) → デイトレスコア
    MA より上にある (0〜3%) が最高点
    """
    m = np.asarray(ma_pct, dtype=float)
    out = np.zeros_like(m)
    m1 = m < -2
    out = np.where(m1, _lininterp(m, -5, -2, 0, 8), out)
    m2 = (m >= -2) & (m < 0)
    out = np.where(m2, _lininterp(m, -2, 0, 8, 15), out)
    m3 = (m >= 0) & (m < 3)
    out = np.where(m3, _lininterp(m, 0, 3, 15, 25), out)
    m4 = (m >= 3) & (m < 7)
    out = np.where(m4, _lininterp(m, 3, 7, 25, 15), out)
    m5 = m >= 7
    out = np.where(m5, _lininterp(m, 7, 15, 15, 5), out)
    return _clip(_fill_nan(out))


def score_rvol_day(rvol: np.ndarray) -> np.ndarray:
    """
    相対出来高 (当日出来高 / 20日平均出来高) → デイトレスコア
    1.5〜3.0 が最高点 (出来高急増だが異常ではない)
    """
    v = np.asarray(rvol, dtype=float)
    out = np.zeros_like(v)
    m1 = v < 0.5
    out = np.where(m1, _lininterp(v, 0, 0.5, 0, 3), out)
    m2 = (v >= 0.5) & (v < 1.0)
    out = np.where(m2, _lininterp(v, 0.5, 1.0, 3, 10), out)
    m3 = (v >= 1.0) & (v < 1.5)
    out = np.where(m3, _lininterp(v, 1.0, 1.5, 10, 18), out)
    m4 = (v >= 1.5) & (v < 3.0)
    out = np.where(m4, _lininterp(v, 1.5, 3.0, 18, 25), out)
    m5 = (v >= 3.0) & (v < 5.0)
    out = np.where(m5, _lininterp(v, 3.0, 5.0, 25, 18), out)
    m6 = v >= 5.0
    out = np.where(m6, _lininterp(v, 5.0, 10.0, 18, 8), out)
    return _clip(_fill_nan(out))


def score_vwap_bull_day(vwap_pct: np.ndarray) -> np.ndarray:
    """
    VWAP 乖離率 (%) → デイトレスコア (VWAP より上が強気)
    +0.2〜+1.5% 上方が最高点
    """
    p = np.asarray(vwap_pct, dtype=float)
    out = np.zeros_like(p)
    m1 = p < -1.0
    out = np.where(m1, _lininterp(p, -3.0, -1.0, 0, 5), out)
    m2 = (p >= -1.0) & (p < 0)
    out = np.where(m2, _lininterp(p, -1.0, 0, 5, 12), out)
    m3 = (p >= 0) & (p < 0.2)
    out = np.where(m3, _lininterp(p, 0, 0.2, 12, 18), out)
    m4 = (p >= 0.2) & (p < 1.5)
    out = np.where(m4, _lininterp(p, 0.2, 1.5, 18, 25), out)
    m5 = (p >= 1.5) & (p < 3.0)
    out = np.where(m5, _lininterp(p, 1.5, 3.0, 25, 15), out)
    m6 = p >= 3.0
    out = np.where(m6, _lininterp(p, 3.0, 5.0, 15, 5), out)
    return _clip(_fill_nan(out))


def score_orb_day(orb_pct: np.ndarray) -> np.ndarray:
    """
    Opening Range Breakout スコア (0=ブレイクアウトなし, 1=完全ブレイク)
    ORB は前処理済み連続値 [0, 1] で渡す。
    > 0.5 が「確実にブレイクアウト」と判断。
    """
    o = np.asarray(orb_pct, dtype=float)
    out = np.zeros_like(o)
    m1 = o < 0
    out = np.where(m1, 0, out)
    m2 = (o >= 0) & (o < 0.3)
    out = np.where(m2, _lininterp(o, 0, 0.3, 0, 8), out)
    m3 = (o >= 0.3) & (o < 0.6)
    out = np.where(m3, _lininterp(o, 0.3, 0.6, 8, 18), out)
    m4 = (o >= 0.6) & (o <= 1.0)
    out = np.where(m4, _lininterp(o, 0.6, 1.0, 18, 25), out)
    m5 = o > 1.0
    out = np.where(m5, 25, out)
    return _clip(_fill_nan(out))


def score_mom3b_day(mom3b_pct: np.ndarray) -> np.ndarray:
    """
    3バー (30分) モメンタム (%) → デイトレスコア
    +0.3〜+1.5% が最高点 (強い短期モメンタム)
    """
    m = np.asarray(mom3b_pct, dtype=float)
    out = np.zeros_like(m)
    m1 = m < -1.0
    out = np.where(m1, 0, out)
    m2 = (m >= -1.0) & (m < 0)
    out = np.where(m2, _lininterp(m, -1.0, 0, 0, 8), out)
    m3 = (m >= 0) & (m < 0.3)
    out = np.where(m3, _lininterp(m, 0, 0.3, 8, 15), out)
    m4 = (m >= 0.3) & (m < 1.5)
    out = np.where(m4, _lininterp(m, 0.3, 1.5, 15, 25), out)
    m5 = (m >= 1.5) & (m < 3.0)
    out = np.where(m5, _lininterp(m, 1.5, 3.0, 25, 15), out)
    m6 = m >= 3.0
    out = np.where(m6, _lininterp(m, 3.0, 5.0, 15, 8), out)
    return _clip(_fill_nan(out))


# デイトレ指標名と関数のマッピング
DAY_INDICATORS = {
    "RSI":    score_rsi_day,
    "MACD":   score_macd_day,
    "BB":     score_bb_day,
    "MA":     score_ma_day,
    "RVOL":   score_rvol_day,
    "VWAP_B": score_vwap_bull_day,
    "ORB":    score_orb_day,
    "MOM3B":  score_mom3b_day,
}
IND_NAMES_DAY = list(DAY_INDICATORS.keys())  # 順序固定
