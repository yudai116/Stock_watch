import numpy as np
import pandas as pd
from app.services.indicators import calc_rsi, calc_macd, calc_bollinger, calc_ema


def _interp(value: float, breakpoints: list) -> float:
    """Piecewise linear interpolation. breakpoints: [(x, score), ...]"""
    if value <= breakpoints[0][0]:
        return float(breakpoints[0][1])
    if value >= breakpoints[-1][0]:
        return float(breakpoints[-1][1])
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return float(breakpoints[-1][1])


def score_rsi(df: pd.DataFrame) -> dict:
    rsi = calc_rsi(df["Close"])
    rsi_clean = rsi.dropna()
    if len(rsi_clean) < 2:
        return {"score": 12, "max": 25, "value": None, "signal": "insufficient_data"}

    current = float(rsi_clean.iloc[-1])
    prev = float(rsi_clean.iloc[-2])
    rising = current > prev

    if current < 25:
        raw = 23.0
        signal = "extreme_oversold"
    elif current < 35:
        base = _interp(current, [(25, 22), (35, 12)])
        raw = base + (3 if rising else 0)
        raw = min(raw, 25)
        signal = "oversold_recovery" if rising else "oversold"
    elif current < 45:
        raw = _interp(current, [(35, 14), (45, 10)])
        signal = "approaching_oversold"
    elif current < 55:
        raw = _interp(current, [(45, 12), (55, 8)])
        signal = "neutral"
    elif current < 65:
        raw = _interp(current, [(55, 8), (65, 5)])
        signal = "neutral_bullish"
    elif current < 75:
        raw = _interp(current, [(65, 5), (75, 2)])
        signal = "overbought"
    else:
        raw = _interp(current, [(75, 2), (85, 0)])
        signal = "extreme_overbought"

    return {"score": round(raw), "max": 25, "value": round(current, 2), "signal": signal}


def score_macd(df: pd.DataFrame) -> dict:
    macd_line, signal_line, histogram = calc_macd(df["Close"])
    valid = macd_line.dropna()
    if len(valid) < 3:
        return {"score": 12, "max": 25, "value": None, "signal": "insufficient_data"}

    m_curr = float(macd_line.iloc[-1])
    m_prev = float(macd_line.iloc[-2])
    s_curr = float(signal_line.iloc[-1])
    s_prev = float(signal_line.iloc[-2])
    h_curr = float(histogram.iloc[-1])
    h_prev = float(histogram.iloc[-2])

    bullish_cross_now = m_prev < s_prev and m_curr >= s_curr
    bullish_cross_recent = False
    for i in range(2, min(4, len(macd_line))):
        mp = float(macd_line.iloc[-(i + 1)])
        sp = float(signal_line.iloc[-(i + 1)])
        mc = float(macd_line.iloc[-i])
        sc = float(signal_line.iloc[-i])
        if mp < sp and mc >= sc:
            bullish_cross_recent = True
            break

    bearish_cross_now = m_prev > s_prev and m_curr <= s_curr

    if bullish_cross_now:
        score, signal = 24, "bullish_crossover"
    elif bullish_cross_recent and h_curr > 0:
        score, signal = 20, "recent_bullish_crossover"
    elif m_curr > s_curr and h_curr > h_prev:
        score, signal = 15, "bullish_momentum"
    elif m_curr > s_curr and h_curr <= h_prev:
        score, signal = 10, "bullish_fading"
    elif bearish_cross_now:
        score, signal = 2, "bearish_crossover"
    elif m_curr < s_curr and h_curr > h_prev:
        score, signal = 6, "bearish_weakening"
    else:
        score, signal = 3, "bearish"

    return {"score": score, "max": 25, "value": round(m_curr, 4), "signal": signal}


def score_bollinger(df: pd.DataFrame) -> dict:
    upper, middle, lower, pct_b = calc_bollinger(df["Close"])
    pct_b_clean = pct_b.dropna()
    if len(pct_b_clean) < 2:
        return {"score": 12, "max": 25, "value": None, "signal": "insufficient_data"}

    current_pct_b = float(pct_b_clean.iloc[-1])

    band_width = (upper - lower) / middle
    bw_clean = band_width.dropna()
    squeeze_bonus = 0
    if len(bw_clean) >= 5:
        bw_now = float(bw_clean.iloc[-1])
        bw_avg = float(bw_clean.iloc[-5:].mean())
        if bw_now < bw_avg * 0.85 and current_pct_b < 0.3:
            squeeze_bonus = 3

    if current_pct_b < 0:
        raw = _interp(current_pct_b, [(-0.2, 25), (0, 21)])
        signal = "below_lower_band"
    elif current_pct_b < 0.1:
        raw = _interp(current_pct_b, [(0, 21), (0.1, 17)])
        signal = "near_lower_band"
    elif current_pct_b < 0.25:
        raw = _interp(current_pct_b, [(0.1, 17), (0.25, 12)])
        signal = "lower_zone"
    elif current_pct_b < 0.5:
        raw = _interp(current_pct_b, [(0.25, 12), (0.5, 8)])
        signal = "lower_half"
    elif current_pct_b < 0.75:
        raw = _interp(current_pct_b, [(0.5, 8), (0.75, 4)])
        signal = "upper_half"
    elif current_pct_b < 0.9:
        raw = _interp(current_pct_b, [(0.75, 4), (0.9, 1)])
        signal = "near_upper_band"
    else:
        raw = _interp(current_pct_b, [(0.9, 1), (1.2, 0)])
        signal = "above_upper_band"

    total = min(round(raw + squeeze_bonus), 25)
    return {"score": total, "max": 25, "value": round(current_pct_b, 3), "signal": signal}


def score_moving_average(df: pd.DataFrame) -> dict:
    close = df["Close"]
    ma20 = calc_ema(close, 20)
    ma50 = calc_ema(close, 50)

    if len(ma20.dropna()) < 2 or len(ma50.dropna()) < 2:
        return {"score": 12, "max": 25, "value": None, "signal": "insufficient_data"}

    price_now = float(close.iloc[-1])
    price_prev = float(close.iloc[-2])
    ma20_now = float(ma20.iloc[-1])
    ma20_prev = float(ma20.iloc[-2])
    ma50_now = float(ma50.iloc[-1])
    ma50_prev = float(ma50.iloc[-2])

    cross_above_today = price_prev < ma20_prev and price_now >= ma20_now
    cross_above_recent = False
    for i in range(2, min(4, len(close))):
        pp = float(close.iloc[-(i + 1)])
        pm = float(ma20.iloc[-(i + 1)])
        pc = float(close.iloc[-i])
        mc = float(ma20.iloc[-i])
        if pp < pm and pc >= mc and price_now > ma20_now:
            cross_above_recent = True
            break

    if cross_above_today:
        price_score = 15
        price_signal = "cross_above_ma20"
    elif cross_above_recent:
        price_score = 12
        price_signal = "recent_cross_above_ma20"
    elif price_now > ma20_now:
        ratio = price_now / ma20_now
        price_score = round(_interp(ratio, [(1.0, 9), (1.05, 6), (1.1, 4)]))
        price_signal = "above_ma20"
    elif abs(price_now - ma20_now) / ma20_now < 0.005:
        price_score = 5
        price_signal = "at_ma20"
    else:
        ratio = price_now / ma20_now
        price_score = round(_interp(ratio, [(0.9, 4), (0.95, 3), (1.0, 0)]))
        price_signal = "below_ma20"

    gap = ma20_now - ma50_now
    gap_prev = ma20_prev - ma50_prev
    widening = abs(gap) > abs(gap_prev)

    if gap > 0:
        ma_score = 9 if widening else 6
        ma_signal = "golden_cross_zone"
    elif abs(gap / ma50_now) < 0.005:
        ma_score = 4
        ma_signal = "at_crossover"
    else:
        ma_score = 1 if widening else 3
        ma_signal = "death_cross_zone"

    total = min(price_score + ma_score, 25)
    return {"score": total, "max": 25, "value": round(price_now / ma20_now, 4), "signal": price_signal}


def compute_score(df: pd.DataFrame) -> dict:
    rsi = score_rsi(df)
    macd = score_macd(df)
    bb = score_bollinger(df)
    ma = score_moving_average(df)
    total = rsi["score"] + macd["score"] + bb["score"] + ma["score"]
    return {
        "total": total,
        "rsi": rsi,
        "macd": macd,
        "bollinger": bb,
        "moving_avg": ma,
    }


def pe_label(pe: float | None, market: str) -> str | None:
    if pe is None or pe <= 0:
        return None
    if market == "JP":
        if pe < 15:
            return "割安"
        elif pe <= 25:
            return "適正"
        else:
            return "割高"
    else:
        if pe < 20:
            return "割安"
        elif pe <= 35:
            return "適正"
        else:
            return "割高"
