import os
import time
import threading
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

from app.services.indicators import calc_rsi, calc_macd, calc_bollinger, calc_ema
from app.services.scorer import compute_score, pe_label as get_pe_label
from app.services.demo_data import get_demo_df, is_known_demo_ticker
from app.models import ScoreComponent, StockScore, StockDetail, OHLCVPoint, SeriesPoint, MACDPoint, BBPoint

DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")

_cache: dict = {}
_cache_lock = threading.Lock()

DEFAULT_WATCHLIST = ["7203.T", "6758.T", "9984.T", "7974.T", "AAPL", "MSFT", "NVDA", "TSLA"]


def _detect_market(ticker: str) -> str:
    return "JP" if ticker.upper().endswith(".T") else "US"


def _get_currency(market: str) -> str:
    return "JPY" if market == "JP" else "USD"


def _fetch_raw(ticker: str) -> tuple[pd.DataFrame, dict]:
    if DEMO_MODE and is_known_demo_ticker(ticker):
        return get_demo_df(ticker)
    t = yf.Ticker(ticker)
    df = t.history(period="120d", interval="1d")
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return df, info


def _build_score_data(ticker: str, df: pd.DataFrame, info: dict) -> StockScore:
    market = _detect_market(ticker)
    currency = _get_currency(market)

    scores = compute_score(df)
    trailing_pe = info.get("trailingPE") or None
    forward_pe = info.get("forwardPE") or None
    if trailing_pe and (trailing_pe < 0 or trailing_pe > 5000):
        trailing_pe = None
    if forward_pe and (forward_pe < 0 or forward_pe > 5000):
        forward_pe = None

    name = info.get("longName") or info.get("shortName") or ticker
    price = float(df["Close"].iloc[-1]) if not df.empty else None
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
    change_pct = round((price - prev_close) / prev_close * 100, 2) if price and prev_close else None

    components = {
        "rsi": ScoreComponent(**scores["rsi"]),
        "macd": ScoreComponent(**scores["macd"]),
        "bollinger": ScoreComponent(**scores["bollinger"]),
        "moving_avg": ScoreComponent(**scores["moving_avg"]),
    }

    return StockScore(
        ticker=ticker,
        market=market,
        name=name,
        price=round(price, 2) if price else None,
        currency=currency,
        change_pct=change_pct,
        score=scores["total"],
        score_components=components,
        trailing_pe=round(trailing_pe, 2) if trailing_pe else None,
        forward_pe=round(forward_pe, 2) if forward_pe else None,
        pe_label=get_pe_label(trailing_pe, market),
        last_updated=datetime.now(timezone.utc).isoformat(),
    )


def fetch_score(ticker: str) -> StockScore:
    with _cache_lock:
        entry = _cache.get(ticker)
        if entry and (time.time() - entry["ts"]) < 300:
            return entry["data"]

    df, info = _fetch_raw(ticker)
    result = _build_score_data(ticker, df, info)

    with _cache_lock:
        _cache[ticker] = {"ts": time.time(), "df": df, "info": info, "data": result}

    return result


def fetch_detail(ticker: str) -> StockDetail:
    with _cache_lock:
        entry = _cache.get(ticker)
        if entry and (time.time() - entry["ts"]) < 300:
            df = entry["df"]
            info = entry["info"]
            score_data = entry["data"]
        else:
            entry = None

    if entry is None:
        df, info = _fetch_raw(ticker)
        score_data = _build_score_data(ticker, df, info)
        with _cache_lock:
            _cache[ticker] = {"ts": time.time(), "df": df, "info": info, "data": score_data}

    rsi_series = calc_rsi(df["Close"])
    macd_line, signal_line, histogram = calc_macd(df["Close"])
    upper, middle, lower, pct_b = calc_bollinger(df["Close"])
    ma20 = calc_ema(df["Close"], 20)
    ma50 = calc_ema(df["Close"], 50)

    history = []
    for idx, row in df.iterrows():
        history.append(OHLCVPoint(
            date=idx.strftime("%Y-%m-%d"),
            open=round(float(row["Open"]), 4),
            high=round(float(row["High"]), 4),
            low=round(float(row["Low"]), 4),
            close=round(float(row["Close"]), 4),
            volume=float(row["Volume"]),
        ))

    def to_series(s: pd.Series) -> list[SeriesPoint]:
        return [
            SeriesPoint(date=idx.strftime("%Y-%m-%d"), value=round(float(v), 4))
            for idx, v in s.items() if pd.notna(v)
        ]

    macd_pts = []
    for idx in df.index:
        if idx in macd_line.index and pd.notna(macd_line.get(idx)) and pd.notna(signal_line.get(idx)):
            macd_pts.append(MACDPoint(
                date=idx.strftime("%Y-%m-%d"),
                macd=round(float(macd_line[idx]), 4),
                signal=round(float(signal_line[idx]), 4),
                histogram=round(float(histogram[idx]), 4),
            ))

    bb_pts = []
    for idx in df.index:
        if idx in upper.index and pd.notna(upper.get(idx)):
            bb_pts.append(BBPoint(
                date=idx.strftime("%Y-%m-%d"),
                upper=round(float(upper[idx]), 4),
                middle=round(float(middle[idx]), 4),
                lower=round(float(lower[idx]), 4),
                pct_b=round(float(pct_b[idx]), 4) if pd.notna(pct_b.get(idx)) else 0,
            ))

    return StockDetail(
        ticker=score_data.ticker,
        market=score_data.market,
        name=score_data.name,
        price=score_data.price,
        currency=score_data.currency,
        change_pct=score_data.change_pct,
        score=score_data.score,
        score_components=score_data.score_components,
        trailing_pe=score_data.trailing_pe,
        forward_pe=score_data.forward_pe,
        pe_label=score_data.pe_label,
        history=history,
        rsi_series=to_series(rsi_series),
        macd_series=macd_pts,
        bb_series=bb_pts,
        ma20_series=to_series(ma20),
        ma50_series=to_series(ma50),
        last_updated=score_data.last_updated,
    )


def validate_ticker(ticker: str) -> tuple[bool, str | None, str | None]:
    if DEMO_MODE and is_known_demo_ticker(ticker):
        from app.services.demo_data import get_demo_meta
        meta = get_demo_meta(ticker)
        return True, meta.get("name", ticker), meta.get("market", _detect_market(ticker))
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="5d", interval="1d")
        if df.empty:
            return False, None, None
        try:
            info = t.info or {}
        except Exception:
            info = {}
        name = info.get("longName") or info.get("shortName") or ticker
        market = _detect_market(ticker)
        return True, name, market
    except Exception:
        return False, None, None


def invalidate_cache(ticker: str):
    with _cache_lock:
        _cache.pop(ticker, None)


def cache_size() -> int:
    with _cache_lock:
        return len(_cache)


def refresh_all_cached():
    with _cache_lock:
        tickers = list(_cache.keys())
    for ticker in tickers:
        try:
            df, info = _fetch_raw(ticker)
            result = _build_score_data(ticker, df, info)
            with _cache_lock:
                _cache[ticker] = {"ts": time.time(), "df": df, "info": info, "data": result}
        except Exception:
            pass
