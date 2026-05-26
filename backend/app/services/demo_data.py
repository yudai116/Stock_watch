"""
Demo data generator for when Yahoo Finance is inaccessible (e.g. restricted network environments).
Set DEMO_MODE=true environment variable to enable.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

_STOCKS = {
    "7203.T": {"name": "トヨタ自動車株式会社", "market": "JP", "currency": "JPY", "base": 3200, "pe": (12.5, 11.2)},
    "6758.T": {"name": "ソニーグループ株式会社", "market": "JP", "currency": "JPY", "base": 2800, "pe": (18.3, 16.9)},
    "9984.T": {"name": "ソフトバンクグループ株式会社", "market": "JP", "currency": "JPY", "base": 9500, "pe": (28.4, 25.1)},
    "7974.T": {"name": "任天堂株式会社", "market": "JP", "currency": "JPY", "base": 8200, "pe": (14.7, 13.8)},
    "AAPL": {"name": "Apple Inc.", "market": "US", "currency": "USD", "base": 185, "pe": (29.3, 27.8)},
    "MSFT": {"name": "Microsoft Corporation", "market": "US", "currency": "USD", "base": 420, "pe": (35.2, 31.5)},
    "NVDA": {"name": "NVIDIA Corporation", "market": "US", "currency": "USD", "base": 950, "pe": (72.1, 45.3)},
    "TSLA": {"name": "Tesla, Inc.", "market": "US", "currency": "USD", "base": 175, "pe": (55.8, 42.1)},
}

_SEEDS = {t: i * 17 + 42 for i, t in enumerate(_STOCKS)}


def _generate_prices(ticker: str, periods: int = 120) -> pd.DataFrame:
    seed = _SEEDS.get(ticker, 0)
    np.random.seed(seed)
    meta = _STOCKS.get(ticker, {"base": 100})
    base = meta["base"]

    trend = np.linspace(0, base * 0.05 * np.random.choice([-1, 1]), periods)
    noise = np.cumsum(np.random.randn(periods) * base * 0.008)
    close = base + trend + noise
    close = np.maximum(close, base * 0.5)

    high = close * (1 + np.abs(np.random.randn(periods)) * 0.008)
    low = close * (1 - np.abs(np.random.randn(periods)) * 0.008)
    open_p = close * (1 + np.random.randn(periods) * 0.004)
    volume = np.abs(np.random.randn(periods) * 5_000_000 + 10_000_000)

    dates = pd.date_range(end=datetime.now(), periods=periods, freq="B")
    return pd.DataFrame({
        "Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)


def get_demo_df(ticker: str) -> tuple[pd.DataFrame, dict]:
    df = _generate_prices(ticker.upper())
    meta = _STOCKS.get(ticker.upper(), {"name": ticker, "market": "US", "currency": "USD", "pe": (None, None)})
    pe_t, pe_f = meta.get("pe", (None, None))
    info = {
        "longName": meta.get("name", ticker),
        "trailingPE": pe_t,
        "forwardPE": pe_f,
    }
    return df, info


def is_known_demo_ticker(ticker: str) -> bool:
    return ticker.upper() in _STOCKS


def get_demo_meta(ticker: str) -> dict:
    return _STOCKS.get(ticker.upper(), {})
