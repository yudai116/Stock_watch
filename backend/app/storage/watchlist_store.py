import json
import os
import threading
from app.config import settings
from app.services.market_data import DEFAULT_WATCHLIST

_lock = threading.Lock()
_filepath = settings.watchlist_file


def _ensure_file():
    os.makedirs(os.path.dirname(_filepath), exist_ok=True)
    if not os.path.exists(_filepath):
        _save_raw(list(DEFAULT_WATCHLIST))


def _load_raw() -> list[str]:
    _ensure_file()
    with open(_filepath, "r") as f:
        data = json.load(f)
    return data.get("tickers", [])


def _save_raw(tickers: list[str]):
    os.makedirs(os.path.dirname(_filepath), exist_ok=True)
    with open(_filepath, "w") as f:
        json.dump({"tickers": tickers}, f, indent=2)


def load() -> list[str]:
    with _lock:
        return _load_raw()


def add(ticker: str) -> list[str]:
    with _lock:
        tickers = _load_raw()
        upper = ticker.upper()
        if upper not in tickers:
            tickers.append(upper)
            _save_raw(tickers)
        return tickers


def remove(ticker: str) -> list[str]:
    with _lock:
        tickers = _load_raw()
        upper = ticker.upper()
        tickers = [t for t in tickers if t != upper]
        _save_raw(tickers)
        return tickers
