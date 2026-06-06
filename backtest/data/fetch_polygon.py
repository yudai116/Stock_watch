"""
fetch_polygon.py — Polygon.io (Massive) から株価データを取得

対応足種:
  SWING: 1Hour足 (multiplier=1, timespan=hour)
  DAY:   10Min足 (multiplier=10, timespan=minute)

環境変数:
  POLYGON_API_KEY : Polygon.io API キー (Developer プラン以上で10年分取得可)

出力フォーマット (list-of-records、loader._to_arrays が対応済み):
  {ticker: [{"date": "YYYY-MM-DD HH:MM", "open": ..., "high": ...,
             "low": ..., "close": ..., "volume": ...}, ...]}
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

HERE = Path(__file__).parent.parent  # backtest/

_BASE = "https://api.polygon.io/v2/aggs/ticker"

# Alpaca timeframe 文字列 → Polygon (multiplier, timespan) への変換
_TIMEFRAME_MAP: dict[str, tuple[int, str]] = {
    "1Hour":  (1,  "hour"),
    "10Min":  (10, "minute"),
    "5Min":   (5,  "minute"),
    "1Min":   (1,  "minute"),
    "1Day":   (1,  "day"),
}


def _api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "POLYGON_API_KEY 環境変数を設定してください。\n"
            "Polygon.io Developer: https://polygon.io/dashboard/api-keys"
        )
    return key


def _fetch_bars(
    symbol: str,
    multiplier: int,
    timespan: str,
    start: str,
    end: str,
) -> list[dict]:
    """Polygon aggregates API からページネーション付きで全バーを取得する。"""
    try:
        import requests
    except ImportError:
        raise ImportError("requests が必要です: pip install requests")

    key = _api_key()
    url = f"{_BASE}/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    50000,
        "apiKey":   key,
    }

    all_results: list[dict] = []
    while url:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(60)
            continue
        if resp.status_code == 403:
            raise PermissionError(
                f"{symbol}: HTTP 403 — プランの履歴上限を超えています (start={start})"
            )
        if resp.status_code != 200:
            raise RuntimeError(f"{symbol}: Polygon API エラー {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        all_results.extend(data.get("results") or [])

        # 次ページ (next_url はクエリパラメータ不要)
        next_url = data.get("next_url")
        if next_url:
            url = f"{next_url}&apiKey={key}"
            params = {}
            time.sleep(0.1)
        else:
            break

    return all_results


def _bars_to_records(bars: list[dict], timespan: str) -> list[dict]:
    """Polygon バーリスト → list-of-records フォーマットに変換する。"""
    records = []
    for b in bars:
        ts_ms = b["t"]
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        if timespan == "day":
            date_str = dt.strftime("%Y-%m-%d")
        else:
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        records.append({
            "date":   date_str,
            "open":   round(float(b["o"]), 4),
            "high":   round(float(b["h"]), 4),
            "low":    round(float(b["l"]), 4),
            "close":  round(float(b["c"]), 4),
            "volume": int(b.get("v", 0)),
        })
    return records


def fetch_stock_data(
    tickers: list[str],
    timeframe: Literal["1Min", "5Min", "10Min", "1Hour", "1Day"] = "10Min",
    lookback_years: int = 10,
    output_path: Path | None = None,
    existing: dict | None = None,
) -> dict:
    """
    Polygon.io から株価データを取得して JSON に保存する。

    Parameters
    ----------
    tickers       : 取得銘柄リスト
    timeframe     : Alpaca 互換の足種文字列
    lookback_years: 遡る年数
    output_path   : 保存先 (None = timeframe に応じてデフォルト)
    existing      : 既存データ (差分取得してマージ)

    Returns
    -------
    dict  {ticker: list[{date, open, high, low, close, volume}]}
    """
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(f"未対応の timeframe: {timeframe}。対応: {list(_TIMEFRAME_MAP)}")

    multiplier, timespan = _TIMEFRAME_MAP[timeframe]

    if output_path is None:
        if timeframe in ("10Min", "5Min", "1Min"):
            output_path = HERE / "price_data_intraday.json"
        elif timeframe == "1Hour":
            output_path = HERE / "price_data_swing.json"
        else:
            output_path = HERE / "price_data.json"

    end_dt    = datetime.now(timezone.utc)
    start_dt  = end_dt - timedelta(days=lookback_years * 365 + 30)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    print(f"[fetch_polygon] timeframe={timeframe} ({multiplier}/{timespan})")
    print(f"[fetch_polygon] 期間: {start_str} 〜 {end_str}  {len(tickers)}銘柄")

    result: dict = {}
    # 既存データをlist-of-records形式で保持（差分取得用）
    if existing:
        for t, v in existing.items():
            if isinstance(v, list):
                result[t] = v
            elif isinstance(v, dict) and "dates" in v:
                # dict形式 (旧Alpaca形式) → list形式に変換
                dates = v.get("dates", [])
                opens   = v.get("open", [])
                highs   = v.get("high", [])
                lows    = v.get("low", [])
                closes  = v.get("close", [])
                volumes = v.get("volume", [])
                result[t] = [
                    {"date": d, "open": o, "high": h, "low": l, "close": c, "volume": vol}
                    for d, o, h, l, c, vol in zip(dates, opens, highs, lows, closes, volumes)
                ]

    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        # 差分取得: 既存データがある場合は最終日の翌日から
        if ticker in result and result[ticker]:
            last_date = result[ticker][-1]["date"][:10]
            fetch_start = (
                datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            is_incremental = True
        else:
            fetch_start = start_str
            is_incremental = False

        print(f"  [{i+1}/{len(tickers)}] {ticker} {'(差分)' if is_incremental else ''}", end=" ", flush=True)
        try:
            bars = _fetch_bars(ticker, multiplier, timespan, fetch_start, end_str)
            if not bars:
                if is_incremental:
                    print("差分なし")
                else:
                    print("データなし")
                    failed.append(ticker)
                continue

            new_records = _bars_to_records(bars, timespan)
            if is_incremental and ticker in result:
                result[ticker].extend(new_records)
            else:
                result[ticker] = new_records

            total = len(result[ticker])
            print(f"{len(bars)}本取得 (累計 {total:,}本)")
            time.sleep(0.12)  # ~8 req/sec でレートリミット対策

        except PermissionError as e:
            print(f"アクセス制限: {e}")
            failed.append(ticker)
        except Exception as e:
            print(f"エラー: {e}")
            failed.append(ticker)

    if failed:
        print(f"\n[fetch_polygon] 取得失敗 ({len(failed)}銘柄): {failed}")

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    )
    total_bars = sum(len(v) for v in result.values())
    print(f"\n[fetch_polygon] 保存完了: {output_path}  {len(result)}銘柄  {total_bars:,}本")
    return result


if __name__ == "__main__":
    import argparse
    from backtest.config import (
        TICKERS_DAY, TICKERS_SWING,
        DAY_LOOKBACK_YEARS, SWING_LOOKBACK_YEARS,
        PRICE_DATA_DAY, PRICE_DATA_SWING,
        DAY_TIMEFRAME, SWING_TIMEFRAME,
    )

    parser = argparse.ArgumentParser(description="Polygon.io 株価データ取得")
    parser.add_argument("--mode", choices=["day", "swing", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("day", "both"):
        existing_day = json.loads(PRICE_DATA_DAY.read_text()) if PRICE_DATA_DAY.exists() else {}
        fetch_stock_data(
            TICKERS_DAY, timeframe=DAY_TIMEFRAME,
            lookback_years=DAY_LOOKBACK_YEARS,
            output_path=PRICE_DATA_DAY,
            existing=existing_day,
        )

    if args.mode in ("swing", "both"):
        existing_swing = json.loads(PRICE_DATA_SWING.read_text()) if PRICE_DATA_SWING.exists() else {}
        fetch_stock_data(
            TICKERS_SWING, timeframe=SWING_TIMEFRAME,
            lookback_years=SWING_LOOKBACK_YEARS,
            output_path=PRICE_DATA_SWING,
            existing=existing_swing,
        )
