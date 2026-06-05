"""
fetch_alpaca.py — Alpaca API から株価データを取得

デイトレ: 10Min足 2年, 50銘柄
スイング: 1Day足 10年, 50銘柄 (有料プランで1H足も可)

環境変数:
  ALPACA_API_KEY    : Alpaca API キー
  ALPACA_API_SECRET : Alpaca API シークレット

依存: alpaca-py (pip install alpaca-py)
      または requests (フォールバック)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

HERE = Path(__file__).parent.parent  # backtest/


# ── Alpaca REST クライアント ──────────────────────────────────────────────────

BASE_URL = "https://data.alpaca.markets/v2"

def _get_headers() -> dict[str, str]:
    key    = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        raise EnvironmentError(
            "ALPACA_API_KEY と ALPACA_API_SECRET 環境変数を設定してください。\n"
            "Alpaca アカウント: https://alpaca.markets/"
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _fetch_bars(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    limit: int = 10000,
) -> list[dict]:
    """Alpaca Bars API からページネーション付きで全バーを取得する。"""
    try:
        import requests
    except ImportError:
        raise ImportError("requests が必要です: pip install requests")

    headers  = _get_headers()
    url      = f"{BASE_URL}/stocks/{symbol}/bars"
    all_bars: list[dict] = []
    page_token: str | None = None

    while True:
        params: dict = {
            "timeframe": timeframe,
            "start":     start,
            "end":       end,
            "limit":     limit,
            "feed":      "iex",   # 無料プランは iex, 有料は sip
            "sort":      "asc",
        }
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:  # レートリミット
            time.sleep(60)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Alpaca API エラー {resp.status_code}: {resp.text[:200]}")

        data       = resp.json()
        bars       = data.get("bars", [])
        all_bars.extend(bars)
        page_token = data.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.2)  # API レートリミット対策

    return all_bars


def _bars_to_dict(bars: list[dict]) -> dict:
    """Alpaca バー一覧を {dates, open, high, low, close, volume} 形式に変換する。"""
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for b in bars:
        t = b["t"]
        # タイムスタンプ正規化 (ISO8601 → YYYY-MM-DD HH:MM)
        if "T" in t:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            dates.append(dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        else:
            dates.append(t[:10])
        opens.append(round(float(b["o"]), 4))
        highs.append(round(float(b["h"]), 4))
        lows.append(round(float(b["l"]), 4))
        closes.append(round(float(b["c"]), 4))
        volumes.append(int(b.get("v", 0)))
    return {
        "dates":  dates,
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }


def fetch_stock_data(
    tickers: list[str],
    timeframe: Literal["1Min", "5Min", "10Min", "15Min", "30Min", "1Hour", "1Day"] = "10Min",
    lookback_years: int = 2,
    output_path: Path | None = None,
    existing: dict | None = None,
    feed: str = "iex",
) -> dict:
    """
    Alpaca API から株価データを取得して JSON に保存する。

    Parameters
    ----------
    tickers : list[str]
        取得する銘柄リスト
    timeframe : str
        足種 ("10Min", "1Hour", "1Day" など)
    lookback_years : int
        遡る年数
    output_path : Path, optional
        保存先。None の場合は timeframe に応じてデフォルトパスを選択。
    existing : dict, optional
        既存データ。これに差分取得してマージする。
    feed : str
        "iex" (無料) or "sip" (有料, 拡張ヒストリー)

    Returns
    -------
    dict
        {ticker: {dates, open, high, low, close, volume}}
    """
    if output_path is None:
        if timeframe in ("10Min", "5Min", "15Min"):
            output_path = HERE / "price_data_intraday.json"
        elif timeframe in ("1Hour",):
            output_path = HERE / "price_data_1h.json"
        else:
            output_path = HERE / "price_data.json"

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_years * 365 + 30)
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str   = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"[fetch_alpaca] timeframe={timeframe}  期間: {start_str[:10]} 〜 {end_str[:10]}")
    print(f"[fetch_alpaca] feed={feed}  {len(tickers)}銘柄")

    result: dict = dict(existing) if existing else {}
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        # 既存データがあれば差分取得（最終日以降のみ）
        if ticker in result and result[ticker].get("dates"):
            last_date = result[ticker]["dates"][-1][:10]
            # 既存最終日+1日から取得
            diff_start = (
                datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%dT00:00:00Z")
            fetch_start = diff_start
            is_incremental = True
        else:
            fetch_start    = start_str
            is_incremental = False

        print(f"  [{i+1}/{len(tickers)}] {ticker} ...", end=" ", flush=True)
        try:
            bars = _fetch_bars(ticker, timeframe, fetch_start, end_str)
            if not bars:
                if is_incremental:
                    print("差分なし (最新)")
                else:
                    print("データなし")
                    failed.append(ticker)
                continue

            new_data = _bars_to_dict(bars)
            if is_incremental and ticker in result:
                # 既存データに追記
                for key in ("dates", "open", "high", "low", "close", "volume"):
                    result[ticker][key].extend(new_data[key])
            else:
                result[ticker] = new_data

            print(f"{len(bars)}本 (累計 {len(result[ticker]['dates'])}本)")
            time.sleep(0.1)

        except Exception as e:
            print(f"エラー: {e}")
            failed.append(ticker)

    if failed:
        print(f"\n[fetch_alpaca] 取得失敗: {failed}")

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    )
    total = sum(len(v["dates"]) for v in result.values())
    print(f"\n[fetch_alpaca] 保存完了: {output_path}  {len(result)}銘柄  {total:,}本")
    return result


if __name__ == "__main__":
    import argparse
    from backtest.config import (
        TICKERS_DAY, TICKERS_SWING,
        DAY_LOOKBACK_YEARS, SWING_LOOKBACK_YEARS,
        PRICE_DATA_DAY, PRICE_DATA_SWING,
        DAY_TIMEFRAME, SWING_TIMEFRAME,
    )

    parser = argparse.ArgumentParser(description="Alpaca 株価データ取得")
    parser.add_argument("--mode", choices=["day", "swing", "both"], default="both")
    parser.add_argument("--feed", default="iex", choices=["iex", "sip"])
    args = parser.parse_args()

    if args.mode in ("day", "both"):
        existing_day = json.loads(PRICE_DATA_DAY.read_text()) if PRICE_DATA_DAY.exists() else {}
        fetch_stock_data(
            TICKERS_DAY, timeframe=DAY_TIMEFRAME,
            lookback_years=DAY_LOOKBACK_YEARS,
            output_path=PRICE_DATA_DAY,
            existing=existing_day, feed=args.feed,
        )

    if args.mode in ("swing", "both"):
        existing_swing = json.loads(PRICE_DATA_SWING.read_text()) if PRICE_DATA_SWING.exists() else {}
        fetch_stock_data(
            TICKERS_SWING, timeframe=SWING_TIMEFRAME,
            lookback_years=SWING_LOOKBACK_YEARS,
            output_path=PRICE_DATA_SWING,
            existing=existing_swing, feed=args.feed,
        )
