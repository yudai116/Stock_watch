"""
fetch_macro.py — マクロ/レジーム用データ取得 (Yahoo Finance)

取得対象: ^VIX, HYG, LQD, SOXX, SMH, SPY の日足OHLCV
出力: backtest/macro_data.json
  {
    "^VIX": {"dates": [...], "open": [...], "high": [...], "low": [...],
              "close": [...], "volume": [...]},
    ...
  }

依存: yfinance
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent.parent  # backtest/

try:
    import yfinance as yf
except ImportError:
    print("yfinance が必要です: pip install yfinance", file=sys.stderr)
    sys.exit(1)


def fetch_macro_data(
    tickers: list[str],
    lookback_years: int = 12,
    output_path: Path | None = None,
) -> dict:
    """
    Yahoo Finance からマクロETFの日足データを取得する。

    Parameters
    ----------
    tickers : list[str]
        取得するティッカーリスト (例: ["^VIX", "HYG", "LQD", "SOXX", "SMH", "SPY"])
    lookback_years : int
        取得する過去年数
    output_path : Path, optional
        保存先パス。None の場合はデフォルトパスを使用。

    Returns
    -------
    dict
        {ticker: {dates, open, high, low, close, volume}}
    """
    if output_path is None:
        output_path = HERE / "macro_data.json"

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_years * 365 + 30)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    print(f"[fetch_macro] 取得期間: {start_str} 〜 {end_str}")
    print(f"[fetch_macro] ティッカー: {tickers}")

    result: dict = {}
    failed: list[str] = []

    for ticker in tickers:
        print(f"  {ticker} ...", end=" ", flush=True)
        try:
            df = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                print("データなし")
                failed.append(ticker)
                continue

            # マルチレベルカラム対応
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.droplevel(1)

            dates  = [d.strftime("%Y-%m-%d") for d in df.index]
            result[ticker] = {
                "dates":  dates,
                "open":   [round(float(v), 4) for v in df["Open"]],
                "high":   [round(float(v), 4) for v in df["High"]],
                "low":    [round(float(v), 4) for v in df["Low"]],
                "close":  [round(float(v), 4) for v in df["Close"]],
                "volume": [int(v) for v in df["Volume"]],
            }
            print(f"{len(dates)}本")
        except Exception as e:
            print(f"エラー: {e}")
            failed.append(ticker)

    if failed:
        print(f"\n[fetch_macro] 取得失敗: {failed}")

    # 保存
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    )
    total_bars = sum(len(v["dates"]) for v in result.values())
    print(f"\n[fetch_macro] 保存完了: {output_path}  {len(result)}銘柄  {total_bars}本")
    return result


def load_macro_data(path: Path | None = None) -> dict:
    """保存済みマクロデータを読み込む。"""
    if path is None:
        path = HERE / "macro_data.json"
    if not path.exists():
        raise FileNotFoundError(f"マクロデータが見つかりません: {path}\n`python -m backtest.data.fetch_macro` で取得してください。")
    return json.loads(path.read_text())


if __name__ == "__main__":
    import argparse
    from backtest.config import TICKERS_MACRO, MACRO_LOOKBACK_YEARS, MACRO_DATA_PATH

    parser = argparse.ArgumentParser(description="マクロETFデータ取得")
    parser.add_argument("--years", type=int, default=MACRO_LOOKBACK_YEARS)
    args = parser.parse_args()

    fetch_macro_data(TICKERS_MACRO, lookback_years=args.years, output_path=MACRO_DATA_PATH)
