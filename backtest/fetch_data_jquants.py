#!/usr/bin/env python3
"""
J-Quants API で JP株 1分足データを取得し、10分足に集約して保存するスクリプト。

前提:
  - J-Quants Light プラン (¥1,650/月) + 分足オプション (¥5,500/月) に加入済みであること
  - 環境変数 JQUANTS_EMAIL / JQUANTS_PASSWORD を設定すること

Usage:
  pip install requests
  python backtest/fetch_data_jquants.py

Output:
  backtest/price_data_10min_jp.json

取得銘柄 (12社):
  半導体・電子部品 (10社): 8035, 6857, 6723, 4063, 6963, 6920, 6146, 6981, 6762, 6902
  重工業・核融合  (2社):  6501, 7011

所要時間:
  約90〜120分 (12銘柄 × 約500取引日 × 2セッション × 0.4秒/call)
  チェックポイントを使って途中から再開可能。
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests が未インストールです。'pip install requests' を実行してください。")
    sys.exit(1)

HERE       = Path(__file__).parent
OUT_FILE   = HERE / "price_data_10min_jp.json"
CHECKPOINT = HERE / "_jq_checkpoint.json"

BASE_URL = "https://api.jquants.com/v1"

JP_TICKERS = [
    "8035.T",  # 東京エレクトロン
    "6857.T",  # アドバンテスト
    "6723.T",  # ルネサスエレクトロニクス
    "4063.T",  # 信越化学工業
    "6963.T",  # ローム
    "6920.T",  # レーザーテック
    "6146.T",  # ディスコ
    "6981.T",  # 村田製作所
    "6762.T",  # TDK
    "6902.T",  # デンソー
    "6501.T",  # 日立製作所
    "7011.T",  # 三菱重工業
]
JP_CODES = [t.replace(".T", "") for t in JP_TICKERS]
CODE_TO_TICKER = {t.replace(".T", ""): t for t in JP_TICKERS}

# ── 認証 ────────────────────────────────────────────────────────────────────

def get_id_token(email: str, password: str) -> str:
    """J-Quants API の idToken を取得する (refresh → id の2ステップ)。"""
    r = requests.post(
        f"{BASE_URL}/token/auth_user",
        json={"mailaddress": email, "password": password},
        timeout=30,
    )
    r.raise_for_status()
    refresh_token = r.json()["refreshToken"]

    r2 = requests.post(
        f"{BASE_URL}/token/auth_refresh",
        params={"refreshtoken": refresh_token},
        timeout=30,
    )
    r2.raise_for_status()
    return r2.json()["idToken"]


# ── 取引カレンダー取得 ────────────────────────────────────────────────────────

def get_trading_days(start: str, end: str, id_token: str) -> list[str]:
    """
    J-Quants の取引カレンダーから東証営業日リストを取得する。
    start/end: "YYYY-MM-DD" 形式
    戻り値:   ["YYYYMMDD", ...] 形式のリスト
    """
    headers = {"Authorization": f"Bearer {id_token}"}
    r = requests.get(
        f"{BASE_URL}/markets/trading_calendar",
        params={"from": start, "to": end},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    calendar = r.json().get("trading_calendar", [])
    # HolidayDivision == "1" が通常営業日
    return [
        d["Date"].replace("-", "")
        for d in calendar
        if d.get("HolidayDivision") == "1"
    ]


# ── 1分足取得 ────────────────────────────────────────────────────────────────

def _fetch_session(
    code: str, date_str: str, session: str, headers: dict
) -> list[dict]:
    """
    1日分の1分足データを1セッション分取得する (ページネーション対応)。

    session  : "prices_am_intraday" | "prices_pm_intraday"
    date_str : "YYYYMMDD"
    """
    bars: list[dict] = []
    pagination_key: str | None = None

    while True:
        params: dict = {"code": code, "date": date_str}
        if pagination_key:
            params["pagination_key"] = pagination_key

        for attempt in range(3):
            try:
                r = requests.get(
                    f"{BASE_URL}/prices/{session}",
                    params=params,
                    headers=headers,
                    timeout=30,
                )
            except requests.RequestException as e:
                if attempt == 2:
                    print(f"    {code} {date_str} {session}: 接続エラー {e}")
                    return bars
                time.sleep(5 * (attempt + 1))
                continue

            if r.status_code == 200:
                break
            if r.status_code == 404:
                return bars  # その日はデータなし (休場 or 停止)
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    レートリミット — {wait}秒待機中...")
                time.sleep(wait)
                continue
            if r.status_code == 401:
                # トークン期限切れはメイン側で検知するため伝播
                raise ValueError("idToken expired")
            # その他のエラー
            if attempt == 2:
                print(f"    {code} {date_str} {session}: HTTP {r.status_code}")
                return bars
            time.sleep(5)
        else:
            return bars

        data = r.json()
        bars.extend(data.get(session, []))
        pagination_key = data.get("pagination_key")
        if not pagination_key:
            break
        time.sleep(0.2)

    return bars


def fetch_minute_bars_day(code: str, date_str: str, headers: dict) -> list[dict]:
    """AM + PM セッションを結合した1日分の1分足データを返す。"""
    am = _fetch_session(code, date_str, "prices_am_intraday", headers)
    time.sleep(0.2)
    pm = _fetch_session(code, date_str, "prices_pm_intraday", headers)
    return am + pm


# ── 10分足への集約 ────────────────────────────────────────────────────────────

def parse_bar_dt(bar: dict) -> str | None:
    """
    J-Quants 1分足バーから ISO 形式 datetime 文字列を生成する。
    対応フォーマット:
      - {"Date": "20240603", "Time": "09:00"}
      - {"Date": "2024-06-03", "Time": "09:00:00"}
      - {"DateTime": "2024-06-03 09:00:00"}
    戻り値: "2024-06-03T09:00" 形式、またはNone
    """
    if "DateTime" in bar:
        dt = str(bar["DateTime"])
        return dt[:10].replace("/", "-") + "T" + dt[11:16]

    d = str(bar.get("Date", ""))
    t = str(bar.get("Time", ""))
    if not d or not t:
        return None

    # Date を YYYY-MM-DD に正規化
    d_clean = d.replace("-", "")
    if len(d_clean) == 8:
        d_fmt = f"{d_clean[:4]}-{d_clean[4:6]}-{d_clean[6:]}"
    else:
        return None

    # Time を HH:MM に正規化
    t_clean = t.replace(":", "")
    if len(t_clean) >= 4:
        t_fmt = f"{t_clean[:2]}:{t_clean[2:4]}"
    else:
        return None

    return f"{d_fmt}T{t_fmt}"


def to_10min_bars(minute_bars: list[dict]) -> list[dict]:
    """
    1分足リストを10分足に集約する。
    split調整後の価格 (AdjustmentClose 等) があれば優先して使用。
    """
    if not minute_bars:
        return []

    def get_ohlcv(bar: dict) -> tuple[float, float, float, float, float] | None:
        # split調整済み価格を優先
        def v(adj_key: str, raw_key: str) -> float | None:
            val = bar.get(adj_key) or bar.get(raw_key) or bar.get(raw_key.lower())
            return float(val) if val is not None else None

        o = v("AdjustmentOpen",   "Open")
        h = v("AdjustmentHigh",   "High")
        lo = v("AdjustmentLow",   "Low")
        c = v("AdjustmentClose",  "Close")
        vol = float(bar.get("AdjustmentVolume") or bar.get("Volume") or bar.get("volume") or 0)
        if c is None:
            return None
        return (o or c, h or c, lo or c, c, vol)

    # 10分バケットに振り分け
    buckets: dict[str, list] = defaultdict(list)
    for bar in minute_bars:
        dt = parse_bar_dt(bar)
        if dt is None:
            continue
        minute_int = int(dt[14:16])
        bucket_min = (minute_int // 10) * 10
        key = f"{dt[:13]}:{bucket_min:02d}"
        ohlcv = get_ohlcv(bar)
        if ohlcv:
            buckets[key].append(ohlcv)

    result: list[dict] = []
    for key in sorted(buckets.keys()):
        group = buckets[key]
        opens  = [x[0] for x in group]
        highs  = [x[1] for x in group]
        lows   = [x[2] for x in group]
        closes = [x[3] for x in group]
        vols   = [x[4] for x in group]
        result.append({
            "date":   key,
            "open":   round(opens[0],   4),
            "high":   round(max(highs), 4),
            "low":    round(min(lows),  4),
            "close":  round(closes[-1], 4),
            "volume": int(sum(vols)),
        })
    return result


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    email    = os.environ.get("JQUANTS_EMAIL",    "")
    password = os.environ.get("JQUANTS_PASSWORD", "")
    if not email or not password:
        print("ERROR: 環境変数 JQUANTS_EMAIL / JQUANTS_PASSWORD が未設定です")
        print("  export JQUANTS_EMAIL=your@email.com")
        print("  export JQUANTS_PASSWORD=yourpassword")
        sys.exit(1)

    print("━" * 60)
    print("J-Quants 認証中...")
    id_token   = get_id_token(email, password)
    token_time = time.time()
    print("  認証成功")

    # 過去2年の日付範囲
    today     = date.today()
    start_d   = today - timedelta(days=730)
    start_iso = start_d.strftime("%Y-%m-%d")
    end_iso   = today.strftime("%Y-%m-%d")

    print(f"\n取引カレンダー取得 ({start_iso} → {end_iso})...")
    trading_days = get_trading_days(start_iso, end_iso, id_token)
    print(f"  取引日数: {len(trading_days)}")

    # チェックポイントから再開
    checkpoint: dict = {}
    if CHECKPOINT.exists():
        try:
            checkpoint = json.loads(CHECKPOINT.read_text())
            n_done = sum(1 for k in checkpoint if not k.startswith("__"))
            print(f"\nチェックポイント読み込み: {n_done} 件処理済み (続きから再開)")
        except Exception:
            checkpoint = {}

    # バッファ: {ticker: [10min_bars...]}
    result: dict[str, list] = {t: [] for t in JP_TICKERS}

    total = len(JP_CODES) * len(trading_days)
    done  = 0

    print(f"\nデータ取得開始 ({len(JP_CODES)}銘柄 × {len(trading_days)}日 = {total}タスク)")
    print("━" * 60)

    for date_str in trading_days:
        for code in JP_CODES:
            key    = f"{code}_{date_str}"
            done  += 1

            # idToken は23時間で期限切れ → 定期更新
            if time.time() - token_time > 82_800:
                print("\n  idToken 更新中...")
                id_token   = get_id_token(email, password)
                token_time = time.time()

            headers = {"Authorization": f"Bearer {id_token}"}

            if key in checkpoint:
                bars_10min = checkpoint[key]
            else:
                try:
                    bars_1min  = fetch_minute_bars_day(code, date_str, headers)
                    bars_10min = to_10min_bars(bars_1min)
                except ValueError as e:
                    if "expired" in str(e):
                        print("\n  idToken 期限切れ → 再認証中...")
                        id_token   = get_id_token(email, password)
                        token_time = time.time()
                        headers    = {"Authorization": f"Bearer {id_token}"}
                        bars_1min  = fetch_minute_bars_day(code, date_str, headers)
                        bars_10min = to_10min_bars(bars_1min)
                    else:
                        raise

                checkpoint[key] = bars_10min
                time.sleep(0.4)  # ~2.5 req/sec (2 API calls per key)

            ticker = CODE_TO_TICKER[code]
            result[ticker].extend(bars_10min)

            if done % 200 == 0:
                pct = 100.0 * done / total
                print(f"  進捗: {done}/{total} ({pct:.1f}%) — 最終: {code} {date_str}")
                CHECKPOINT.write_text(json.dumps(checkpoint))

    # 最終チェックポイント保存
    CHECKPOINT.write_text(json.dumps(checkpoint))

    # 結果確認
    print("\n━" * 60)
    print("取得結果:")
    for ticker in JP_TICKERS:
        bars = result[ticker]
        if bars:
            first = bars[0]["date"][:10]
            last  = bars[-1]["date"][:10]
            print(f"  {ticker}: {len(bars)} 10min bars ({first} → {last})")
        else:
            print(f"  {ticker}: 0 bars (取得失敗 — コードや日付範囲を確認)")

    # 空の銘柄を除外して保存
    result_clean = {t: v for t, v in result.items() if len(v) >= 100}

    if not result_clean:
        print("\nERROR: データ取得に失敗しました。APIキー・プランを確認してください。")
        sys.exit(1)

    OUT_FILE.write_text(json.dumps(result_clean, ensure_ascii=False))
    print(f"\n保存完了 → {OUT_FILE.name}")
    print(f"  取得銘柄数: {len(result_clean)}/{len(JP_TICKERS)}")

    # チェックポイント削除
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("  チェックポイントファイルを削除しました")

    print("\n完了！このデータを保存したら J-Quants を解約できます。")


if __name__ == "__main__":
    main()
