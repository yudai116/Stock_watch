/**
 * 株価データ取得スクリプト
 *
 * ■ データ取得先と保存先
 *   node backtest/fetch_data.mjs
 *     → 日足 (2015〜) を Yahoo Finance から取得
 *     → backtest/price_data.json に保存
 *     → スイングトレードバックテストが参照
 *
 *   node backtest/fetch_data.mjs --intraday
 *     → 1時間足を取得 (スイングトレード用)
 *     → US株: Alpaca Markets (最大10年, ALPACA_API_KEY が必要)
 *            ALPACA_API_KEY 未設定時は Yahoo Finance フォールバック (最大730日)
 *     → JP株: Yahoo Finance (最大730日)
 *     → backtest/price_data_intraday.json に保存
 *     → スイングトレードバックテストが参照
 *
 *   node backtest/fetch_data.mjs --day
 *     → 10分足を取得 (デイトレード用, US株のみ)
 *     → US株: Alpaca Markets (最大10年, ALPACA_API_KEY が必要)
 *            ALPACA_API_KEY 未設定時は Yahoo Finance フォールバック (最大60日のみ)
 *     → JP株: 非対応 (Yahoo Finance 10分足は最大60日のみのため除外)
 *     → backtest/price_data_10min.json に保存 (.gitignore対象、大容量)
 *     → デイトレードバックテストが参照
 *
 * ■ ネットワーク注意
 *   Yahoo Finance / Alpaca はローカル開発環境ではネットワーク制限で
 *   ブロックされる場合があります。GitHub Actions 上では制限なく動作します。
 *   データ取得は GitHub Actions (backtest.yml) 経由で実行してください。
 *
 * ■ 銘柄数: 50社 (US 38社 + JP 12社)
 *   - 半導体 (US 15社 + JP 10社)
 *   - AI/クラウド/セキュリティ (US 13社)
 *   - 核融合・原子力・エネルギー (US 5社 + JP 2社)
 *   - 高ボラ/新興 (US 5社)
 */
import { createRequire } from "module";
import { writeFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass();

const INTRADAY = process.argv.includes("--intraday");
const DAY      = process.argv.includes("--day");

// Alpaca credentials (GitHub Secrets 経由で設定)
const ALPACA_KEY    = process.env.ALPACA_API_KEY    ?? "";
const ALPACA_SECRET = process.env.ALPACA_API_SECRET ?? "";
const USE_ALPACA    = (INTRADAY || DAY) && !!ALPACA_KEY && !!ALPACA_SECRET;

// Alpaca IEX feed の最古取得可能日
const ALPACA_START = "2016-01-01";

// ── 50銘柄リスト ────────────────────────────────────────────────────────────
const US_TICKERS = [
  // ── 半導体 (15社) ─────────────────────────────────────────────────────────
  "NVDA",    // NVIDIA — GPU/AI加速器
  "AMD",     // Advanced Micro Devices — CPU/GPU
  "INTC",    // Intel — CPU/ファウンドリ転換中
  "TSM",     // TSMC ADR — 世界最大ファウンドリ
  "ASML",    // ASML — EUV露光装置独占
  "AMAT",    // Applied Materials — 成膜・エッチング装置
  "LRCX",    // Lam Research — エッチング装置
  "KLAC",    // KLA Corp — 検査・計測装置
  "MU",      // Micron Technology — DRAM/NAND
  "TXN",     // Texas Instruments — アナログ半導体
  "ADI",     // Analog Devices — 混合信号半導体
  "AVGO",    // Broadcom — ネットワーク/AI ASIC
  "QCOM",    // Qualcomm — モバイル/車載半導体
  "MRVL",    // Marvell Technology — データセンター半導体
  "MPWR",    // Monolithic Power Systems — パワー半導体

  // ── AI・クラウド・ソフトウェア (9社) ──────────────────────────────────────
  "MSFT",    // Microsoft — Azure/OpenAI
  "META",    // Meta Platforms — AI/メタバース
  "GOOGL",   // Alphabet — Gemini/GCP
  "AMZN",    // Amazon — AWS/AI
  "ORCL",    // Oracle — クラウドDB/AI
  "PLTR",    // Palantir — AI/データ分析
  "SMCI",    // Super Micro Computer — AIサーバー
  "ANET",    // Arista Networks — データセンターNW
  "NOW",     // ServiceNow — エンタープライズAI

  // ── セキュリティ・データ (4社) ────────────────────────────────────────────
  "CRWD",    // CrowdStrike — AI-SecOps
  "PANW",    // Palo Alto Networks — ゼロトラスト
  "DDOG",    // Datadog — 可観測性/AI
  "ZS",      // Zscaler — クラウドセキュリティ

  // ── 核融合・原子力・エネルギー (5社) ─────────────────────────────────────
  "BWXT",    // BWX Technologies — 核燃料・原子炉コンポーネント製造
  "CCJ",     // Cameco — 世界最大級ウラン採掘
  "LEU",     // Centrus Energy — 濃縮ウラン (次世代炉向け HALEU)
  "CEG",     // Constellation Energy — 米最大原子力発電事業者 (2022年スピンオフ)
  "VST",     // Vistra — 原子力・天然ガス発電 (AI電力需要受益)

  // ── 高ボラティリティ・新興 (5社) ─────────────────────────────────────────
  "SOUN",    // SoundHound AI — 音声AI (SPAC上場 2022)
  "IONQ",    // IonQ — 量子コンピュータ (SPAC上場 2021)
  "NXPI",    // NXP Semiconductors — 車載/IoT半導体
  "ON",      // ON Semiconductor — パワー半導体 (EV向け)
  "ACLS",    // Axcelis Technologies — イオン注入装置
];

const JP_TICKERS = [
  // ── 半導体・電子部品 (10社) ───────────────────────────────────────────────
  "8035.T",  // 東京エレクトロン — 半導体製造装置世界3位
  "6857.T",  // アドバンテスト — 半導体テスト装置世界首位
  "6723.T",  // ルネサスエレクトロニクス — 車載マイコン
  "4063.T",  // 信越化学工業 — シリコンウェーハ世界首位
  "6963.T",  // ローム — パワー半導体
  "6920.T",  // レーザーテック — EUVマスク欠陥検査装置独占
  "6146.T",  // ディスコ — ダイシング・研削装置
  "6981.T",  // 村田製作所 — 積層セラミックコンデンサ
  "6762.T",  // TDK — 電子部品・エネルギー
  "6902.T",  // デンソー — 車載半導体・EV部品

  // ── 核融合・重工業 (2社) ──────────────────────────────────────────────────
  "6501.T",  // 日立製作所 — 原子力プラント・社会インフラ (ITER参画)
  "7011.T",  // 三菱重工業 — 核融合炉コンポーネント (ITER真空容器)
];

const TICKERS = [...US_TICKERS, ...JP_TICKERS];

// ── IPOフィルタ ─────────────────────────────────────────────────────────────
// Yahoo Finance は SPAC上場前の「別会社」データを同一ティッカーで返すことがある。
// このフィルタにより実際の上場日以前のデータを除外する。
const IPO_FILTER = {
  "SOUN": "2022-04-26",  // SoundHound AI (SPAC merger with ATSP)
  "IONQ": "2021-10-01",  // IonQ (SPAC merger with dMY Technology IV)
  "CEG":  "2022-02-02",  // Constellation Energy (Exelon spin-off)
};

// ── 日足: 2015〜現在 ─────────────────────────────────────────────────────────
const DAILY_START = "2015-01-01";
const DAILY_END   = "2026-12-31";

// ── 1時間足: Yahoo Finance は最大730日 ────────────────────────────────────────
function getIntradayRange() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 720);
  return {
    start: start.toISOString().split("T")[0],
    end:   end.toISOString().split("T")[0],
  };
}

// ── 取得関数: Yahoo Finance 日足 ──────────────────────────────────────────────
async function fetchTickerDaily(ticker) {
  console.log(`  Fetching ${ticker} (daily, Yahoo Finance) ...`);
  try {
    const rows = await yf.historical(ticker, {
      period1:  DAILY_START,
      period2:  DAILY_END,
      interval: "1d",
    });
    const ipoStart = IPO_FILTER[ticker] ?? DAILY_START;
    const filtered = rows
      .filter(r => r.close != null && r.close > 0)
      .map(r => ({
        date:   r.date.toISOString().split("T")[0],
        close:  Math.round(r.close  * 10000) / 10000,
        open:   Math.round((r.open  ?? r.close) * 10000) / 10000,
        high:   Math.round((r.high  ?? r.close) * 10000) / 10000,
        low:    Math.round((r.low   ?? r.close) * 10000) / 10000,
        volume: r.volume ?? 0,
      }))
      .filter(r => r.date >= ipoStart);
    console.log(`    ${ticker}: ${filtered.length} bars (${filtered[0]?.date} → ${filtered[filtered.length-1]?.date})`);
    return filtered;
  } catch (e) {
    console.error(`    ${ticker}: FAILED — ${e.message}`);
    return null;
  }
}

// ── 取得関数: Yahoo Finance 1時間足 ──────────────────────────────────────────
async function fetchTickerIntraday(ticker, start, end) {
  console.log(`  Fetching ${ticker} (1h, Yahoo Finance) ...`);
  try {
    const result = await yf.chart(ticker, {
      period1:  start,
      period2:  end,
      interval: "1h",
    });
    const quotes = result?.quotes ?? [];
    const filtered = quotes
      .filter(r => r.close != null && r.close > 0)
      .map(r => {
        const dt = r.date instanceof Date ? r.date : new Date(r.date);
        return {
          date:   dt.toISOString(),
          close:  Math.round(r.close  * 10000) / 10000,
          open:   Math.round((r.open  ?? r.close) * 10000) / 10000,
          high:   Math.round((r.high  ?? r.close) * 10000) / 10000,
          low:    Math.round((r.low   ?? r.close) * 10000) / 10000,
          volume: r.volume ?? 0,
        };
      });
    console.log(`    ${ticker}: ${filtered.length} 1h bars (${filtered[0]?.date?.slice(0,10)} → ${filtered[filtered.length-1]?.date?.slice(0,10)})`);
    return filtered;
  } catch (e) {
    console.error(`    ${ticker}: FAILED — ${e.message}`);
    return null;
  }
}

// ── 取得関数: Alpaca Markets 1時間足 (US株のみ、最大10年) ─────────────────────
async function fetchAlpacaIntraday(ticker) {
  const end = new Date().toISOString().split("T")[0];
  console.log(`  Fetching ${ticker} (1h, Alpaca ${ALPACA_START}→${end}) ...`);

  const allBars = [];
  let pageToken = null;
  do {
    const params = new URLSearchParams({
      timeframe: "1Hour",
      start:     ALPACA_START,
      end,
      feed:      "iex",
      limit:     "10000",
    });
    if (pageToken) params.set("page_token", pageToken);

    let res;
    try {
      res = await fetch(
        `https://data.alpaca.markets/v2/stocks/${ticker}/bars?${params}`,
        { headers: {
            "APCA-API-KEY-ID":     ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }},
      );
    } catch (e) {
      console.error(`    ${ticker}: network error — ${e.message}`);
      return null;
    }

    if (!res.ok) {
      const text = await res.text();
      console.error(`    ${ticker}: Alpaca ${res.status} — ${text.slice(0, 120)}`);
      return null;
    }

    const json = await res.json();
    allBars.push(...(json.bars ?? []));
    pageToken = json.next_page_token ?? null;
    if (pageToken) await new Promise(r => setTimeout(r, 300));
  } while (pageToken);

  if (allBars.length === 0) {
    console.log(`    ${ticker}: no data from Alpaca`);
    return null;
  }

  const rows = allBars.map(b => ({
    date:   b.t,
    close:  Math.round(b.c * 10000) / 10000,
    open:   Math.round(b.o * 10000) / 10000,
    high:   Math.round(b.h * 10000) / 10000,
    low:    Math.round(b.l * 10000) / 10000,
    volume: b.v ?? 0,
  }));

  console.log(`    ${ticker}: ${rows.length} 1h bars (${rows[0]?.date?.slice(0,10)} → ${rows[rows.length-1]?.date?.slice(0,10)})`);
  return rows;
}

// ── 取得関数: Alpaca Markets 10分足 (US株のみ、最大10年) ──────────────────────
async function fetchAlpacaDay(ticker) {
  const end = new Date().toISOString().split("T")[0];
  console.log(`  Fetching ${ticker} (10min, Alpaca ${ALPACA_START}→${end}) ...`);

  const allBars = [];
  let pageToken = null;
  do {
    const params = new URLSearchParams({
      timeframe: "10Min",
      start:     ALPACA_START,
      end,
      feed:      "iex",
      limit:     "10000",
    });
    if (pageToken) params.set("page_token", pageToken);

    let res;
    try {
      res = await fetch(
        `https://data.alpaca.markets/v2/stocks/${ticker}/bars?${params}`,
        { headers: {
            "APCA-API-KEY-ID":     ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }},
      );
    } catch (e) {
      console.error(`    ${ticker}: network error — ${e.message}`);
      return null;
    }

    if (!res.ok) {
      const text = await res.text();
      console.error(`    ${ticker}: Alpaca ${res.status} — ${text.slice(0, 120)}`);
      return null;
    }

    const json = await res.json();
    allBars.push(...(json.bars ?? []));
    pageToken = json.next_page_token ?? null;
    if (pageToken) await new Promise(r => setTimeout(r, 300));
  } while (pageToken);

  if (allBars.length === 0) {
    console.log(`    ${ticker}: no data from Alpaca`);
    return null;
  }

  const rows = allBars.map(b => ({
    date:   b.t,
    close:  Math.round(b.c * 10000) / 10000,
    open:   Math.round(b.o * 10000) / 10000,
    high:   Math.round(b.h * 10000) / 10000,
    low:    Math.round(b.l * 10000) / 10000,
    volume: b.v ?? 0,
  }));

  console.log(`    ${ticker}: ${rows.length} 10min bars (${rows[0]?.date?.slice(0,10)} → ${rows[rows.length-1]?.date?.slice(0,10)})`);
  return rows;
}

// ── メイン ────────────────────────────────────────────────────────────────────
async function main() {
  const dir = path.dirname(fileURLToPath(import.meta.url));

  if (DAY) {
    // ── 10分足: US=Alpaca(10年) → price_data_10min.json (JP非対応) ──────────
    const result = {};

    if (USE_ALPACA) {
      console.log(`\n[US株] Alpaca 10min足 (${US_TICKERS.length}社, 最大10年) ...`);
      for (const ticker of US_TICKERS) {
        const data = await fetchAlpacaDay(ticker);
        if (data && data.length >= 300) {
          result[ticker] = data;
        } else {
          console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 300)`);
        }
      }
    } else {
      const { start, end } = getIntradayRange();
      console.log(`\n[US株] ALPACA_API_KEY 未設定 → Yahoo Finance fallback (10min, ${start}〜${end}) ...`);
      console.log(`       ※ Yahoo Finance の10分足は最大60日のみ。バックテストには不十分です。`);
      for (const ticker of US_TICKERS) {
        try {
          const result_yf = await yf.chart(ticker, {
            period1:  start,
            period2:  end,
            interval: "10m",
          });
          const quotes = result_yf?.quotes ?? [];
          const filtered = quotes
            .filter(r => r.close != null && r.close > 0)
            .map(r => {
              const dt = r.date instanceof Date ? r.date : new Date(r.date);
              return {
                date:   dt.toISOString(),
                close:  Math.round(r.close  * 10000) / 10000,
                open:   Math.round((r.open  ?? r.close) * 10000) / 10000,
                high:   Math.round((r.high  ?? r.close) * 10000) / 10000,
                low:    Math.round((r.low   ?? r.close) * 10000) / 10000,
                volume: r.volume ?? 0,
              };
            });
          if (filtered.length >= 100) {
            result[ticker] = filtered;
            console.log(`    ${ticker}: ${filtered.length} 10min bars`);
          } else {
            console.log(`    ${ticker}: skipped (${filtered.length} bars < 100)`);
          }
        } catch (e) {
          console.error(`    ${ticker}: FAILED — ${e.message}`);
        }
      }
    }

    const out = path.join(dir, "price_data_10min.json");
    writeFileSync(out, JSON.stringify(result, null, 0));
    console.log(`\n保存完了 → price_data_10min.json`);
    console.log(`  取得銘柄数: ${Object.keys(result).length}/${US_TICKERS.length} (US株のみ)`);
    for (const [t, d] of Object.entries(result)) {
      const first = d[0]?.date?.slice(0, 10);
      const last  = d[d.length - 1]?.date?.slice(0, 10);
      console.log(`  ${t}: ${d.length} 10min bars (${first} → ${last})`);
    }

  } else if (INTRADAY) {
    // ── 1時間足: US=Alpaca(10年) / JP=Yahoo(730日) → price_data_intraday.json ──
    const result = {};

    // US株 ── Alpaca優先 (キーあり), なければ Yahoo Finance fallback
    const usTickers = US_TICKERS;
    if (USE_ALPACA) {
      console.log(`\n[US株] Alpaca 1h足 (${usTickers.length}社, 最大10年) ...`);
      for (const ticker of usTickers) {
        const data = await fetchAlpacaIntraday(ticker);
        if (data && data.length >= 300) {
          result[ticker] = data;
        } else {
          console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 300)`);
        }
      }
    } else {
      const { start, end } = getIntradayRange();
      console.log(`\n[US株] ALPACA_API_KEY 未設定 → Yahoo Finance fallback (${start}〜${end}) ...`);
      for (const ticker of usTickers) {
        const data = await fetchTickerIntraday(ticker, start, end);
        if (data && data.length >= 100) {
          result[ticker] = data;
        } else {
          console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
        }
      }
    }

    // JP株 ── Yahoo Finance (730日上限, Alpacaは非対応)
    const { start, end } = getIntradayRange();
    console.log(`\n[JP株] Yahoo Finance 1h足 (${JP_TICKERS.length}社, ${start}〜${end}) ...`);
    console.log(`       ※ Yahoo Finance のイントラデイは最大730日のみ取得可能`);
    for (const ticker of JP_TICKERS) {
      const data = await fetchTickerIntraday(ticker, start, end);
      if (data && data.length >= 100) {
        result[ticker] = data;
      } else {
        console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
      }
    }

    const out = path.join(dir, "price_data_intraday.json");
    writeFileSync(out, JSON.stringify(result, null, 0));
    console.log(`\n保存完了 → price_data_intraday.json`);
    console.log(`  取得銘柄数: ${Object.keys(result).length}/${TICKERS.length}`);
    const usCount = Object.keys(result).filter(t => !t.endsWith(".T")).length;
    const jpCount = Object.keys(result).filter(t => t.endsWith(".T")).length;
    console.log(`  US: ${usCount}社  JP: ${jpCount}社`);
    for (const [t, d] of Object.entries(result)) {
      console.log(`  ${t}: ${d.length} 1h bars`);
    }

  } else {
    // ── 日足: 全銘柄 Yahoo Finance (10年) → price_data.json ─────────────────
    console.log(`\n[全銘柄] Yahoo Finance 日足 (${TICKERS.length}社, ${DAILY_START}〜${DAILY_END}) ...`);
    const result = {};
    for (const ticker of TICKERS) {
      const data = await fetchTickerDaily(ticker);
      if (data && data.length >= 100) {
        result[ticker] = data;
      } else {
        console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
      }
    }

    // 全銘柄の時系列を共通開始日で揃える
    // (SPAC上場後の銘柄が最も遅い開始日になるので、それに全銘柄を合わせる)
    if (Object.keys(result).length > 0) {
      const firstDates = Object.entries(result).map(([t, d]) => ({ ticker: t, date: d[0].date }));
      firstDates.sort((a, b) => a.date < b.date ? 1 : -1);
      const latestTicker = firstDates[0];
      const commonStart  = latestTicker.date;
      console.log(`\n時系列アライメント: 共通開始日 = ${commonStart} (${latestTicker.ticker} の上場日)`);
      for (const ticker of Object.keys(result)) {
        const before = result[ticker].length;
        result[ticker] = result[ticker].filter(r => r.date >= commonStart);
        if (before !== result[ticker].length)
          console.log(`  ${ticker}: ${before} → ${result[ticker].length} bars`);
        if (result[ticker].length < 300) {
          console.log(`  ${ticker}: skipped after alignment (${result[ticker].length} bars < 300)`);
          delete result[ticker];
        }
      }
    }

    const out = path.join(dir, "price_data.json");
    writeFileSync(out, JSON.stringify(result, null, 0));
    console.log(`\n保存完了 → price_data.json`);
    console.log(`  取得銘柄数: ${Object.keys(result).length}/${TICKERS.length}`);
    const usCount = Object.keys(result).filter(t => !t.endsWith(".T")).length;
    const jpCount = Object.keys(result).filter(t => t.endsWith(".T")).length;
    console.log(`  US: ${usCount}社  JP: ${jpCount}社`);
    if (Object.keys(result).length > 0) {
      const sample = Object.values(result)[0];
      console.log(`  期間: ${sample[0].date} → ${sample[sample.length-1].date}`);
    }
  }
}

main().catch(console.error);
