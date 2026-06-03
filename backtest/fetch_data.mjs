/**
 * Fetch real OHLCV data for backtest using yahoo-finance2 (Node.js).
 *
 * Usage:
 *   node backtest/fetch_data.mjs              # daily (2015-2025) → price_data.json
 *   node backtest/fetch_data.mjs --intraday   # 1h bars → price_data_intraday.json
 *
 * Intraday data source priority:
 *   1. Alpaca Markets (env: ALPACA_API_KEY + ALPACA_API_SECRET) — US株のみ, 最大10年
 *   2. Yahoo Finance フォールバック — 全銘柄(日本株含む), 最大730日
 */
import { createRequire } from "module";
import { writeFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const require = createRequire(import.meta.url);
const YFClass = require("yahoo-finance2").default;
const yf = new YFClass();

const INTRADAY = process.argv.includes("--intraday");

// Alpaca credentials (GitHub Secrets 経由で設定)
const ALPACA_KEY    = process.env.ALPACA_API_KEY    ?? "";
const ALPACA_SECRET = process.env.ALPACA_API_SECRET ?? "";
const USE_ALPACA    = INTRADAY && !!ALPACA_KEY && !!ALPACA_SECRET;

// Alpaca で取得できる最古の日付 (IEX feed)
const ALPACA_START = "2016-01-01";

// 47 tickers — semiconductor / AI / cloud tech focus
const TICKERS = [
  // ── US 大型半導体・AI ──────────────────────────────────────────
  "NVDA",    // NVIDIA (GPU/AI)
  "ASML",    // ASML (半導体露光装置)
  "MSFT",    // Microsoft (クラウド/AI)
  "AMD",     // AMD (CPU/GPU)
  "AVGO",    // Broadcom (半導体/ネットワーク)
  "QCOM",    // Qualcomm (モバイル半導体)
  "MU",      // Micron Technology (DRAM/NAND)
  "TSM",     // TSMC ADR (ファウンドリ)
  "ARM",     // Arm Holdings (チップIP, IPO 2023)
  "INTC",    // Intel Corp
  "TXN",     // Texas Instruments
  // ── US 製造装置・EDA ─────────────────────────────────────────
  "AMAT",    // Applied Materials
  "LRCX",    // Lam Research
  "KLAC",    // KLA Corp
  "ENTG",    // Entegris (半導体材料・クリーニング)
  "ACLS",    // Axcelis Technologies (イオン注入装置)
  // ── US アナログ・混合信号半導体 ──────────────────────────────
  "ADI",     // Analog Devices
  "NXPI",    // NXP Semiconductors (自動車/IoT)
  "MCHP",    // Microchip Technology (マイコン)
  "ON",      // ON Semiconductor (パワー半導体)
  "MPWR",    // Monolithic Power Systems
  "SWKS",    // Skyworks Solutions (RF半導体)
  // ── US AI・クラウド成長株 ─────────────────────────────────────
  "MRVL",    // Marvell Technology
  "CRWD",    // CrowdStrike (セキュリティ)
  "PLTR",    // Palantir (AI/データ分析)
  "SMCI",    // Super Micro Computer
  "META",    // Meta Platforms (AI/メタバース)
  "GOOGL",   // Alphabet (AI/クラウド)
  "AMZN",    // Amazon (AWS/AI)
  "ORCL",    // Oracle (クラウドDB/AI)
  "ANET",    // Arista Networks (データセンターネットワーク)
  "NOW",     // ServiceNow (AI/エンタープライズSaaS)
  // ── US セキュリティ・データ ───────────────────────────────────
  "PANW",    // Palo Alto Networks (AI-SecOps)
  "DDOG",    // Datadog (可観測性/AI)
  "ZS",      // Zscaler (ゼロトラスト)
  // ── US 小型・高ボラ ───────────────────────────────────────────
  "SOUN",    // SoundHound AI (SPAC 2022)
  "IONQ",    // IonQ (量子コンピュータ, SPAC 2021)
  // ── JP 半導体・電子部品 ────────────────────────────────────────
  "8035.T",  // 東京エレクトロン
  "6857.T",  // アドバンテスト
  "6723.T",  // ルネサスエレクトロニクス
  "4063.T",  // 信越化学工業
  "6963.T",  // ローム
  "6920.T",  // レーザーテック
  "6146.T",  // ディスコ
  "6981.T",  // 村田製作所 (電子部品)
  "6762.T",  // TDK (電子部品)
  "6902.T",  // デンソー (車載半導体)
];

// Daily: 10 years of history
const DAILY_START = "2015-01-01";
const DAILY_END   = "2025-12-31";

// IPO/上場日フィルタ: この日付以前のデータはSPAC前別会社データのため除外
const IPO_FILTER = {
  "SOUN": "2022-04-13",  // SoundHound AI (SPAC merger)
  "IONQ": "2021-10-01",  // IonQ (SPAC merger)
  "ARM":  "2023-09-14",  // Arm Holdings IPO
};

// Intraday 1h: Yahoo Finance allows up to 730 days back
// Use 2 years ago from today minus 1 week safety margin
function getIntradayRange() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 720); // ~2 years, safely within 730-day limit
  return {
    start: start.toISOString().split("T")[0],
    end:   end.toISOString().split("T")[0],
  };
}

async function fetchTickerDaily(ticker) {
  console.log(`  Fetching ${ticker} (daily) ...`);
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
      .filter(r => r.date >= ipoStart);  // IPO前のSPACデータを除外
    console.log(`    ${ticker}: ${filtered.length} bars (${filtered[0]?.date} → ${filtered[filtered.length-1]?.date})`);
    return filtered;
  } catch (e) {
    console.error(`    ${ticker}: FAILED — ${e.message}`);
    return null;
  }
}

async function fetchTickerIntraday(ticker, start, end) {
  console.log(`  Fetching ${ticker} (1h) ...`);
  try {
    // yahoo-finance2 chart() supports intraday intervals
    const result = await yf.chart(ticker, {
      period1:  start,
      period2:  end,
      interval: "1h",
    });
    const quotes = result?.quotes ?? [];
    const filtered = quotes
      .filter(r => r.close != null && r.close > 0)
      .map(r => {
        // chart() returns Date objects; format as ISO datetime
        const dt = r.date instanceof Date ? r.date : new Date(r.date);
        return {
          date:   dt.toISOString(),  // keep full timestamp for intraday
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

// ── Alpaca Markets API (US株のみ, 1h足, 最大10年) ─────────────────────────────
async function fetchAlpacaIntraday(ticker) {
  const end = new Date().toISOString().split("T")[0];
  console.log(`  Fetching ${ticker} (Alpaca 1h, ${ALPACA_START} → ${end}) ...`);

  const allBars = [];
  let pageToken = null;

  do {
    const params = new URLSearchParams({
      timeframe: "1Hour",
      start:     ALPACA_START,
      end,
      feed:      "iex",   // 無料プラン (IEX Realtime Feed)
      limit:     "10000", // 1リクエスト最大10,000本
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

    // レート制限 (無料プラン ~200 req/min) を考慮して少し待つ
    if (pageToken) await new Promise(r => setTimeout(r, 300));
  } while (pageToken);

  if (allBars.length === 0) {
    console.log(`    ${ticker}: no data returned`);
    return null;
  }

  const rows = allBars.map(b => ({
    date:   b.t,                               // ISO 8601 UTC タイムスタンプ
    close:  Math.round(b.c * 10000) / 10000,
    open:   Math.round(b.o * 10000) / 10000,
    high:   Math.round(b.h * 10000) / 10000,
    low:    Math.round(b.l * 10000) / 10000,
    volume: b.v ?? 0,
  }));

  console.log(`    ${ticker}: ${rows.length} 1h bars (${rows[0]?.date?.slice(0,10)} → ${rows[rows.length-1]?.date?.slice(0,10)})`);
  return rows;
}

async function main() {
  const dir = path.dirname(fileURLToPath(import.meta.url));

  if (INTRADAY) {
    const result = {};

    if (USE_ALPACA) {
      // ── Alpaca: US株のみ (日本株は Alpaca 非対応なので除外) ──────────────
      const usTickers = TICKERS.filter(t => !t.endsWith(".T"));
      console.log(`Alpaca 1h足取得 (US株 ${usTickers.length}銘柄, ${ALPACA_START}〜) ...`);
      console.log(`※ 日本株 (.T) はAlpaca非対応のため今回スキップ\n`);

      for (const ticker of usTickers) {
        const data = await fetchAlpacaIntraday(ticker);
        if (data && data.length >= 300) {
          result[ticker] = data;
        } else {
          console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 300)`);
        }
      }
    } else {
      // ── Yahoo Finance フォールバック (全銘柄, 730日) ──────────────────────
      const { start, end } = getIntradayRange();
      console.log(`ALPACA_API_KEY 未設定 → Yahoo Finance フォールバック (${start} → ${end})`);
      console.log(`Fetching ${TICKERS.length} tickers — 1h bars ...\n`);

      for (const ticker of TICKERS) {
        const data = await fetchTickerIntraday(ticker, start, end);
        if (data && data.length >= 100) {
          result[ticker] = data;
        } else {
          console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
        }
      }
    }

    const out = path.join(dir, "price_data_intraday.json");
    writeFileSync(out, JSON.stringify(result, null, 0));
    console.log(`\nSaved ${Object.keys(result).length} tickers → price_data_intraday.json`);
    for (const [t, d] of Object.entries(result)) {
      console.log(`  ${t}: ${d.length} 1h bars`);
    }
  } else {
    console.log(`Fetching ${TICKERS.length} tickers (${DAILY_START} → ${DAILY_END}) ...`);
    const result = {};
    for (const ticker of TICKERS) {
      const data = await fetchTickerDaily(ticker);
      if (data && data.length >= 100) {
        result[ticker] = data;
      } else {
        console.log(`    ${ticker}: skipped (${data?.length ?? 0} bars < 100)`);
      }
    }

    // 全銘柄の最新の開始日を揃える（時系列アライメント）
    // 各銘柄の最初の日付を取得し、最も遅い日付を共通開始日とする
    if (Object.keys(result).length > 0) {
      const firstDates = Object.entries(result).map(([t, d]) => ({ ticker: t, date: d[0].date }));
      firstDates.sort((a, b) => a.date < b.date ? 1 : -1);
      const commonStart = firstDates[0].date;  // 最も遅い開始日
      console.log(`\n時系列アライメント: 共通開始日 = ${commonStart} (${firstDates[0].ticker} のIPO/開始日)`);
      for (const ticker of Object.keys(result)) {
        const before = result[ticker].length;
        result[ticker] = result[ticker].filter(r => r.date >= commonStart);
        if (before !== result[ticker].length)
          console.log(`  ${ticker}: ${before} → ${result[ticker].length} bars (${commonStart}以前を除外)`);
        if (result[ticker].length < 300) {
          console.log(`  ${ticker}: skipped after alignment (${result[ticker].length} bars < 300)`);
          delete result[ticker];
        }
      }
    }

    const out = path.join(dir, "price_data.json");
    writeFileSync(out, JSON.stringify(result, null, 0));
    console.log(`\nSaved ${Object.keys(result).length} tickers → price_data.json`);
    for (const [t, d] of Object.entries(result)) {
      console.log(`  ${t}: ${d.length} bars`);
    }
  }
}

main().catch(console.error);
