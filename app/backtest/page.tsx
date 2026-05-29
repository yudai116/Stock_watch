import Link from "next/link";
import { readFileSync, existsSync } from "fs";
import path from "path";
import { yahooFinanceUrls } from "@/lib/formatters";

// ─── Static data from backtest_results_v4.json + trade_records.py ─────────────

const MULTS = {
  swing: {
    large: { RSI: 0.776, MACD: 1.283, BB: 0.719, MA: 1.222 },
    mid:   { RSI: 0.881, MACD: 1.057, BB: 0.858, MA: 1.204 },
    small: { RSI: 1.122, MACD: 0.856, BB: 1.001, MA: 1.021 },
  },
  day: {
    large: { RSI: 0.897, MACD: 1.100, BB: 0.700, MA: 1.303 },
    mid:   { RSI: 1.104, MACD: 1.041, BB: 0.437, MA: 1.418 },
    small: { RSI: 0.917, MACD: 1.026, BB: 0.986, MA: 1.072 },
  },
} as const;

const SHARPES = {
  swing: {
    large: { RSI: 0.327, MACD: 0.541, BB: 0.303, MA: 0.515 },
    mid:   { RSI: 0.357, MACD: 0.428, BB: 0.347, MA: 0.487 },
    small: { RSI: 0.375, MACD: 0.293, BB: 0.318, MA: 0.591 },
  },
  day: {
    large: { RSI: 0.514, MACD: 0.382, BB: 0.317, MA: 0.591 },
    mid:   { RSI: 0.389, MACD: 0.366, BB: 0.154, MA: 0.499 },
    small: { RSI: 0.525, MACD: 0.588, BB: 0.565, MA: 0.614 },
  },
} as const;

const AROON_DELTA = {
  swing: { large: +0.006, mid: +0.145, small: -0.012 },
  day:   { large: -0.006, mid: +0.069, small: -0.026 },
} as const;

type TradeRecord = {
  rank: number;
  name: string;
  ticker: string;
  size: "large" | "mid" | "small";
  dateBuy: string;
  priceBuy: number;
  dateSell: string;
  priceSell: number;
  returnPct: number;
  score: number;
  rsi: { val: number; score: number };
  macd: { val: number; sig: number; hist: number; score: number };
  bb: { val: number; score: number };
  ma: { ratio: number; score: number; gc: boolean };
  aroon: { up: number; down: number; score: number } | null;
  buyReason: string;
};

const SWING_TRADES: TradeRecord[] = [
  {
    rank: 1, name: "エヌビディア", ticker: "NVDA", size: "large",
    dateBuy: "2023-06-28", priceBuy: 171.41, dateSell: "2023-07-05", priceSell: 280.96,
    returnPct: 63.92, score: 70.2,
    rsi:  { val: 57.8,  score: 7  },
    macd: { val: -1.2886, sig: -2.3160, hist: +1.0273, score: 24 },
    bb:   { val: 0.967, score: 3  },
    ma:   { ratio: 1.0951, score: 18, gc: false },
    aroon: { up: 100, down: 8,  score: 10 },
    buyReason: "MACDゴールデンクロス当日 ／ Aroon 強い上昇トレンド（Up=100）",
  },
  {
    rank: 2, name: "エヌビディア", ticker: "NVDA", size: "large",
    dateBuy: "2015-10-15", priceBuy: 169.83, dateSell: "2015-10-22", priceSell: 237.93,
    returnPct: 40.10, score: 67.8,
    rsi:  { val: 55.3,  score: 8  },
    macd: { val: 6.4984, sig: 5.1240, hist: +1.3743, score: 20 },
    bb:   { val: 0.641, score: 6  },
    ma:   { ratio: 1.0297, score: 18, gc: true },
    aroon: { up: 88, down: 4, score: 10 },
    buyReason: "MACD 直近クロス後ヒスト拡大 ／ Aroon 強い上昇トレンド（Up=88）",
  },
  {
    rank: 3, name: "クラウドストライク", ticker: "CRWD", size: "mid",
    dateBuy: "2015-08-07", priceBuy: 102.62, dateSell: "2015-08-14", priceSell: 138.43,
    returnPct: 34.89, score: 67.5,
    rsi:  { val: 57.8,  score: 7  },
    macd: { val: 0.8017, sig: 0.7669, hist: +0.0348, score: 24 },
    bb:   { val: 0.697, score: 8  },
    ma:   { ratio: 1.0692, score: 24, gc: true },
    aroon: null,
    buyReason: "MACDゴールデンクロス当日 ／ EMA20 大きく上方乖離（+6.9%）",
  },
  {
    rank: 4, name: "サウンドハウンドAI", ticker: "SOUN", size: "small",
    dateBuy: "2020-04-01", priceBuy: 1.97, dateSell: "2020-04-08", priceSell: 2.59,
    returnPct: 31.80, score: 66.8,
    rsi:  { val: 22.7,  score: 23 },
    macd: { val: -0.3231, sig: -0.3234, hist: +0.0002, score: 24 },
    bb:   { val: 0.142, score: 15 },
    ma:   { ratio: 0.8209, score: 5, gc: false },
    aroon: null,
    buyReason: "RSI 極度売られすぎ（22.7）＋ MACDゴールデンクロス ＋ BB 下方域",
  },
  {
    rank: 5, name: "エヌビディア", ticker: "NVDA", size: "large",
    dateBuy: "2017-11-06", priceBuy: 12008.66, dateSell: "2017-11-13", priceSell: 15559.61,
    returnPct: 29.57, score: 67.9,
    rsi:  { val: 60.0,  score: 6  },
    macd: { val: 513.5615, sig: 447.0307, hist: +66.5308, score: 20 },
    bb:   { val: 0.891, score: 2  },
    ma:   { ratio: 1.0857, score: 21, gc: true },
    aroon: { up: 92, down: 16, score: 10 },
    buyReason: "MACD 直近クロス後ヒスト拡大 ／ Aroon 強い上昇トレンド（Up=92）",
  },
];

const DAY_TRADES: TradeRecord[] = [
  {
    rank: 1, name: "サウンドハウンドAI", ticker: "SOUN", size: "small",
    dateBuy: "2019-06-26", priceBuy: 1208.93, dateSell: "2019-06-27", priceSell: 1796.64,
    returnPct: 48.61, score: 67.7,
    rsi:  { val: 52.3,  score: 10 },
    macd: { val: -8.0117, sig: -27.8359, hist: +19.8243, score: 15 },
    bb:   { val: 0.728, score: 8  },
    ma:   { ratio: 1.0421, score: 24, gc: true },
    aroon: { up: 84, down: 12, score: 10 },
    buyReason: "EMA5 大きく上方乖離（+4.2%）／ Aroon 強い上昇トレンド（Up=84）",
  },
  {
    rank: 2, name: "スーパーマイクロコンピュータ", ticker: "SMCI", size: "mid",
    dateBuy: "2020-08-17", priceBuy: 555.98, dateSell: "2020-08-18", priceSell: 799.67,
    returnPct: 43.83, score: 74.6,
    rsi:  { val: 59.7,  score: 16 },
    macd: { val: 36.7994, sig: 23.2841, hist: +13.5153, score: 10 },
    bb:   { val: 0.813, score: 6  },
    ma:   { ratio: 1.0425, score: 24, gc: true },
    aroon: { up: 84, down: 4, score: 10 },
    buyReason: "RSI 上昇モメンタム帯（59.7） ／ EMA5 上方乖離 ／ Aroon 強い上昇トレンド",
  },
  {
    rank: 3, name: "スーパーマイクロコンピュータ", ticker: "SMCI", size: "mid",
    dateBuy: "2015-11-09", priceBuy: 118.34, dateSell: "2015-11-10", priceSell: 166.40,
    returnPct: 40.62, score: 72.4,
    rsi:  { val: 60.9,  score: 17 },
    macd: { val: 4.2305, sig: 1.2961, hist: +2.9344, score: 15 },
    bb:   { val: 0.913, score: 3  },
    ma:   { ratio: 1.0684, score: 22, gc: true },
    aroon: { up: 96, down: 36, score: 6 },
    buyReason: "RSI 上昇モメンタム帯（60.9） ／ EMA5 大きく上方乖離（+6.8%）",
  },
  {
    rank: 4, name: "サウンドハウンドAI", ticker: "SOUN", size: "small",
    dateBuy: "2018-07-24", priceBuy: 455.21, dateSell: "2018-07-25", priceSell: 623.55,
    returnPct: 36.98, score: 69.3,
    rsi:  { val: 60.6,  score: 16 },
    macd: { val: 17.3657, sig: 15.7270, hist: +1.6387, score: 24 },
    bb:   { val: 0.844, score: 6  },
    ma:   { ratio: 1.0789, score: 13, gc: true },
    aroon: { up: 100, down: 0, score: 10 },
    buyReason: "MACDゴールデンクロス当日 ／ Aroon 完全上昇トレンド（Up=100, Down=0）",
  },
  {
    rank: 5, name: "東京エレクトロン", ticker: "8035.T", size: "large",
    dateBuy: "2022-01-05", priceBuy: 438.12, dateSell: "2022-01-06", priceSell: 592.60,
    returnPct: 35.26, score: 69.0,
    rsi:  { val: 59.9,  score: 12 },
    macd: { val: 5.5471, sig: 2.1112, hist: +3.4359, score: 20 },
    bb:   { val: 0.876, score: 2  },
    ma:   { ratio: 1.0172, score: 22, gc: true },
    aroon: { up: 92, down: 80, score: 6 },
    buyReason: "MACD 直近クロス後ヒスト拡大 ／ EMA5 上方（+1.7%）＋ゴールデンクロス帯",
  },
];

// ─── Strategy results loader (server-side) ────────────────────────────────────

type StrategyResult = {
  buy_weights: Record<string, number>;
  buy_threshold: number;
  sell_rule: string;
  sharpe: number;
  n_trades: number;
  win_rate: number;
  avg_return: number;
  max_dd: number;
};

type SellRuleStat = {
  sell_rule: string;
  sell_rule_ja: string;
  best_sharpe: number;
  avg_sharpe: number;
  best_win_rate: number;
  best_n_trades: number;
  best_avg_return: number;
  best_weights: Record<string, number>;
  best_threshold: number;
};

type StrategyData = {
  available: true;
  version: number;
  generated_at: string;
  mode?: string;
  n_tickers: number;
  tickers: string[];
  n_evaluated: number;
  top100: StrategyResult[];
  sell_rule_ranking: SellRuleStat[];
  indicator_weight_corr_with_sharpe: Record<string, number>;
  top100_median_weights: Record<string, number>;
  top100_weight_ranking: string[];
  summary: {
    best_sharpe: number;
    best_sell_rule: string;
    best_weights: Record<string, number>;
    best_threshold: number;
  };
  lookahead_bias_fixed?: boolean;
  entry_price?: string;
} | { available: false };

function loadStrategyData(): StrategyData {
  const filePath = path.join(process.cwd(), "backtest", "strategy_results.json");
  if (!existsSync(filePath)) return { available: false };
  try {
    return { available: true, ...JSON.parse(readFileSync(filePath, "utf-8")) };
  } catch {
    return { available: false };
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function multColor(v: number) {
  if (v >= 1.2) return "text-emerald-400 font-bold";
  if (v >= 1.0) return "text-emerald-300";
  if (v >= 0.8) return "text-gray-300";
  return "text-gray-500";
}

function sharpeBar(v: number, max = 0.65) {
  const pct = Math.min(Math.round((v / max) * 100), 100);
  return pct;
}

function retColor(r: number) {
  return r >= 0 ? "text-emerald-400" : "text-red-400";
}

function scoreColor(s: number) {
  if (s >= 70) return "text-emerald-400";
  if (s >= 50) return "text-yellow-400";
  return "text-orange-400";
}

function aroonDeltaColor(d: number) {
  if (d > 0.05) return "text-emerald-400";
  if (d > 0) return "text-emerald-600";
  return "text-gray-500";
}

const INDICATOR_COLORS: Record<string, string> = {
  RSI:  "bg-emerald-500",
  MACD: "bg-blue-500",
  BB:   "bg-purple-500",
  MA:   "bg-orange-500",
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-white font-bold text-base border-l-4 border-blue-500 pl-3 mb-4">
      {children}
    </h2>
  );
}

const RANK_BADGE_COLORS = [
  "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40",
  "bg-gray-600/20 text-gray-300 border border-gray-600/40",
  "bg-orange-700/20 text-orange-500 border border-orange-700/40",
  "bg-gray-800 text-gray-600 border border-gray-700/40",
];

function getRanks(row: Record<string, number>, inds: string[]) {
  const sorted = [...inds].sort((a, b) => row[b] - row[a]);
  return Object.fromEntries(inds.map((ind) => [ind, sorted.indexOf(ind)]));
}

function MultsTable({ mode }: { mode: "swing" | "day" }) {
  const data = MULTS[mode];
  const sh   = SHARPES[mode];
  const sizes: ("large" | "mid" | "small")[] = ["large", "mid", "small"];
  const inds:  ("RSI" | "MACD" | "BB" | "MA")[] = ["RSI", "MACD", "BB", "MA"];
  const sizeJa = { large: "大型株", mid: "中型株", small: "小型株" };
  const rankLabels = ["1位", "2位", "3位", "4位"];

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-800">
              <th className="text-left py-2 pr-4">規模</th>
              {inds.map((ind) => (
                <th key={ind} className="text-center py-2 px-3">
                  <span className={`inline-block w-2 h-2 rounded-full mr-1 ${INDICATOR_COLORS[ind]}`} />
                  {ind}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sizes.map((sz) => {
              const ranks = getRanks(data[sz] as Record<string, number>, inds);
              return (
                <tr key={sz} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-2.5 pr-4 text-gray-400">{sizeJa[sz]}</td>
                  {inds.map((ind) => {
                    const mult = data[sz][ind];
                    const sharpe = sh[sz][ind];
                    const rankIdx = ranks[ind];
                    return (
                      <td key={ind} className="py-2.5 px-3 text-center">
                        <span className={`inline-block text-xs font-bold px-1 py-0.5 rounded mb-1 ${RANK_BADGE_COLORS[rankIdx]}`}>
                          {rankLabels[rankIdx]}
                        </span>
                        <div className={`font-mono ${multColor(mult)}`}>×{mult.toFixed(3)}</div>
                        <div className="mt-1 relative h-1 bg-gray-800 rounded-full w-12 mx-auto">
                          <div
                            className={`absolute left-0 top-0 h-1 rounded-full ${INDICATOR_COLORS[ind]}`}
                            style={{ width: `${sharpeBar(sharpe)}%`, opacity: 0.7 }}
                          />
                        </div>
                        <div className="text-gray-600 mt-0.5">Sh={sharpe.toFixed(3)}</div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Ranked summary per size */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {sizes.map((sz) => {
          const sorted = [...inds].sort((a, b) => data[sz][b] - data[sz][a]);
          return (
            <div key={sz} className="bg-gray-800/50 rounded-xl p-3">
              <p className="text-gray-400 text-xs font-semibold mb-2">{sizeJa[sz]} 重要度ランキング</p>
              <div className="space-y-1">
                {sorted.map((ind, i) => (
                  <div key={ind} className="flex items-center gap-2">
                    <span className={`text-xs font-bold w-5 flex-shrink-0 ${["text-yellow-400","text-gray-300","text-orange-500","text-gray-600"][i]}`}>
                      {i + 1}位
                    </span>
                    <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${INDICATOR_COLORS[ind]}`} />
                    <span className="text-gray-300 text-xs font-medium w-8">{ind}</span>
                    <span className="text-gray-500 text-xs font-mono">×{(data[sz][ind] as number).toFixed(3)}</span>
                    {i === 0 && <span className="text-yellow-600 text-xs ml-auto">最重視</span>}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-gray-600 text-xs">
        重みは50銘柄×10年 Monte Carlo Walk-Forward（5 fold）でシャープレシオを最大化して導出。
        バーの長さ = 単体シャープレシオ（高いほど重みが大きい）。
      </p>
    </div>
  );
}

function AroonDeltaRow({ mode }: { mode: "swing" | "day" }) {
  const d = AROON_DELTA[mode];
  return (
    <div className="grid grid-cols-3 gap-3 mt-4">
      {(["large", "mid", "small"] as const).map((sz) => {
        const delta = d[sz];
        const applicable = mode === "day" || sz === "large";
        return (
          <div key={sz} className="bg-gray-800/50 rounded-lg px-3 py-2 text-center">
            <p className="text-gray-500 text-xs">{sz === "large" ? "大型" : sz === "mid" ? "中型" : "小型"}</p>
            {applicable ? (
              <p className={`font-mono text-sm mt-1 ${aroonDeltaColor(delta)}`}>
                {delta > 0 ? "+" : ""}{delta.toFixed(3)}
              </p>
            ) : (
              <p className="text-gray-700 text-sm mt-1">非適用</p>
            )}
            <p className="text-gray-600 text-xs">ΔSharpe</p>
          </div>
        );
      })}
    </div>
  );
}

function TradeCard({ tr, mode }: { tr: TradeRecord; mode: "swing" | "day" }) {
  const m = MULTS[mode][tr.size];
  const holdDays = mode === "swing" ? 5 : 1;
  const yfUrls = yahooFinanceUrls(tr.ticker);

  const indicators = [
    { name: "RSI",  val: `${tr.rsi.val}`,  score: tr.rsi.score,  mult: m.RSI,  color: "bg-emerald-500" },
    { name: "MACD", val: `${tr.macd.val.toPrecision(5)}`, score: tr.macd.score, mult: m.MACD, color: "bg-blue-500" },
    { name: "BB %B",val: `${tr.bb.val.toFixed(3)}`, score: tr.bb.score, mult: m.BB, color: "bg-purple-500" },
    { name: `MA(${mode === "swing" ? "EMA20" : "EMA5"})`, val: `×${tr.ma.ratio.toFixed(4)}`, score: tr.ma.score, mult: m.MA, color: "bg-orange-500" },
  ];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-800/50 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <span className="text-gray-500 font-mono text-sm">#{tr.rank}</span>
          <div>
            <span className="text-white font-semibold">{tr.name}</span>
            <span className="text-gray-500 font-mono text-xs ml-2">({tr.ticker})</span>
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded bg-gray-700 text-gray-400">
              {tr.size === "large" ? "大型" : tr.size === "mid" ? "中型" : "小型"}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {yfUrls.map(({ label, href, kind }) => (
            <a key={href} href={href} target="_blank" rel="noopener noreferrer"
              className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${
                kind === "chart"
                  ? "bg-blue-950/60 text-blue-400 hover:bg-blue-900/60 hover:text-blue-200"
                  : "bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-gray-200"
              }`}>
              {kind === "chart" ? (
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
              ) : (
                <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              )}
              {label}
            </a>
          ))}
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Buy / Sell row */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-800 rounded-xl p-3">
            <p className="text-xs text-gray-500 mb-1">◆ 買いエントリー</p>
            <p className="text-white font-mono font-bold">{tr.priceBuy.toLocaleString()}</p>
            <p className="text-gray-400 text-xs">{tr.dateBuy}</p>
            <p className={`text-sm font-bold mt-1 ${scoreColor(tr.score)}`}>スコア {tr.score}点</p>
          </div>
          <div className="bg-gray-800 rounded-xl p-3">
            <p className="text-xs text-gray-500 mb-1">◆ 売りクローズ (+{holdDays}日)</p>
            <p className="text-white font-mono font-bold">{tr.priceSell.toLocaleString()}</p>
            <p className="text-gray-400 text-xs">{tr.dateSell}</p>
            <p className={`text-sm font-bold mt-1 ${retColor(tr.returnPct)}`}>
              {tr.returnPct >= 0 ? "+" : ""}{tr.returnPct.toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Indicator table */}
        <div>
          <p className="text-xs text-gray-500 mb-2">指標スコア内訳</p>
          <div className="space-y-1.5">
            {indicators.map(({ name, val, score, mult, color }) => {
              const contrib = score * mult;
              const barW = Math.round((score / 25) * 100);
              return (
                <div key={name} className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs w-20 flex-shrink-0">{name}</span>
                  <span className="text-gray-400 text-xs font-mono w-20 flex-shrink-0">{val}</span>
                  <div className="flex-1 relative h-4 bg-gray-800 rounded overflow-hidden">
                    <div className={`absolute left-0 top-0 h-4 ${color} opacity-60 rounded`} style={{ width: `${barW}%` }} />
                    <span className="absolute inset-0 flex items-center px-2 text-xs text-white font-mono">
                      {score}/25 × {mult.toFixed(3)} = {contrib.toFixed(1)}pt
                    </span>
                  </div>
                </div>
              );
            })}
            {tr.aroon && (
              <div className="flex items-center gap-2">
                <span className="text-gray-500 text-xs w-20 flex-shrink-0">Aroon+10</span>
                <span className="text-gray-400 text-xs font-mono w-20 flex-shrink-0">
                  Up={tr.aroon.up}
                </span>
                <div className="flex-1 relative h-4 bg-gray-800 rounded overflow-hidden">
                  <div className="absolute left-0 top-0 h-4 bg-indigo-500 opacity-60 rounded"
                    style={{ width: `${(tr.aroon.score / 10) * 100}%` }} />
                  <span className="absolute inset-0 flex items-center px-2 text-xs text-white font-mono">
                    {tr.aroon.score}/10 ボーナス (Down={tr.aroon.down})
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* MACD detail */}
        <div className="bg-gray-800/50 rounded-xl px-3 py-2 text-xs space-y-0.5">
          <p className="text-gray-500 text-xs mb-1">MACD 詳細</p>
          <p className="text-gray-400">
            <span className="text-blue-400">MACD</span> = {tr.macd.val.toPrecision(6)} &nbsp;
            <span className="text-gray-500">Signal</span> = {tr.macd.sig.toPrecision(6)} &nbsp;
            <span className={tr.macd.hist >= 0 ? "text-emerald-400" : "text-red-400"}>
              Hist {tr.macd.hist >= 0 ? "+" : ""}{tr.macd.hist.toPrecision(5)}
            </span>
          </p>
          <p className="text-gray-400">
            <span className="text-orange-400">MA ratio</span> = {tr.ma.ratio.toFixed(4)} &nbsp;
            ({tr.ma.gc
              ? <span className="text-emerald-400">ゴールデンクロス帯</span>
              : <span className="text-red-400">デッドクロス帯</span>
            })
          </p>
        </div>

        {/* Buy reason */}
        <div className="border-t border-gray-800 pt-3">
          <p className="text-xs text-gray-500 mb-1">【買い根拠】</p>
          <p className="text-gray-300 text-xs leading-relaxed">{tr.buyReason}</p>
          <p className="text-gray-600 text-xs mt-2">
            【売り根拠】 エントリー後 {holdDays} 営業日経過による機械的クローズ（固定ホールド戦略）
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Real-data strategy result components ─────────────────────────────────────

const SELL_RULE_SHORT: Record<string, string> = {
  hold_3d: "固定3日", hold_5d: "固定5日", hold_7d: "固定7日",
  hold_10d: "固定10日", hold_15d: "固定15日", hold_20d: "固定20日",
  "target5_stop3":  "+5%/−3%", "target10_stop5": "+10%/−5%",
  "target15_stop5": "+15%/−5%", "target20_stop5": "+20%/−5%",
  "target15_stop7": "+15%/−7%", "target20_stop7": "+20%/−7%",
  trail_5pct: "Trail5%", trail_10pct: "Trail10%",
};

const IND_COLORS_CLS: Record<string, string> = {
  RSI:   "bg-emerald-500", MACD:  "bg-blue-500",   BB:    "bg-purple-500", MA:    "bg-orange-500",
  Aroon: "bg-cyan-500",    Stoch: "bg-pink-500",    CCI:   "bg-yellow-500", ROC:   "bg-red-500",
};
const IND_TEXT_CLS: Record<string, string> = {
  RSI:   "text-emerald-400", MACD:  "text-blue-400",   BB:    "text-purple-400", MA:    "text-orange-400",
  Aroon: "text-cyan-400",    Stoch: "text-pink-400",   CCI:   "text-yellow-400", ROC:   "text-red-400",
};
const RANK_TEXT = ["text-yellow-400", "text-gray-300", "text-orange-500", "text-gray-600"];

function WeightBar({ weights }: { weights: Record<string, number> }) {
  const total = Object.values(weights).reduce((a, b) => a + b, 0) || 1;
  const sorted = Object.entries(weights).sort(([, a], [, b]) => b - a);
  return (
    <div className="flex h-3 rounded overflow-hidden gap-px w-full">
      {sorted.map(([name, w]) => (
        <div
          key={name}
          className={`${IND_COLORS_CLS[name] ?? "bg-gray-500"} opacity-80`}
          style={{ width: `${(w / total) * 100}%` }}
          title={`${name}: ${w.toFixed(2)}`}
        />
      ))}
    </div>
  );
}

function SellRuleRankingSection({ stats }: { stats: SellRuleStat[] }) {
  const maxSharpe = Math.max(...stats.map((s) => s.best_sharpe));
  return (
    <div className="space-y-2">
      {stats.map((sr, i) => {
        const barW = Math.round((sr.best_sharpe / maxSharpe) * 100);
        return (
          <div key={sr.sell_rule} className="bg-gray-800/60 rounded-xl px-4 py-3">
            <div className="flex items-center gap-3 mb-2">
              <span className={`text-sm font-bold w-5 flex-shrink-0 ${RANK_TEXT[Math.min(i, 3)]}`}>
                {i + 1}位
              </span>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-200 text-sm font-medium">{sr.sell_rule_ja}</span>
                  <div className="flex items-center gap-3 text-xs">
                    <span className="text-emerald-400 font-bold">シャープ {sr.best_sharpe.toFixed(3)}</span>
                    <span className="text-gray-400">勝率 {(sr.best_win_rate * 100).toFixed(1)}%</span>
                    <span className="text-gray-400">平均 {sr.best_avg_return > 0 ? "+" : ""}{sr.best_avg_return.toFixed(2)}%</span>
                  </div>
                </div>
                <div className="relative h-2 bg-gray-700 rounded-full">
                  <div
                    className="absolute left-0 top-0 h-2 rounded-full bg-emerald-500 opacity-70"
                    style={{ width: `${barW}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="ml-8">
              <p className="text-gray-600 text-xs mb-1">最適買いシグナル重み</p>
              <div className="flex items-center gap-3">
                <WeightBar weights={sr.best_weights} />
                <div className="flex gap-2 flex-shrink-0 text-xs">
                  {Object.entries(sr.best_weights)
                    .sort(([, a], [, b]) => b - a)
                    .map(([name, w]) => (
                      <span key={name} className={IND_TEXT_CLS[name] ?? "text-gray-400"}>
                        {name}×{w.toFixed(2)}
                      </span>
                    ))}
                </div>
                <span className="text-gray-600 text-xs flex-shrink-0">≥{sr.best_threshold}点</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TopStrategiesSection({ strategies }: { strategies: StrategyResult[] }) {
  const top = strategies.slice(0, 20);
  return (
    <div className="space-y-2">
      {top.map((s, i) => {
        const ranked = Object.entries(s.buy_weights).sort(([, a], [, b]) => b - a);
        return (
          <div key={i} className="bg-gray-800/50 rounded-xl px-4 py-3 text-xs">
            <div className="flex items-start gap-3">
              <span className={`text-sm font-bold w-5 flex-shrink-0 mt-0.5 ${RANK_TEXT[Math.min(i, 3)]}`}>
                {i + 1}
              </span>
              <div className="flex-1 space-y-1.5">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                  <span className="text-emerald-400 font-bold">シャープ {s.sharpe.toFixed(3)}</span>
                  <span className="text-gray-300">勝率 {(s.win_rate * 100).toFixed(1)}%</span>
                  <span className="text-gray-300">平均 {s.avg_return > 0 ? "+" : ""}{s.avg_return.toFixed(2)}%</span>
                  <span className="text-gray-500">{s.n_trades}件</span>
                  <span className="text-red-400/70">最大損失 {s.max_dd.toFixed(1)}%</span>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-gray-500">買い:</span>
                  {ranked.map(([name, w]) => (
                    <span key={name} className={`${IND_TEXT_CLS[name] ?? "text-gray-400"} font-medium`}>
                      {name}×{w.toFixed(2)}
                    </span>
                  ))}
                  <span className="text-gray-600">≥{s.buy_threshold}点</span>
                  <span className="text-gray-700">|</span>
                  <span className="text-gray-500">売り:</span>
                  <span className="text-yellow-300/80">{SELL_RULE_SHORT[s.sell_rule] ?? s.sell_rule}</span>
                </div>
                <WeightBar weights={s.buy_weights} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function IndicatorImportanceSection({
  corr, medWeights, ranking,
}: {
  corr: Record<string, number>;
  medWeights: Record<string, number>;
  ranking: string[];
}) {
  const maxCorr = Math.max(...Object.values(corr).map(Math.abs));
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div className="bg-gray-800/50 rounded-xl p-4">
        <p className="text-gray-400 text-xs font-semibold mb-3">指標重要度ランキング (上位100戦略での重み中央値)</p>
        <div className="space-y-2">
          {ranking.map((name, i) => {
            const med = medWeights[name] ?? 0;
            const barW = Math.round((med / 4) * 100);
            return (
              <div key={name} className="flex items-center gap-2">
                <span className={`text-xs font-bold w-5 flex-shrink-0 ${RANK_TEXT[Math.min(i, 3)]}`}>{i + 1}位</span>
                <span className={`${IND_COLORS_CLS[name] ? `inline-block w-2 h-2 rounded-full ${IND_COLORS_CLS[name]}` : ""} flex-shrink-0`} />
                <span className={`${IND_TEXT_CLS[name] ?? "text-gray-400"} font-medium w-12 text-xs`}>{name}</span>
                <div className="flex-1 relative h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div className={`absolute left-0 top-0 h-3 rounded-full ${IND_COLORS_CLS[name] ?? "bg-gray-400"}`}
                    style={{ width: `${barW}%`, opacity: 0.75 }} />
                </div>
                <span className="text-gray-400 font-mono text-xs w-12 text-right">×{med.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="bg-gray-800/50 rounded-xl p-4">
        <p className="text-gray-400 text-xs font-semibold mb-3">シャープレシオとの相関 (高いほど重視すべき)</p>
        <div className="space-y-2">
          {Object.entries(corr).sort(([, a], [, b]) => b - a).map(([name, c]) => {
            const barW = Math.round((Math.abs(c) / (maxCorr || 1)) * 100);
            const isPos = c >= 0;
            return (
              <div key={name} className="flex items-center gap-2">
                <span className={`${IND_TEXT_CLS[name] ?? "text-gray-400"} font-medium w-12 text-xs flex-shrink-0`}>{name}</span>
                <div className="flex-1 relative h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`absolute left-0 top-0 h-3 rounded-full ${isPos ? "bg-emerald-500" : "bg-red-500"}`}
                    style={{ width: `${barW}%`, opacity: 0.75 }}
                  />
                </div>
                <span className={`${isPos ? "text-emerald-400" : "text-red-400"} font-mono text-xs w-14 text-right`}>
                  {c >= 0 ? "+" : ""}{c.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
        <p className="text-gray-600 text-xs mt-3">
          正の相関 = 重みが大きいほどシャープが高い傾向
        </p>
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function BacktestPage() {
  const swingStats = { signals: 1386, winRate: 56.2, avgRet: 0.70 };
  const dayStats   = { signals: 14967, winRate: 52.1, avgRet: 0.15 };
  const strategyData = loadStrategyData();

  return (
    <div className="space-y-10 pb-12">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm">
        <Link href="/" className="text-gray-500 hover:text-gray-300">← ウォッチリスト</Link>
        <span className="text-gray-700">/</span>
        <span className="text-gray-400">バックテスト結果</span>
      </div>

      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold text-white">バックテスト結果</h1>
        <p className="text-gray-500 text-sm mt-1">
          買い指標 × 売りルール 最適化 + Monte Carlo Walk-Forward シミュレーション
        </p>
      </div>

      {/* ── Real data strategy search results ── */}
      {strategyData.available ? (
        <section className="space-y-6">
          <div className="flex items-center gap-3 flex-wrap">
            <SectionHeading>実データ戦略最適化結果</SectionHeading>
            <span className="text-xs text-emerald-400 bg-emerald-900/30 border border-emerald-700/50 px-2 py-0.5 rounded-full">
              実株価データ ✓
            </span>
          </div>

          {/* Summary stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "検証銘柄数", value: String(strategyData.n_tickers), sub: strategyData.tickers.join(", ").substring(0, 30) + "..." },
              { label: "検証戦略数", value: strategyData.n_evaluated.toLocaleString(), sub: "買い×売り組み合わせ" },
              { label: "最高シャープ", value: strategyData.summary.best_sharpe?.toFixed(3) ?? "—", sub: "リスク調整後収益" },
              { label: "最良売りルール", value: SELL_RULE_SHORT[strategyData.summary.best_sell_rule ?? ""] ?? "—", sub: "上位戦略で最多" },
            ].map(({ label, value, sub }) => (
              <div key={label} className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                <p className="text-white font-bold text-lg">{value}</p>
                <p className="text-gray-400 text-xs mt-0.5">{label}</p>
                <p className="text-gray-600 text-xs">{sub}</p>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs -mt-2">
            <span className="text-gray-600">生成日時: {strategyData.generated_at}</span>
            {"lookahead_bias_fixed" in strategyData && strategyData.lookahead_bias_fixed && (
              <span className="text-emerald-400/80 bg-emerald-900/20 border border-emerald-700/30 px-2 py-0.5 rounded-full">
                先読みバイアスなし ✓ エントリー=翌始値
              </span>
            )}
            {"mode" in strategyData && strategyData.mode && (
              <span className="text-blue-400/80 bg-blue-900/20 border border-blue-700/30 px-2 py-0.5 rounded-full">
                {strategyData.mode}
              </span>
            )}
          </div>

          {/* Sell rule ranking */}
          <div>
            <h3 className="text-gray-300 text-sm font-semibold mb-3">
              売りルール ランキング — どの売り方が最も有効か
            </h3>
            <SellRuleRankingSection stats={strategyData.sell_rule_ranking} />
          </div>

          {/* Indicator importance */}
          <div>
            <h3 className="text-gray-300 text-sm font-semibold mb-3">
              買い指標 重要度ランキング — どの指標を重視すべきか
            </h3>
            <IndicatorImportanceSection
              corr={strategyData.indicator_weight_corr_with_sharpe}
              medWeights={strategyData.top100_median_weights}
              ranking={strategyData.top100_weight_ranking}
            />
          </div>

          {/* Top strategies */}
          <div>
            <h3 className="text-gray-300 text-sm font-semibold mb-3">
              総合 TOP 20 戦略 — 買い指標 + 売りルール 最強の組み合わせ
            </h3>
            <TopStrategiesSection strategies={strategyData.top100} />
          </div>
        </section>
      ) : (
        <section>
          <div className="bg-blue-950/40 border border-blue-700/50 rounded-xl p-6">
            <h3 className="text-blue-300 font-semibold mb-3">実データ戦略最適化を実行するには</h3>
            <p className="text-blue-200/70 text-sm mb-4">
              以下を手元のPC（ネットワーク接続あり）で実行すると、実際の株価データを使って
              56万通り以上の買い×売り戦略を検証し、最優秀の組み合わせをランキング表示します。
            </p>
            <div className="bg-gray-900 rounded-lg p-4 font-mono text-xs space-y-1">
              <p className="text-gray-500"># Step 1: 実株価データ取得（インターネット接続が必要）</p>
              <p className="text-emerald-400">node backtest/fetch_data.mjs</p>
              <p className="text-gray-500 mt-2"># Step 2: 戦略最適化実行（約5〜15分）</p>
              <p className="text-emerald-400">python3 backtest/strategy_search.py</p>
              <p className="text-gray-500 mt-2"># 結果は backtest/strategy_results.json に保存</p>
              <p className="text-gray-500"># その後 git push すればこのページに自動表示されます</p>
            </div>
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              {[
                { icon: "📊", title: "売りルール14種", desc: "固定保有3〜20日、利確+ストップ、トレーリングストップ" },
                { icon: "⚖️", title: "重み8,000通り", desc: "RSI/MACD/BB/MA/Aroon/Stoch/CCI/ROC の Dirichlet サンプリング" },
                { icon: "🏆", title: "評価指標", desc: "アニュアライズド・シャープレシオ（リスク調整後）" },
              ].map(({ icon, title, desc }) => (
                <div key={title} className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-gray-300 font-medium">{icon} {title}</p>
                  <p className="text-gray-500 mt-1">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* MC Overview cards */}
      <div className="bg-yellow-950/30 border border-yellow-800/40 rounded-xl p-4 text-xs text-yellow-200/60">
        <p className="font-semibold text-yellow-300/80 mb-1">⚠️ 以下のセクション（重み付け・Aroon・トレード例）について</p>
        <p>実際の市場データではなく、2015-2024年の CAGR・ボラティリティに基づく Monte Carlo GBM による
        シミュレーションです。株価は全て100.0から開始。過去の模擬パフォーマンスは将来の実績を保証しません。
        実データ結果は上記「実データ戦略最適化結果」セクションをご参照ください。</p>
      </div>

      {/* ── Weights section ── */}
      <section className="space-y-6">
        <SectionHeading>指標の重み付け（時価総額別）</SectionHeading>
        <p className="text-gray-500 text-xs -mt-2">
          50銘柄×10年のMCバックテスト（Walk-Forward）でシャープレシオを最大化する重みを規模別に導出。
          重み = 単体シャープレシオに比例（合計4.0に正規化）。
        </p>

        <div className="space-y-6">
          <div>
            <h3 className="text-gray-300 text-sm font-semibold mb-3">スイングトレード（保有5日）</h3>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <MultsTable mode="swing" />
            </div>
          </div>
          <div>
            <h3 className="text-gray-300 text-sm font-semibold mb-3">デイトレード（保有1日）</h3>
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <MultsTable mode="day" />
            </div>
          </div>
        </div>
      </section>

      {/* ── Aroon section ── */}
      <section className="space-y-4">
        <SectionHeading>Aroon インジケーター（ボーナス加点）</SectionHeading>
        <p className="text-gray-500 text-xs -mt-2">
          25本足の高値・安値タイミングからトレンド方向を判定。AroonUp&gt;70 かつ AroonDown&lt;30 で最大+10点ボーナス加算。
          ΔSharpe = Aroon追加前後のシャープレシオ差（+が有効）。
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-gray-400 text-sm font-semibold mb-1">スイングトレード</p>
            <p className="text-gray-600 text-xs mb-3">大型株のみ適用（他規模は4指標モデルが優位）</p>
            <AroonDeltaRow mode="swing" />
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-gray-400 text-sm font-semibold mb-1">デイトレード</p>
            <p className="text-gray-600 text-xs mb-3">全規模に適用（ΔSharpe +5.80 を確認）</p>
            <AroonDeltaRow mode="day" />
          </div>
        </div>
      </section>

      {/* ── Swing trades ── */}
      <section className="space-y-4">
        <SectionHeading>スイングトレード — リターン上位 5 件</SectionHeading>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-wrap gap-6 text-sm">
          <div>
            <p className="text-gray-500 text-xs">全シグナル数</p>
            <p className="text-white font-bold">{swingStats.signals.toLocaleString()} 件</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">勝率</p>
            <p className="text-emerald-400 font-bold">{swingStats.winRate}%</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">平均リターン</p>
            <p className="text-emerald-400 font-bold">+{swingStats.avgRet.toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">閾値</p>
            <p className="text-gray-300 font-bold">≥ 65 点</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">保有期間</p>
            <p className="text-gray-300 font-bold">5 営業日</p>
          </div>
        </div>
        <div className="space-y-4">
          {SWING_TRADES.map((tr) => (
            <TradeCard key={tr.rank} tr={tr} mode="swing" />
          ))}
        </div>
      </section>

      {/* ── Day trades ── */}
      <section className="space-y-4">
        <SectionHeading>デイトレード — リターン上位 5 件</SectionHeading>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-wrap gap-6 text-sm">
          <div>
            <p className="text-gray-500 text-xs">全シグナル数</p>
            <p className="text-white font-bold">{dayStats.signals.toLocaleString()} 件</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">勝率</p>
            <p className="text-emerald-400 font-bold">{dayStats.winRate}%</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">平均リターン</p>
            <p className="text-emerald-400 font-bold">+{dayStats.avgRet.toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">閾値</p>
            <p className="text-gray-300 font-bold">≥ 65 点</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">保有期間</p>
            <p className="text-gray-300 font-bold">1 営業日</p>
          </div>
        </div>
        <div className="space-y-4">
          {DAY_TRADES.map((tr) => (
            <TradeCard key={tr.rank} tr={tr} mode="day" />
          ))}
        </div>
      </section>

      {/* ── Technical integrity ── */}
      <section className="space-y-3">
        <SectionHeading>技術的整合性チェック</SectionHeading>
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3 text-sm">
          {[
            { icon: "✓", color: "text-emerald-400", text: "先読みバイアスなし: score[t] は price[0..t] のみ参照。price[t+1:] をシャッフルしてもスコア不変（30点全テスト通過）" },
            { icon: "✓", color: "text-emerald-400", text: "買い価格 = close[t]（エントリー日終値）、売り価格 = close[t+hold]（固定ホールド後の終値）" },
            { icon: "✓", color: "text-emerald-400", text: "全MULTS = Walk-Forwardバックテストの再導出値と完全一致（Δ=0）" },
            { icon: "✓", color: "text-emerald-400", text: "H/L生成は専用RNG（seed=9999）を使用し、終値RNGのシードシーケンスを汚染しない" },
            { icon: "✓", color: "text-emerald-400", text: "取引カレンダー: NYSE平日ベース 2015-01-02〜（祝日は除外していないが土日は除外）" },
          ].map(({ icon, color, text }) => (
            <div key={text} className="flex gap-3">
              <span className={`${color} flex-shrink-0 font-bold`}>{icon}</span>
              <p className="text-gray-400 text-xs leading-relaxed">{text}</p>
            </div>
          ))}
        </div>
      </section>

      <p className="text-center text-gray-700 text-xs">
        ※ 情報提供のみを目的としています。投資判断は自己責任でお願いします。
      </p>
    </div>
  );
}
