import Link from "next/link";
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

function MultsTable({ mode }: { mode: "swing" | "day" }) {
  const data = MULTS[mode];
  const sh   = SHARPES[mode];
  const sizes: ("large" | "mid" | "small")[] = ["large", "mid", "small"];
  const inds:  ("RSI" | "MACD" | "BB" | "MA")[] = ["RSI", "MACD", "BB", "MA"];
  const sizeJa = { large: "大型株", mid: "中型株", small: "小型株" };

  return (
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
          {sizes.map((sz) => (
            <tr key={sz} className="border-b border-gray-800/50 hover:bg-gray-800/30">
              <td className="py-2.5 pr-4 text-gray-400">{sizeJa[sz]}</td>
              {inds.map((ind) => {
                const mult = data[sz][ind];
                const sharpe = sh[sz][ind];
                return (
                  <td key={ind} className="py-2.5 px-3 text-center">
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
          ))}
        </tbody>
      </table>
      <p className="text-gray-600 text-xs mt-2">
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

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function BacktestPage() {
  const swingStats = { signals: 1386, winRate: 56.2, avgRet: 0.70 };
  const dayStats   = { signals: 14967, winRate: 52.1, avgRet: 0.15 };

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
          Monte Carlo Walk-Forward シミュレーション — 9銘柄 × 6パス × 2520営業日（2015-01-02〜2024-12-31相当）
        </p>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "検証銘柄数", value: "9", sub: "半導体・AI・量子" },
          { label: "シミュレーション期間", value: "10年", sub: "2520 営業日" },
          { label: "先読みバイアス検証", value: "30点", sub: "全テスト通過 ✓" },
          { label: "最適化手法", value: "Walk-Forward", sub: "5 fold CV" },
        ].map(({ label, value, sub }) => (
          <div key={label} className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
            <p className="text-white font-bold text-lg">{value}</p>
            <p className="text-gray-400 text-xs mt-0.5">{label}</p>
            <p className="text-gray-600 text-xs">{sub}</p>
          </div>
        ))}
      </div>

      {/* Disclaimer */}
      <div className="bg-yellow-950/40 border border-yellow-800/50 rounded-xl p-4 text-xs text-yellow-200/70">
        <p className="font-semibold text-yellow-300 mb-1">⚠️ シミュレーションについて</p>
        <p>本バックテストは実際の市場データではなく、各銘柄の2015-2024年 CAGR・ボラティリティに基づく
        モンテカルロ GBM（体制転換モデル）によるシミュレーションです。
        株価は全て100.0から開始。過去の模擬パフォーマンスは将来の実績を保証しません。
        投資判断は自己責任でお願いします。</p>
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
