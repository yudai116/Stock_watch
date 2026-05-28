"use client";
import type { StockScore } from "@/types";
import { signalToJa, analystSignalToJa } from "@/lib/formatters";

interface Props {
  stock: StockScore;
  mode?: "swing" | "day";
}

const SWING_INDICATORS = [
  { key: "rsi" as const,        label: "RSI",           description: "相対力指数(14) — 逆張り" },
  { key: "macd" as const,       label: "MACD",          description: "移動平均収束拡散" },
  { key: "bollinger" as const,  label: "ボリンジャーバンド", description: "価格帯位置(%B)" },
  { key: "moving_avg" as const, label: "移動平均線",    description: "EMA20/50" },
];

const DAY_INDICATORS = [
  { key: "rsi" as const,        label: "RSI (デイ)",    description: "V字型 — 逆張り+モメンタム" },
  { key: "macd" as const,       label: "MACD",          description: "移動平均収束拡散" },
  { key: "bollinger" as const,  label: "ボリンジャーバンド", description: "%B + 方向ボーナス" },
  { key: "moving_avg" as const, label: "移動平均線 (デイ)", description: "EMA5/10" },
];

export default function IndicatorBreakdown({ stock, mode = "swing" }: Props) {
  const technicalIndicators = mode === "day" ? DAY_INDICATORS : SWING_INDICATORS;
  const analyst = stock.score_components.analyst;
  const hasAnalyst = stock.analyst_count > 0;

  return (
    <div className="space-y-3">
      {technicalIndicators.map(({ key, label, description }) => {
        const comp = stock.score_components[key];
        const pct = (comp.score / comp.max) * 100;
        const barColor =
          pct >= 70 ? "bg-emerald-500" :
          pct >= 50 ? "bg-yellow-500" :
          pct >= 30 ? "bg-orange-500" :
          "bg-red-500";

        return (
          <div key={key} className="bg-gray-800 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <div>
                <span className="text-white font-medium text-sm">{label}</span>
                <span className="text-gray-500 text-xs ml-2">{description}</span>
              </div>
              <div className="text-right">
                <span className="text-white font-semibold">{comp.score}</span>
                <span className="text-gray-500 text-xs">/{comp.max}</span>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
              <div className={`${barColor} h-2 rounded-full transition-all`} style={{ width: `${pct}%` }} />
            </div>
            <div className="flex justify-between text-xs text-gray-400">
              <span>{signalToJa(comp.signal)}</span>
              {comp.value !== null && (
                <span className="tabular-nums">
                  {key === "rsi" ? `RSI: ${comp.value}` :
                   key === "bollinger" ? `%B: ${comp.value}` :
                   key === "moving_avg" ? `比率: ${comp.value}` :
                   `値: ${comp.value}`}
                </span>
              )}
            </div>
          </div>
        );
      })}

      {/* Analyst recommendation */}
      {hasAnalyst && (
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex justify-between items-center mb-2">
            <div>
              <span className="text-white font-medium text-sm">アナリスト評価</span>
              <span className="text-gray-500 text-xs ml-2">{stock.analyst_count}名の推奨</span>
            </div>
            <div className="text-right">
              <span className="text-white font-semibold">{analyst.score}</span>
              <span className="text-gray-500 text-xs">/30</span>
            </div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
            <div
              className={`${(analyst.score / 30) >= 0.7 ? "bg-emerald-500" : (analyst.score / 30) >= 0.5 ? "bg-yellow-500" : (analyst.score / 30) >= 0.3 ? "bg-orange-500" : "bg-red-500"} h-2 rounded-full transition-all`}
              style={{ width: `${(analyst.score / 30) * 100}%` }}
            />
          </div>
          <div className="text-xs text-gray-400">
            {analystSignalToJa(stock.analyst_signal)}
          </div>
        </div>
      )}

      {/* Large cap bonus indicators */}
      {stock.size === "large" && stock.score_components.high52w && (
        <>
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <div>
                <span className="text-white font-medium text-sm">52週高値距離</span>
                <span className="text-gray-500 text-xs ml-2">大型株ボーナス</span>
              </div>
              <div className="text-right">
                <span className="text-white font-semibold">{stock.score_components.high52w.score}</span>
                <span className="text-gray-500 text-xs">/12</span>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
              <div className="bg-cyan-500 h-2 rounded-full transition-all" style={{ width: `${(stock.score_components.high52w.score / 12) * 100}%` }} />
            </div>
            <div className="text-xs text-gray-400">
              {stock.score_components.high52w.dist_pct !== null ? `高値から −${stock.score_components.high52w.dist_pct}%` : "高値データなし"}
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <div>
                <span className="text-white font-medium text-sm">OBV (出来高圧力)</span>
                <span className="text-gray-500 text-xs ml-2">大型株ボーナス</span>
              </div>
              <div className="text-right">
                <span className="text-white font-semibold">{stock.score_components.obv.score}</span>
                <span className="text-gray-500 text-xs">/8</span>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
              <div className="bg-cyan-500 h-2 rounded-full transition-all" style={{ width: `${(stock.score_components.obv.score / 8) * 100}%` }} />
            </div>
            <div className="text-xs text-gray-400">
              {signalToJa(stock.score_components.obv.signal) || stock.score_components.obv.signal}
            </div>
          </div>
        </>
      )}

      {/* Aroon bonus: large cap (swing) or all sizes (day trade) */}
      {(stock.size === "large" || mode === "day") && stock.score_components.aroon && stock.score_components.aroon.score > 0 && (
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex justify-between items-center mb-2">
            <div>
              <span className="text-white font-medium text-sm">Aroon (トレンド方向)</span>
              <span className="text-gray-500 text-xs ml-2">{mode === "day" ? "デイトレボーナス" : "大型株ボーナス"}</span>
            </div>
            <div className="text-right">
              <span className="text-white font-semibold">{stock.score_components.aroon.score}</span>
              <span className="text-gray-500 text-xs">/10</span>
            </div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
            <div className="bg-indigo-500 h-2 rounded-full transition-all" style={{ width: `${(stock.score_components.aroon.score / 10) * 100}%` }} />
          </div>
          <div className="text-xs text-gray-400">
            {signalToJa(stock.score_components.aroon.signal)}
          </div>
        </div>
      )}

      {/* Score breakdown note */}
      <div className="text-xs text-gray-600 px-1 space-y-0.5">
        {mode === "swing" ? (
          <>
            {stock.size === "large" && <p>大型株 (MACD/MA重視): MACD×1.283 + MA×1.222 + RSI×0.776 + BB×0.719</p>}
            {stock.size === "mid"   && <p>中型株 (MA重視): MA×1.204 + MACD×1.057 + BB×0.858 + RSI×0.881</p>}
            {stock.size === "small" && <p>小型株 (RSI/BB重視): RSI×1.122 + BB×1.001 + MA×1.021 + MACD×0.856</p>}
          </>
        ) : (
          <>
            {stock.size === "large" && <p>大型株デイトレ (MA最強): MA×1.303 + MACD×1.100 + RSI×0.897 + BB×0.700</p>}
            {stock.size === "mid"   && <p>中型株デイトレ (MA突出): MA×1.418 + RSI×1.104 + MACD×1.041 + BB×0.437</p>}
            {stock.size === "small" && <p>小型株デイトレ (バランス型): MA×1.072 + MACD×1.026 + BB×0.986 + RSI×0.917</p>}
          </>
        )}
        <p>{hasAnalyst ? "合計 = テクニカル×70% + アナリスト評価×30%" : "合計 = テクニカルスコア (アナリストデータなし)"}</p>
      </div>
    </div>
  );
}
