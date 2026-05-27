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

      {/* Score breakdown note */}
      <div className="text-xs text-gray-600 px-1 space-y-0.5">
        {mode === "swing" ? (
          <>
            {stock.size === "large" && <p>大型株 (MACD/MA重視): RSI×0.722 + MACD×1.274 + BB×0.795 + MA×1.209</p>}
            {stock.size === "mid"   && <p>中型株 (MA重視): RSI×0.842 + MACD×1.021 + BB×0.956 + MA×1.182</p>}
            {stock.size === "small" && <p>小型株 (RSI/BB重視): RSI×1.086 + MACD×0.913 + BB×1.001 + MA×1.000</p>}
          </>
        ) : (
          <>
            {stock.size === "large" && <p>大型株デイトレ (MA/MACD重視): RSI×0.885 + MACD×1.109 + BB×0.743 + MA×1.264</p>}
            {stock.size === "mid"   && <p>中型株デイトレ (MA/MACD重視): RSI×0.966 + MACD×1.092 + BB×0.757 + MA×1.185</p>}
            {stock.size === "small" && <p>小型株デイトレ (バランス型): RSI×0.911 + MACD×1.015 + BB×0.969 + MA×1.105</p>}
          </>
        )}
        <p>{hasAnalyst ? "合計 = テクニカル×70% + アナリスト評価×30%" : "合計 = テクニカルスコア (アナリストデータなし)"}</p>
      </div>
    </div>
  );
}
