"use client";
import type { StockScore } from "@/types";
import { signalToJa, analystSignalToJa } from "@/lib/formatters";

interface Props {
  stock: StockScore;
}

const technicalIndicators = [
  { key: "rsi" as const,        label: "RSI",           description: "相対力指数(14)" },
  { key: "macd" as const,       label: "MACD",          description: "移動平均収束拡散" },
  { key: "bollinger" as const,  label: "ボリンジャーバンド", description: "価格帯位置(%B)" },
  { key: "moving_avg" as const, label: "移動平均線",    description: "EMA20/50" },
];

export default function IndicatorBreakdown({ stock }: Props) {
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
      <div className="text-xs text-gray-600 px-1">
        {hasAnalyst
          ? "スコア = テクニカル×70% + アナリスト評価×30%"
          : "スコア = テクニカル指標の合計 (アナリストデータなし)"}
      </div>
    </div>
  );
}
