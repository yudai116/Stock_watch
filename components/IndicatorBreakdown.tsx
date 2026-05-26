"use client";
import type { StockScore } from "@/types";
import { signalToJa } from "@/lib/formatters";

interface Props {
  stock: StockScore;
}

const indicators = [
  { key: "rsi" as const, label: "RSI", description: "相対力指数" },
  { key: "macd" as const, label: "MACD", description: "移動平均収束拡散" },
  { key: "bollinger" as const, label: "ボリンジャーバンド", description: "価格帯位置" },
  { key: "moving_avg" as const, label: "移動平均線", description: "EMA20/50" },
];

export default function IndicatorBreakdown({ stock }: Props) {
  return (
    <div className="space-y-3">
      {indicators.map(({ key, label, description }) => {
        const comp = stock.score_components[key];
        const pct = (comp.score / comp.max) * 100;
        const barColor = pct >= 70 ? "bg-emerald-500" : pct >= 50 ? "bg-yellow-500" : pct >= 30 ? "bg-orange-500" : "bg-red-500";

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
    </div>
  );
}
