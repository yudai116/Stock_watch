"use client";
import type { StockScore } from "@/types";
import { signalToJa, analystSignalToJa } from "@/lib/formatters";
import { SWING_WEIGHTS, DAY_WEIGHTS } from "@/lib/backtestWeights";

interface Props {
  stock: StockScore;
  mode?: "swing" | "day";
}

const SWING_INDICATORS = [
  { key: "rsi" as const,        label: "RSI",               description: "相対力指数(14) — 逆張り" },
  { key: "macd" as const,       label: "MACD",              description: "移動平均収束拡散" },
  { key: "bollinger" as const,  label: "ボリンジャーバンド",   description: "価格帯位置(%B)" },
  { key: "moving_avg" as const, label: "移動平均線",          description: "EMA20/50" },
  { key: "aroon" as const,      label: "Aroon",             description: "トレンド方向と強度(25期)" },
  { key: "stoch" as const,      label: "ストキャスティクス",   description: "K/D ライン(14/3)" },
  { key: "cci" as const,        label: "CCI",               description: "商品チャンネル指数(20)" },
  { key: "roc" as const,        label: "ROC",               description: "変化率(10日)" },
];

const DAY_INDICATORS = [
  { key: "rsi" as const,        label: "RSI (デイ)",         description: "V字型 — 逆張り+モメンタム" },
  { key: "macd" as const,       label: "MACD",              description: "移動平均収束拡散" },
  { key: "bollinger" as const,  label: "ボリンジャーバンド",   description: "%B + 方向ボーナス" },
  { key: "moving_avg" as const, label: "移動平均線 (デイ)",   description: "EMA5/10" },
  { key: "stoch" as const,      label: "ストキャスティクス",   description: "K/D ライン(14/3)" },
  { key: "cci" as const,        label: "CCI",               description: "商品チャンネル指数(20)" },
  { key: "roc" as const,        label: "ROC",               description: "変化率(10日)" },
];

export default function IndicatorBreakdown({ stock, mode = "swing" }: Props) {
  const technicalIndicators = mode === "day" ? DAY_INDICATORS : SWING_INDICATORS;
  const analyst = stock.score_components.analyst;
  const hasAnalyst = stock.analyst_count > 0;

  return (
    <div className="space-y-3">
      {technicalIndicators.map(({ key, label, description }) => {
        const comp = stock.score_components[key];
        if (!comp) return null;
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
                   key === "aroon" ? `Up-Down: ${comp.value}` :
                   key === "stoch" ? `%K: ${comp.value}` :
                   key === "cci" ? `CCI: ${comp.value}` :
                   key === "roc" ? `ROC: ${comp.value}%` :
                   `値: ${comp.value}`}
                </span>
              )}
            </div>
          </div>
        );
      })}

      {/* Day mode: Aroon shown as informational only (not in day GA scoring) */}
      {mode === "day" && stock.score_components.aroon && (
        <div className="bg-gray-800 rounded-lg p-3 opacity-60">
          <div className="flex justify-between items-center mb-2">
            <div>
              <span className="text-gray-400 font-medium text-sm">参考: アルーン</span>
              <span className="text-gray-600 text-xs ml-2">デイトレスコア対象外</span>
            </div>
            <div className="text-right">
              <span className="text-gray-400 font-semibold">{stock.score_components.aroon.score}</span>
              <span className="text-gray-600 text-xs">/25</span>
            </div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
            <div className="bg-gray-500 h-2 rounded-full transition-all" style={{ width: `${(stock.score_components.aroon.score / 25) * 100}%` }} />
          </div>
          <div className="text-xs text-gray-500">
            {signalToJa(stock.score_components.aroon.signal)}
          </div>
        </div>
      )}

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

      {/* Score breakdown note */}
      <div className="text-xs text-gray-600 px-1 space-y-0.5">
        {mode === "swing" ? (
          <p>スイングGA最適化 (v7): ROC×{SWING_WEIGHTS.ROC} + Aroon×{SWING_WEIGHTS.Aroon} + RSI×{SWING_WEIGHTS.RSI} + MACD×{SWING_WEIGHTS.MACD} + BB×{SWING_WEIGHTS.BB} + CCI×{SWING_WEIGHTS.CCI} + Stoch×{SWING_WEIGHTS.Stoch} + MA×{SWING_WEIGHTS.MA}</p>
        ) : (
          <p>デイトレGA最適化 (v7): BB×{DAY_WEIGHTS.BB} + Stoch×{DAY_WEIGHTS.Stoch} + RSI×{DAY_WEIGHTS.RSI} + MACD×{DAY_WEIGHTS.MACD} + MA×{DAY_WEIGHTS.MA} + ROC×{DAY_WEIGHTS.ROC} + CCI×{DAY_WEIGHTS.CCI}</p>
        )}
        <p>{hasAnalyst ? "合計 = テクニカル×70% + アナリスト評価×30%" : "合計 = テクニカルスコア (アナリストデータなし)"}</p>
      </div>
    </div>
  );
}
