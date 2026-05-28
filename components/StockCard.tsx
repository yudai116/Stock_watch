"use client";
import Link from "next/link";
import type { StockScore } from "@/types";
import { formatPrice, formatChangePct, analystSignalToJa, analystSignalColor, sectorBadgeColor, yahooFinanceUrls } from "@/lib/formatters";
import { getSector } from "@/lib/sectors";

function ScoreRing({ score }: { score: number }) {
  const ringColor =
    score >= 70 ? "border-emerald-400" :
    score >= 50 ? "border-yellow-400" :
    score >= 30 ? "border-orange-400" :
    "border-red-400";
  const textColor =
    score >= 70 ? "text-emerald-400" :
    score >= 50 ? "text-yellow-400" :
    score >= 30 ? "text-orange-400" :
    "text-red-400";
  const label =
    score >= 70 ? "強い買い" :
    score >= 50 ? "要観察" :
    score >= 30 ? "弱い" :
    "回避";

  return (
    <div className="flex flex-col items-center gap-1">
      <div className={`w-16 h-16 rounded-full border-4 ${ringColor} flex items-center justify-center bg-gray-950`}>
        <span className={`text-2xl font-black tabular-nums ${textColor}`}>{score}</span>
      </div>
      <span className={`text-xs font-medium ${textColor}`}>{label}</span>
    </div>
  );
}

function MiniBar({ label, score, max }: { label: string; score: number; max: number }) {
  const pct = Math.max(0, Math.min(100, (score / max) * 100));
  const barColor =
    pct >= 70 ? "bg-emerald-500" :
    pct >= 50 ? "bg-yellow-500" :
    pct >= 30 ? "bg-orange-500" :
    "bg-red-500";

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-500 w-24 flex-shrink-0 truncate">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-1.5">
        <div className={`${barColor} h-1.5 rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-gray-500 tabular-nums w-10 text-right">{score}/{max}</span>
    </div>
  );
}

interface Props {
  stock: StockScore;
  onRemove: (ticker: string) => Promise<unknown>;
}

export default function StockCard({ stock, onRemove }: Props) {
  const sector = getSector(stock.ticker);
  const sectorClass = sectorBadgeColor(sector.color);

  return (
    <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
      <div className="p-4">
        <div className="flex items-start gap-3">
          {/* Left: name + price info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-2 flex-wrap">
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${sectorClass}`}>
                {stock.sector_label || sector.label}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                stock.market === "JP"
                  ? "bg-red-900/60 text-red-300"
                  : "bg-blue-900/60 text-blue-300"
              }`}>
                {stock.market}
              </span>
            </div>

            <Link href={`/stock/${stock.ticker}`} className="block group">
              <h3 className="font-bold text-white leading-tight truncate group-hover:text-blue-300 transition-colors">
                {stock.name}
              </h3>
              <p className="text-gray-500 text-xs font-mono mt-0.5">{stock.ticker}</p>
            </Link>

            <div className="flex items-baseline gap-2 mt-2">
              <span className="text-white font-mono font-medium">
                {formatPrice(stock.price, stock.currency)}
              </span>
              <span className={`text-sm font-mono ${
                stock.change_pct !== null && stock.change_pct >= 0
                  ? "text-emerald-400"
                  : "text-red-400"
              }`}>
                {formatChangePct(stock.change_pct)}
              </span>
            </div>
          </div>

          {/* Right: score ring + remove */}
          <div className="flex flex-col items-end gap-3 flex-shrink-0">
            <ScoreRing score={stock.score} />
            <button
              onClick={() => onRemove(stock.ticker)}
              className="text-gray-700 hover:text-red-400 transition-colors text-lg leading-none px-1"
              title="削除"
            >
              ×
            </button>
          </div>
        </div>
      </div>

      {/* Indicator bars */}
      <div className="px-4 pb-4 space-y-1.5 border-t border-gray-800 pt-3">
        <MiniBar label="RSI" score={stock.score_components.rsi.score} max={stock.score_components.rsi.max} />
        <MiniBar label="MACD" score={stock.score_components.macd.score} max={stock.score_components.macd.max} />
        <MiniBar label="BB" score={stock.score_components.bollinger.score} max={stock.score_components.bollinger.max} />
        <MiniBar label="移動平均" score={stock.score_components.moving_avg.score} max={stock.score_components.moving_avg.max} />
        {stock.analyst_count > 0 && (
          <MiniBar
            label={`アナリスト(${stock.analyst_count}名)`}
            score={stock.score_components.analyst.score}
            max={30}
          />
        )}
      </div>

      {/* Analyst signal footer */}
      {stock.analyst_count > 0 && (
        <div className="px-4 py-2 bg-gray-900/50 border-t border-gray-800 flex items-center justify-between text-xs">
          <span className="text-gray-600">アナリスト評価</span>
          <span className={`font-medium ${analystSignalColor(stock.analyst_signal)}`}>
            {analystSignalToJa(stock.analyst_signal)}
          </span>
        </div>
      )}

      {/* Yahoo Finance links */}
      <div className="px-4 py-2 border-t border-gray-800 flex items-center gap-2 flex-wrap">
        <span className="text-gray-700 text-xs flex-shrink-0">Yahoo Finance:</span>
        {yahooFinanceUrls(stock.ticker).map(({ label, href, kind }) => (
          <a
            key={href}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs transition-colors ${
              kind === "chart"
                ? "bg-blue-950/60 text-blue-400 hover:bg-blue-900/60 hover:text-blue-200"
                : "bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-300"
            }`}
          >
            {kind === "chart" && (
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
            )}
            {label}
          </a>
        ))}
      </div>
    </div>
  );
}
