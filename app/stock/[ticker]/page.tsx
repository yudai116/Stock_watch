"use client";
import { use, useState } from "react";
import Link from "next/link";
import { useStockDetail } from "@/hooks/useStockDetail";
import ScoreGauge from "@/components/ScoreGauge";
import IndicatorBreakdown from "@/components/IndicatorBreakdown";
import PriceChart from "@/components/PriceChart";
import ModeToggle, { type TradeMode } from "@/components/ModeToggle";
import MacroRegimeBanner from "@/components/MacroRegimeBanner";
import ScoreExplainButton from "@/components/ScoreExplainButton";
import { formatPrice, formatChangePct, peLabelColor, yahooFinanceUrls } from "@/lib/formatters";
import SellTargetCard from "@/components/SellTargetCard";

interface Props {
  params: Promise<{ ticker: string }>;
}

export default function StockDetailPage({ params }: Props) {
  const { ticker } = use(params);
  const [tradeMode, setTradeMode] = useState<TradeMode>("swing");
  const { detail, isLoading, error } = useStockDetail(decodeURIComponent(ticker), tradeMode);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-4">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-gray-400">{decodeURIComponent(ticker)} のデータを取得中...</p>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="text-center py-24">
        <p className="text-red-400 mb-4">{error?.message ?? "データの取得に失敗しました"}</p>
        <Link href="/" className="text-blue-400 hover:text-blue-300">← ウォッチリストへ戻る</Link>
      </div>
    );
  }

  const changePctPositive = detail.change_pct !== null && detail.change_pct >= 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-gray-500 hover:text-gray-300 text-sm">← 戻る</Link>
          <span className="text-gray-700">/</span>
          <span className="text-gray-400 text-sm font-mono">{detail.ticker}</span>
        </div>
        <ModeToggle mode={tradeMode} onChange={setTradeMode} />
      </div>

      {/* Macro regime banner */}
      <MacroRegimeBanner macro={detail.macro} />

      {/* ヘッダー */}
      <div className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
        <div className="flex flex-col lg:flex-row lg:items-center gap-6">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <span className={`text-xs px-2 py-0.5 rounded font-medium ${detail.market === "JP" ? "bg-red-900/50 text-red-300" : "bg-blue-900/50 text-blue-300"}`}>
                {detail.market}
              </span>
              <span className="font-mono font-bold text-xl text-white">{detail.ticker}</span>
            </div>
            <h1 className="text-gray-300 text-lg mb-4">{detail.name}</h1>
            <div className="flex items-baseline gap-4">
              <span className="text-3xl font-bold text-white tabular-nums">
                {formatPrice(detail.price, detail.currency)}
              </span>
              <span className={`text-lg font-medium ${changePctPositive ? "text-emerald-400" : "text-red-400"}`}>
                {formatChangePct(detail.change_pct)}
              </span>
            </div>
            <div className="flex gap-6 mt-4 text-sm">
              <div>
                <span className="text-gray-500">PER (実績)</span>
                <p className="text-white font-mono">{detail.trailing_pe?.toFixed(1) ?? "N/A"}</p>
              </div>
              <div>
                <span className="text-gray-500">PER (予想)</span>
                <p className="text-white font-mono">{detail.forward_pe?.toFixed(1) ?? "N/A"}</p>
              </div>
              <div>
                <span className="text-gray-500">PER評価</span>
                <p className={`font-medium ${peLabelColor(detail.pe_label)}`}>{detail.pe_label ?? "N/A"}</p>
              </div>
              <div>
                <span className="text-gray-500">最終更新</span>
                <p className="text-gray-400 text-xs mt-0.5">
                  {new Date(detail.last_updated).toLocaleString("ja-JP")}
                </p>
              </div>
            </div>
            {/* Yahoo Finance links */}
            <div className="flex flex-wrap gap-2 mt-4 items-center">
              <span className="text-xs text-gray-600">Yahoo Finance:</span>
              {yahooFinanceUrls(detail.ticker).map(({ label, href, kind }) => (
                <a
                  key={href}
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors ${
                    kind === "chart"
                      ? "bg-blue-950/60 border border-blue-800/50 text-blue-400 hover:bg-blue-900/60 hover:text-blue-200"
                      : "bg-gray-800 border border-gray-700 text-gray-400 hover:bg-gray-700 hover:text-gray-200"
                  }`}
                >
                  {kind === "chart" ? (
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                    </svg>
                  ) : (
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  )}
                  {label}
                </a>
              ))}
            </div>
          </div>
          <div className="flex flex-col items-center gap-2">
            <ScoreGauge score={detail.score} />
            <ScoreExplainButton mode={tradeMode} />
          </div>
        </div>
      </div>

      {/* 指標内訳 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h2 className="text-white font-semibold mb-3">指標スコア内訳</h2>
          <IndicatorBreakdown stock={detail} mode={tradeMode} />
        </div>
        <div>
          <h2 className="text-white font-semibold mb-3">スコアの見方</h2>
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800 space-y-3 text-sm text-gray-400">
            {tradeMode === "swing" ? (
              <>
                <p><span className="text-orange-400 font-medium">移動平均線 (最大25点)</span>: EMA20上抜け・ゴールデンクロス帯で高得点。大型・中型株で重視。</p>
                <p><span className="text-blue-400 font-medium">MACD (最大25点)</span>: シグナル線上抜け（ゴールデンクロス）で最高点。大型株のトレンド追従に有効。</p>
                <p><span className="text-emerald-400 font-medium">RSI (最大25点)</span>: 売られすぎからの回復で高得点。小型株の逆張りで特に有効。</p>
                <p><span className="text-purple-400 font-medium">ボリンジャーバンド (最大25点)</span>: 下限帯付近・突破で高得点。%Bで計算。小型株で重視。</p>
              </>
            ) : (
              <>
                <p><span className="text-orange-400 font-medium">移動平均線 (最大25点)</span>: EMA5上抜けで即反応。短期モメンタムを優先。</p>
                <p><span className="text-blue-400 font-medium">MACD (最大25点)</span>: ゴールデンクロスとヒストグラム方向でモメンタム判定。</p>
                <p><span className="text-emerald-400 font-medium">RSI (最大25点)</span>: V字型採点 — 売られすぎ反発と上昇モメンタム(RSI 55〜70)を両方評価。</p>
                <p><span className="text-purple-400 font-medium">ボリンジャーバンド (最大25点)</span>: %B絶対値に加え、方向ボーナス（上昇中+3点）でモメンタム対応。</p>
              </>
            )}
            <p><span className="text-cyan-400 font-medium">アナリスト評価 (最大30点)</span>: 機関投資家推奨の買い/売り比率をスコア化。</p>
            <div className="border-t border-gray-800 pt-3 text-xs space-y-1">
              {tradeMode === "swing" ? (
                <>
                  {detail.size === "large" && <p className="text-gray-500">大型株: MACD×1.283 + MA×1.222 + RSI×0.776 + BB×0.719</p>}
                  {detail.size === "mid"   && <p className="text-gray-500">中型株: MA×1.204 + MACD×1.057 + BB×0.858 + RSI×0.881</p>}
                  {detail.size === "small" && <p className="text-gray-500">小型株: RSI×1.122 + BB×1.001 + MA×1.021 + MACD×0.856</p>}
                  <p className="text-gray-600">重みは規模別・50銘柄10年MCバックテスト（シャープレシオ最大化）で最適化</p>
                </>
              ) : (
                <>
                  {detail.size === "large" && <p className="text-gray-500">大型株: MA×1.303 + MACD×1.100 + RSI×0.897 + BB×0.700</p>}
                  {detail.size === "mid"   && <p className="text-gray-500">中型株: MA×1.418 + RSI×1.104 + MACD×1.041 + BB×0.437</p>}
                  {detail.size === "small" && <p className="text-gray-500">小型株: MA×1.072 + MACD×1.026 + BB×0.986 + RSI×0.917</p>}
                  <p className="text-gray-600">重みは規模別・50銘柄 hold=1日 MCバックテスト（シャープレシオ最大化）で最適化</p>
                </>
              )}
              <p className="text-gray-500">合計スコア = テクニカル×70% + アナリスト評価×30%</p>
              {detail.macro && detail.macro.multiplier !== 1.0 && (
                <p className={detail.macro.regime === "bullish" ? "text-emerald-600" : detail.macro.regime === "cautious" ? "text-yellow-600" : "text-red-600"}>
                  ※ マクロ補正適用中: {detail.macro.label} (×{detail.macro.multiplier})
                </p>
              )}
              <p className="text-yellow-400 mt-1">⚠️ 投資判断は自己責任でお願いします。</p>
            </div>
          </div>
        </div>
      </div>

      {/* 売りターゲット */}
      {detail.sell_signals && (
        <div>
          <h2 className="text-white font-semibold mb-3">売りターゲット・リスク管理</h2>
          <SellTargetCard sellSignals={detail.sell_signals} currency={detail.currency} currentPrice={detail.price ?? 0} />
        </div>
      )}

      {/* チャート */}
      <div>
        <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
          <h2 className="text-white font-semibold">チャート (90日)</h2>
          <div className="flex items-center gap-2 flex-wrap">
            {yahooFinanceUrls(detail.ticker).map(({ label, href, kind }) => (
              <a
                key={href}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors ${
                  kind === "chart"
                    ? "bg-blue-950/60 border border-blue-800/50 text-blue-400 hover:bg-blue-900/60 hover:text-blue-200"
                    : "bg-gray-800 border border-gray-700 text-gray-400 hover:bg-gray-700 hover:text-gray-200"
                }`}
              >
                {kind === "chart" ? (
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                ) : (
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                )}
                {label}
              </a>
            ))}
          </div>
        </div>
        <PriceChart detail={detail} />
      </div>
    </div>
  );
}
