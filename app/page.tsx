"use client";
import { useState, useMemo } from "react";
import { useWatchlist } from "@/hooks/useWatchlist";
import StockCard from "@/components/StockCard";
import AddTickerForm from "@/components/AddTickerForm";
import AutoRefreshToggle from "@/components/AutoRefreshToggle";
import ModeToggle, { type TradeMode } from "@/components/ModeToggle";
import { SECTOR_LABELS, getSector, type SectorKey } from "@/lib/sectors";

type Filter = "ALL" | SectorKey;
type SortMode = "score" | "change_pct";

const SECTOR_FILTERS: { key: Filter; label: string }[] = [
  { key: "ALL", label: SECTOR_LABELS.ALL },
  { key: "semiconductor", label: SECTOR_LABELS.semiconductor },
  { key: "memory", label: SECTOR_LABELS.memory },
  { key: "ai", label: SECTOR_LABELS.ai },
  { key: "quantum", label: SECTOR_LABELS.quantum },
  { key: "nuclear", label: SECTOR_LABELS.nuclear },
];

export default function Dashboard() {
  const [refreshInterval, setRefreshInterval] = useState(300_000);
  const [filter, setFilter] = useState<Filter>("ALL");
  const [sortMode, setSortMode] = useState<SortMode>("score");
  const [tradeMode, setTradeMode] = useState<TradeMode>("swing");
  const { stocks, isLoading, addTicker, removeTicker, mutate, errors } = useWatchlist(refreshInterval, tradeMode);

  const filtered = useMemo(() => {
    let list =
      filter === "ALL"
        ? stocks
        : stocks.filter((s) => (s.sector ?? getSector(s.ticker).key) === filter);
    return [...list].sort((a, b) => {
      if (sortMode === "score") return b.score - a.score;
      return (b.change_pct ?? -Infinity) - (a.change_pct ?? -Infinity);
    });
  }, [stocks, filter, sortMode]);

  const errorEntries = Object.entries(errors);

  return (
    <div className="space-y-4">
      {/* Title + mode toggle */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">テーマ株 監視</h1>
          <p className="text-gray-500 text-xs mt-0.5">
            半導体・メモリ・AI・量子・核融合 ─{" "}
            {tradeMode === "day" ? (
              <span className="text-orange-400">デイトレードスコア (EMA5/10・モメンタムRSI)</span>
            ) : (
              "スイングトレードスコア"
            )}
          </p>
        </div>
        <ModeToggle mode={tradeMode} onChange={(m) => { setTradeMode(m); mutate(); }} />
      </div>

      {/* Add ticker form */}
      <AddTickerForm onAdd={addTicker} />

      {/* Sector filter tabs */}
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none">
        {SECTOR_FILTERS.map(({ key, label }) => {
          const count =
            key === "ALL"
              ? stocks.length
              : stocks.filter((s) => (s.sector ?? getSector(s.ticker).key) === key).length;
          return (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={`flex-shrink-0 px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                filter === key
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-white"
              }`}
            >
              {label}
              {count > 0 && (
                <span className="ml-1 opacity-60">({count})</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Sort + refresh controls */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex gap-1">
          <button
            onClick={() => setSortMode("score")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              sortMode === "score" ? "bg-gray-700 text-white" : "text-gray-500 hover:text-white"
            }`}
          >
            スコア順
          </button>
          <button
            onClick={() => setSortMode("change_pct")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              sortMode === "change_pct" ? "bg-gray-700 text-white" : "text-gray-500 hover:text-white"
            }`}
          >
            前日比順
          </button>
        </div>
        <AutoRefreshToggle
          interval={refreshInterval}
          onIntervalChange={setRefreshInterval}
          onRefresh={() => mutate()}
        />
      </div>

      {/* Score legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
        {[
          { dot: "bg-emerald-500", text: "70+ 強い買い" },
          { dot: "bg-yellow-500",  text: "50+ 要観察" },
          { dot: "bg-orange-500",  text: "30+ 弱い" },
          { dot: "bg-red-500",     text: "〜29 回避" },
        ].map((item) => (
          <div key={item.text} className="flex items-center gap-1.5">
            <span className={`${item.dot} w-2 h-2 rounded-full flex-shrink-0`} />
            {item.text}
          </div>
        ))}
      </div>

      {/* Errors */}
      {errorEntries.length > 0 && (
        <div className="bg-red-950/50 border border-red-800 rounded-xl p-3 text-xs">
          <p className="text-red-300 font-medium mb-1">取得エラー:</p>
          {errorEntries.map(([ticker, msg]) => (
            <p key={ticker} className="text-red-400">{ticker}: {msg}</p>
          ))}
        </div>
      )}

      {/* Loading */}
      {isLoading && stocks.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-400 text-sm">株価データ取得中...</p>
          <p className="text-gray-600 text-xs">Yahoo Finance から取得しています</p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((s) => (
            <StockCard key={s.ticker} stock={s} onRemove={removeTicker} />
          ))}
          {filtered.length === 0 && stocks.length > 0 && (
            <div className="py-10 text-center text-gray-500 text-sm">
              このセクターの銘柄はありません
            </div>
          )}
        </div>
      )}

      <p className="text-center text-gray-700 text-xs pb-4">
        ※ 情報提供のみ。投資判断は自己責任でお願いします。
      </p>
    </div>
  );
}
