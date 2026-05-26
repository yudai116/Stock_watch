"use client";
import { useState } from "react";
import { useWatchlist } from "@/hooks/useWatchlist";
import WatchlistTable from "@/components/WatchlistTable";
import AddTickerForm from "@/components/AddTickerForm";
import AutoRefreshToggle from "@/components/AutoRefreshToggle";

export default function Dashboard() {
  const [interval, setInterval] = useState(300_000);
  const { stocks, isLoading, addTicker, removeTicker, mutate, errors } = useWatchlist(interval);

  const errorEntries = Object.entries(errors);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">ウォッチリスト</h1>
          <p className="text-gray-400 text-sm mt-1">
            スコアが高いほどスイングトレードの買いエントリー機会
          </p>
        </div>
        <div className="flex flex-col gap-3">
          <AddTickerForm onAdd={addTicker} />
          <AutoRefreshToggle interval={interval} onIntervalChange={setInterval} onRefresh={() => mutate()} />
        </div>
      </div>

      {/* スコア凡例 */}
      <div className="flex flex-wrap gap-3 text-xs">
        {[
          { range: "70〜100", label: "強い買いシグナル", color: "bg-emerald-500" },
          { range: "50〜69", label: "中立/要観察", color: "bg-yellow-500" },
          { range: "30〜49", label: "弱い", color: "bg-orange-500" },
          { range: "0〜29", label: "回避/弱気", color: "bg-red-500" },
        ].map((item) => (
          <div key={item.range} className="flex items-center gap-1.5">
            <span className={`${item.color} w-3 h-3 rounded-full`} />
            <span className="text-gray-400">{item.range}点: {item.label}</span>
          </div>
        ))}
      </div>

      {errorEntries.length > 0 && (
        <div className="bg-red-950/50 border border-red-800 rounded-lg p-3 text-sm">
          <p className="text-red-300 font-medium mb-1">取得エラー:</p>
          {errorEntries.map(([ticker, msg]) => (
            <p key={ticker} className="text-red-400 text-xs">{ticker}: {msg}</p>
          ))}
        </div>
      )}

      {isLoading && stocks.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-400">株価データを取得中...</p>
          <p className="text-gray-600 text-xs">Yahoo Financeから情報を取得しています</p>
        </div>
      ) : (
        <WatchlistTable stocks={stocks} onRemove={removeTicker} />
      )}

      <p className="text-center text-gray-600 text-xs pt-4">
        ※ このツールは情報提供のみを目的としています。投資判断は自己責任でお願いします。
      </p>
    </div>
  );
}
