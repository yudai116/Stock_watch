"use client";
import { useState, useMemo } from "react";
import Link from "next/link";
import type { StockScore } from "@/types";
import ScoreBadge from "./ScoreBadge";
import { formatPrice, formatChangePct, peLabelColor, signalToJa } from "@/lib/formatters";

type SortKey = "score" | "ticker" | "price" | "change_pct" | "trailing_pe";
type Market = "ALL" | "JP" | "US";

interface Props {
  stocks: StockScore[];
  onRemove: (ticker: string) => Promise<unknown>;
}

export default function WatchlistTable({ stocks, onRemove }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortAsc, setSortAsc] = useState(false);
  const [market, setMarket] = useState<Market>("ALL");
  const [removing, setRemoving] = useState<string | null>(null);

  function handleSort(key: SortKey) {
    if (sortKey === key) setSortAsc((v) => !v);
    else { setSortKey(key); setSortAsc(false); }
  }

  const filtered = useMemo(() => {
    let list = market === "ALL" ? stocks : stocks.filter((s) => s.market === market);
    list = [...list].sort((a, b) => {
      let av: number | string = sortKey === "ticker" ? a.ticker : (a[sortKey] ?? -Infinity) as number;
      let bv: number | string = sortKey === "ticker" ? b.ticker : (b[sortKey] ?? -Infinity) as number;
      if (typeof av === "string" && typeof bv === "string") {
        return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
    });
    return list;
  }, [stocks, sortKey, sortAsc, market]);

  function SortArrow({ col }: { col: SortKey }) {
    if (sortKey !== col) return <span className="text-gray-600 ml-1">↕</span>;
    return <span className="text-blue-400 ml-1">{sortAsc ? "↑" : "↓"}</span>;
  }

  function Th({ col, label }: { col: SortKey; label: string }) {
    return (
      <th className="text-left py-3 px-3 text-gray-400 font-medium text-xs cursor-pointer hover:text-white whitespace-nowrap" onClick={() => handleSort(col)}>
        {label}<SortArrow col={col} />
      </th>
    );
  }

  async function handleRemove(ticker: string) {
    setRemoving(ticker);
    try { await onRemove(ticker); } finally { setRemoving(null); }
  }

  return (
    <div>
      <div className="flex gap-2 mb-4">
        {(["ALL", "JP", "US"] as Market[]).map((m) => (
          <button
            key={m}
            onClick={() => setMarket(m)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              market === m ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            {m === "ALL" ? "全て" : m}
            <span className="ml-1.5 text-xs opacity-70">
              ({m === "ALL" ? stocks.length : stocks.filter((s) => s.market === m).length})
            </span>
          </button>
        ))}
      </div>

      <div className="overflow-x-auto rounded-xl border border-gray-700">
        <table className="w-full text-sm">
          <thead className="bg-gray-800/80 border-b border-gray-700">
            <tr>
              <Th col="ticker" label="ティッカー" />
              <th className="text-left py-3 px-3 text-gray-400 font-medium text-xs">銘柄名</th>
              <th className="text-left py-3 px-3 text-gray-400 font-medium text-xs">市場</th>
              <Th col="price" label="株価" />
              <Th col="change_pct" label="前日比" />
              <Th col="score" label="スコア" />
              <th className="text-left py-3 px-3 text-gray-400 font-medium text-xs">RSI</th>
              <th className="text-left py-3 px-3 text-gray-400 font-medium text-xs">MACD</th>
              <th className="text-left py-3 px-3 text-gray-400 font-medium text-xs">BB</th>
              <Th col="trailing_pe" label="PER(実績)" />
              <th className="text-left py-3 px-3 text-gray-400 font-medium text-xs">PER評価</th>
              <th className="py-3 px-3" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((s) => (
              <tr key={s.ticker} className="border-b border-gray-800 hover:bg-gray-800/40 transition-colors group">
                <td className="py-3 px-3">
                  <Link href={`/stock/${s.ticker}`} className="text-blue-400 hover:text-blue-300 font-mono font-medium">
                    {s.ticker}
                  </Link>
                </td>
                <td className="py-3 px-3 text-gray-300 max-w-[180px] truncate">{s.name}</td>
                <td className="py-3 px-3">
                  <span className={`text-xs px-2 py-0.5 rounded font-medium ${s.market === "JP" ? "bg-red-900/50 text-red-300" : "bg-blue-900/50 text-blue-300"}`}>
                    {s.market}
                  </span>
                </td>
                <td className="py-3 px-3 text-white font-mono tabular-nums whitespace-nowrap">
                  {formatPrice(s.price, s.currency)}
                </td>
                <td className={`py-3 px-3 font-mono tabular-nums ${s.change_pct !== null && s.change_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {formatChangePct(s.change_pct)}
                </td>
                <td className="py-3 px-3">
                  <ScoreBadge score={s.score} />
                </td>
                <td className="py-3 px-3 text-gray-300 text-xs whitespace-nowrap">
                  {s.score_components.rsi.value !== null ? (
                    <span title={signalToJa(s.score_components.rsi.signal)}>
                      {s.score_components.rsi.value}
                    </span>
                  ) : "—"}
                </td>
                <td className="py-3 px-3 text-gray-300 text-xs whitespace-nowrap">
                  <span title={signalToJa(s.score_components.macd.signal)}>
                    {signalToJa(s.score_components.macd.signal)}
                  </span>
                </td>
                <td className="py-3 px-3 text-gray-300 text-xs whitespace-nowrap">
                  {s.score_components.bollinger.value !== null ? (
                    <span title={signalToJa(s.score_components.bollinger.signal)}>
                      {s.score_components.bollinger.value}
                    </span>
                  ) : "—"}
                </td>
                <td className="py-3 px-3 text-gray-300 font-mono tabular-nums">
                  {s.trailing_pe !== null ? s.trailing_pe.toFixed(1) : "N/A"}
                </td>
                <td className={`py-3 px-3 text-xs font-medium ${peLabelColor(s.pe_label)}`}>
                  {s.pe_label ?? "N/A"}
                </td>
                <td className="py-3 px-3">
                  <button
                    onClick={() => handleRemove(s.ticker)}
                    disabled={removing === s.ticker}
                    className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 transition-all text-lg leading-none"
                    title="削除"
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={12} className="py-12 text-center text-gray-500">
                  銘柄がありません
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
