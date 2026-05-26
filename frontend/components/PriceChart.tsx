"use client";
import {
  ComposedChart, Line, Area, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from "recharts";
import type { StockDetail } from "@/types";

interface Props {
  detail: StockDetail;
}

export default function PriceChart({ detail }: Props) {
  const { history, rsi_series, macd_series, bb_series, ma20_series, ma50_series } = detail;

  const priceData = history.map((h) => {
    const rsi = rsi_series.find((r) => r.date === h.date);
    const bb = bb_series.find((b) => b.date === h.date);
    const ma20 = ma20_series.find((m) => m.date === h.date);
    const ma50 = ma50_series.find((m) => m.date === h.date);
    return {
      date: h.date,
      close: h.close,
      volume: h.volume,
      rsi: rsi?.value ?? null,
      bb_upper: bb?.upper ?? null,
      bb_middle: bb?.middle ?? null,
      bb_lower: bb?.lower ?? null,
      ma20: ma20?.value ?? null,
      ma50: ma50?.value ?? null,
    };
  });

  const macdData = macd_series.map((m) => ({
    date: m.date,
    macd: m.macd,
    signal: m.signal,
    histogram: m.histogram,
  }));

  const tickFormatter = (date: string) => {
    const d = new Date(date);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  };

  const priceRange = priceData.filter((d) => d.close);
  const minPrice = Math.min(...priceRange.map((d) => d.bb_lower ?? d.close).filter(Boolean)) * 0.98;
  const maxPrice = Math.max(...priceRange.map((d) => d.bb_upper ?? d.close).filter(Boolean)) * 1.02;

  return (
    <div className="space-y-4">
      {/* Price + BB + MA */}
      <div className="bg-gray-800 rounded-xl p-4">
        <h3 className="text-sm text-gray-400 mb-3">株価 / ボリンジャーバンド / 移動平均線</h3>
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={priceData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tickFormatter={tickFormatter} tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <YAxis domain={[minPrice, maxPrice]} tick={{ fill: "#9ca3af", fontSize: 11 }} width={70}
              tickFormatter={(v) => detail.currency === "JPY" ? `¥${v.toFixed(0)}` : `$${v.toFixed(2)}`} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: "8px" }}
              labelStyle={{ color: "#e5e7eb" }}
              formatter={(value: unknown, name: unknown) => {
                const labels: Record<string, string> = { close: "終値", ma20: "MA20", ma50: "MA50", bb_upper: "BB上限", bb_middle: "BB中央", bb_lower: "BB下限" };
                const num = typeof value === "number" ? value : 0;
                const key = typeof name === "string" ? name : "";
                return [detail.currency === "JPY" ? `¥${num.toFixed(0)}` : `$${num.toFixed(2)}`, labels[key] ?? key] as [string, string];
              }}
            />
            <Legend formatter={(v: string) => ({ close: "終値", ma20: "EMA20", ma50: "EMA50", bb_upper: "BB上限", bb_lower: "BB下限" } as Record<string, string>)[v] ?? v} />
            <Area dataKey="bb_upper" fill="#1e3a5f" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 2" dot={false} name="bb_upper" fillOpacity={0.15} />
            <Area dataKey="bb_lower" fill="#1e3a5f" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 2" dot={false} name="bb_lower" fillOpacity={0} />
            <Line type="monotone" dataKey="close" stroke="#f59e0b" strokeWidth={2} dot={false} name="close" />
            <Line type="monotone" dataKey="ma20" stroke="#60a5fa" strokeWidth={1.5} dot={false} name="ma20" strokeDasharray="none" />
            <Line type="monotone" dataKey="ma50" stroke="#f97316" strokeWidth={1.5} dot={false} name="ma50" strokeDasharray="6 2" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* RSI */}
      <div className="bg-gray-800 rounded-xl p-4">
        <h3 className="text-sm text-gray-400 mb-3">RSI (14)</h3>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={priceData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tickFormatter={tickFormatter} tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fill: "#9ca3af", fontSize: 11 }} width={35} />
            <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: "8px" }} labelStyle={{ color: "#e5e7eb" }} formatter={(v: unknown) => [(typeof v === "number" ? v.toFixed(2) : ""), "RSI"] as [string, string]} />
            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
            <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="rsi" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* MACD */}
      <div className="bg-gray-800 rounded-xl p-4">
        <h3 className="text-sm text-gray-400 mb-3">MACD (12, 26, 9)</h3>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={macdData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tickFormatter={tickFormatter} tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} width={50} />
            <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: "8px" }} labelStyle={{ color: "#e5e7eb" }} formatter={(v: unknown, name: unknown) => [(typeof v === "number" ? v.toFixed(4) : ""), name === "macd" ? "MACD" : name === "signal" ? "シグナル" : "ヒストグラム"] as [string, string]} />
            <ReferenceLine y={0} stroke="#6b7280" />
            <Bar dataKey="histogram" fill="#6366f1" opacity={0.7} name="histogram" />
            <Line type="monotone" dataKey="macd" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="macd" />
            <Line type="monotone" dataKey="signal" stroke="#ef4444" strokeWidth={1.5} dot={false} name="signal" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
