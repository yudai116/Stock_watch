"use client";
import { useState } from "react";

interface Props {
  onAdd: (ticker: string) => Promise<unknown>;
}

export default function AddTickerForm({ onAdd }: Props) {
  const [value, setValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const ticker = value.trim().toUpperCase();
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      await onAdd(ticker);
      setValue("");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "追加に失敗しました");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="ティッカー例: AAPL, 7203.T"
        className="bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 flex-1 text-sm"
        disabled={loading}
      />
      <button
        type="submit"
        disabled={loading || !value.trim()}
        className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
      >
        {loading ? "確認中..." : "+ 追加"}
      </button>
      {error && <p className="absolute mt-11 text-red-400 text-xs">{error}</p>}
    </form>
  );
}
