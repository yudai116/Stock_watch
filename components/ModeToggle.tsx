"use client";

export type TradeMode = "swing" | "day";

interface Props {
  mode: TradeMode;
  onChange: (m: TradeMode) => void;
}

export default function ModeToggle({ mode, onChange }: Props) {
  return (
    <div className="inline-flex bg-gray-800 rounded-lg p-0.5 gap-0.5">
      <button
        onClick={() => onChange("swing")}
        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
          mode === "swing"
            ? "bg-blue-600 text-white shadow"
            : "text-gray-400 hover:text-white"
        }`}
      >
        スイング
      </button>
      <button
        onClick={() => onChange("day")}
        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
          mode === "day"
            ? "bg-orange-500 text-white shadow"
            : "text-gray-400 hover:text-white"
        }`}
      >
        デイトレ
      </button>
    </div>
  );
}
