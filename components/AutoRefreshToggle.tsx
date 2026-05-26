"use client";
import { useState, useEffect } from "react";

interface Props {
  interval: number;
  onIntervalChange: (ms: number) => void;
  onRefresh: () => void;
}

const OPTIONS = [
  { label: "1分", ms: 60_000 },
  { label: "3分", ms: 180_000 },
  { label: "5分", ms: 300_000 },
];

export default function AutoRefreshToggle({ interval, onIntervalChange, onRefresh }: Props) {
  const [secondsLeft, setSecondsLeft] = useState(interval / 1000);

  useEffect(() => {
    setSecondsLeft(interval / 1000);
    const timer = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) { return interval / 1000; }
        return s - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [interval]);

  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="text-gray-400">更新間隔:</span>
      <div className="flex gap-1">
        {OPTIONS.map((opt) => (
          <button
            key={opt.ms}
            onClick={() => onIntervalChange(opt.ms)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              interval === opt.ms ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2 text-gray-400 text-xs">
        <span>次回更新: {secondsLeft}秒後</span>
        <button
          onClick={onRefresh}
          className="bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1 rounded text-xs transition-colors"
        >
          今すぐ更新
        </button>
      </div>
    </div>
  );
}
