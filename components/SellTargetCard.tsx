import type { SellSignals } from "@/types";
import { formatPrice } from "@/lib/formatters";

interface Props {
  sellSignals: SellSignals;
  currency: string;
  currentPrice: number;
}

function Row({ label, value, sub, highlight }: { label: string; value: string; sub?: string; highlight?: "green" | "yellow" | "red" }) {
  const color = highlight === "green" ? "text-emerald-400" : highlight === "yellow" ? "text-yellow-400" : highlight === "red" ? "text-red-400" : "text-white";
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-800 last:border-0">
      <span className="text-gray-400 text-sm">{label}</span>
      <div className="text-right">
        <span className={`font-mono font-semibold ${color}`}>{value}</span>
        {sub && <p className="text-gray-500 text-xs">{sub}</p>}
      </div>
    </div>
  );
}

export default function SellTargetCard({ sellSignals, currency, currentPrice }: Props) {
  const { bb_target_2sigma, bb_target_2_5sigma, atr_14d, stop_loss_5pct, dist_from_52w_high_pct, high_52w } = sellSignals;

  const upside2s = bb_target_2sigma && currentPrice > 0
    ? Math.round(((bb_target_2sigma - currentPrice) / currentPrice) * 1000) / 10
    : null;
  const upside25s = bb_target_2_5sigma && currentPrice > 0
    ? Math.round(((bb_target_2_5sigma - currentPrice) / currentPrice) * 1000) / 10
    : null;

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-white font-semibold text-sm">売りターゲット・リスク管理</span>
        <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">バックテスト: BB 2σ + 5%ストップ → シャープ 1.153</span>
      </div>
      <div className="space-y-0">
        <Row
          label="売りターゲット① (BB 2σ)"
          value={bb_target_2sigma ? formatPrice(bb_target_2sigma, currency) : "N/A"}
          sub={upside2s !== null ? `現在値から +${upside2s}%` : undefined}
          highlight="green"
        />
        <Row
          label="売りターゲット② (BB 2.5σ)"
          value={bb_target_2_5sigma ? formatPrice(bb_target_2_5sigma, currency) : "N/A"}
          sub={upside25s !== null ? `現在値から +${upside25s}%` : undefined}
          highlight="green"
        />
        <Row
          label="ストップロス (−5%)"
          value={stop_loss_5pct ? formatPrice(stop_loss_5pct, currency) : "N/A"}
          sub="バックテスト推奨ストップ水準"
          highlight="red"
        />
        <Row
          label="ATR 14日 (ボラティリティ)"
          value={atr_14d ? formatPrice(atr_14d, currency) : "N/A"}
          sub="ポジションサイジングの参考"
          highlight="yellow"
        />
        <Row
          label="52週高値"
          value={high_52w ? formatPrice(high_52w, currency) : "N/A"}
          sub={dist_from_52w_high_pct !== null ? `現在値は高値から −${dist_from_52w_high_pct}%` : undefined}
        />
      </div>
      <p className="text-gray-600 text-xs mt-3">
        ① BB上限2σ = 主要利確目標 / ② BB上限2.5σ = 強トレンド時の延長目標 / ATRはポジションサイズ計算に使用
      </p>
    </div>
  );
}
