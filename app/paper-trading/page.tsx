import { readFileSync, existsSync } from "fs";
import path from "path";

export const dynamic = "force-dynamic";

// ── Types ────────────────────────────────────────────────────────────────────

type Algorithm = {
  id: number;
  name: string;
  sell_rule: string;
  threshold: number;
  target_pct: number | null;
  stop_pct: number | null;
  trail_pct: number | null;
  max_hold_days: number;
};

type OpenPosition = {
  entry_price: number;
  shares: number;
  entry_date: string;
  score: number;
  target_pct: number | null;
  stop_pct: number | null;
  trail_pct: number | null;
  max_price: number;
};

type AlgoState = {
  capital_remaining: number;
  open: Record<string, OpenPosition>;
  closed_pnl: number;
  total_trades: number;
  winning_trades: number;
};

type TradeLog = {
  algo_id: number;
  algo_name: string;
  ticker: string;
  action: "BUY" | "SELL";
  price: number;
  shares: number;
  date: string;
  score: number | null;
  pnl: number | null;
  return_pct: number | null;
  reason?: string;
};

// ── Data loading ─────────────────────────────────────────────────────────────

function loadData() {
  const tradingDir = path.join(process.cwd(), "trading");
  const posPath = path.join(tradingDir, "positions.json");
  const logPath = path.join(tradingDir, "trade_log.json");
  const algoPath = path.join(tradingDir, "algorithms.json");

  if (!existsSync(posPath) || !existsSync(algoPath)) return null;

  const positions: Record<string, AlgoState> = JSON.parse(readFileSync(posPath, "utf-8"));
  const logs: TradeLog[] = existsSync(logPath) ? JSON.parse(readFileSync(logPath, "utf-8")) : [];
  const algorithms: Algorithm[] = JSON.parse(readFileSync(algoPath, "utf-8"));
  return { positions, logs, algorithms };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatCurrency(v: number) {
  return v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

function pctClass(v: number) {
  if (v > 0) return "text-green-400";
  if (v < 0) return "text-red-400";
  return "text-gray-400";
}

function daysSince(dateStr: string) {
  const d = new Date(dateStr);
  const now = new Date();
  return Math.floor((now.getTime() - d.getTime()) / 86400000);
}

// ── Components ────────────────────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-base font-bold text-white">{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function AlgoCard({ algo, state }: { algo: Algorithm; state: AlgoState }) {
  const openValue = Object.values(state.open).reduce(
    (s, p) => s + p.entry_price * p.shares, 0
  );
  const equity = state.capital_remaining + openValue;
  const pnl = equity - 100000;
  const pnlPct = (pnl / 100000) * 100;
  const winRate = state.total_trades > 0
    ? (state.winning_trades / state.total_trades) * 100
    : null;

  const sellLabel = algo.target_pct
    ? `+${(algo.target_pct * 100).toFixed(0)}% / -${(algo.stop_pct! * 100).toFixed(0)}%`
    : algo.trail_pct
    ? `トレール ${(algo.trail_pct * 100).toFixed(0)}%`
    : algo.sell_rule;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <span className="text-xs text-gray-500 mr-2">#{algo.id}</span>
          <span className="text-sm font-bold text-white">{algo.name}</span>
        </div>
        <span className="text-xs text-gray-600 bg-gray-800 px-2 py-0.5 rounded">{sellLabel}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-gray-500">総資産</span>
          <div className="text-white font-mono">{formatCurrency(equity)}</div>
        </div>
        <div>
          <span className="text-gray-500">損益</span>
          <div className={`font-mono font-bold ${pctClass(pnl)}`}>
            {pnl >= 0 ? "+" : ""}{formatCurrency(pnl)}
            <span className="text-xs ml-1">({pnlPct >= 0 ? "+" : ""}{pnlPct.toFixed(1)}%)</span>
          </div>
        </div>
        <div>
          <span className="text-gray-500">オープン</span>
          <div className="text-white">{Object.keys(state.open).length}ポジション</div>
        </div>
        <div>
          <span className="text-gray-500">勝率</span>
          <div className={winRate !== null ? pctClass(winRate - 50) : "text-gray-500"}>
            {winRate !== null ? `${winRate.toFixed(0)}% (${state.total_trades}件)` : "—"}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function PaperTradingPage() {
  const data = loadData();

  if (!data) {
    return (
      <div className="py-12 text-center text-gray-500">
        <div className="text-2xl mb-2">⏳</div>
        <div>ペーパートレードデータがまだありません。</div>
        <div className="text-xs mt-2">GitHub Actions が初回実行されると表示されます。</div>
      </div>
    );
  }

  const { positions, logs, algorithms } = data;

  // Overall stats
  const totalEquity = algorithms.reduce((sum, algo) => {
    const state = positions[String(algo.id)];
    const openVal = Object.values(state.open).reduce((s, p) => s + p.entry_price * p.shares, 0);
    return sum + state.capital_remaining + openVal;
  }, 0);
  const totalPnl = totalEquity - 100000 * algorithms.length;
  const totalPnlPct = (totalPnl / (100000 * algorithms.length)) * 100;
  const totalTrades = algorithms.reduce((s, a) => s + positions[String(a.id)].total_trades, 0);
  const totalWins = algorithms.reduce((s, a) => s + positions[String(a.id)].winning_trades, 0);
  const overallWinRate = totalTrades > 0 ? (totalWins / totalTrades) * 100 : null;

  // All open positions across all algos
  const allOpen: Array<{ algoId: number; algoName: string; ticker: string; pos: OpenPosition }> = [];
  for (const algo of algorithms) {
    for (const [ticker, pos] of Object.entries(positions[String(algo.id)].open)) {
      allOpen.push({ algoId: algo.id, algoName: algo.name, ticker, pos });
    }
  }

  // Recent trades (last 50)
  const recentLogs = [...logs].reverse().slice(0, 50);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">ペーパートレード</h1>
        <p className="text-xs text-gray-500 mt-1">
          10アルゴリズム並行仮想売買 — 各 $100,000 初期資金 — 毎営業日 UTC 23:00 自動更新
        </p>
      </div>

      {/* Overall summary */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard
          label="合計資産 (10アルゴ)"
          value={formatCurrency(totalEquity)}
          sub={`初期: ${formatCurrency(100000 * algorithms.length)}`}
        />
        <StatCard
          label="合計損益"
          value={`${totalPnl >= 0 ? "+" : ""}${formatCurrency(totalPnl)}`}
          sub={`${totalPnlPct >= 0 ? "+" : ""}${totalPnlPct.toFixed(1)}%`}
        />
        <StatCard
          label="全取引数 / 勝率"
          value={totalTrades > 0 ? `${totalTrades}件` : "—"}
          sub={overallWinRate !== null ? `勝率 ${overallWinRate.toFixed(0)}%` : "取引なし"}
        />
        <StatCard
          label="オープンポジション"
          value={`${allOpen.length}件`}
          sub={`${algorithms.length}アルゴ稼働中`}
        />
      </div>

      {/* Algorithm cards */}
      <div>
        <h2 className="text-sm font-semibold text-gray-300 mb-3">アルゴリズム別サマリー</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {algorithms.map((algo) => (
            <AlgoCard key={algo.id} algo={algo} state={positions[String(algo.id)]} />
          ))}
        </div>
      </div>

      {/* Open positions */}
      {allOpen.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-gray-300 mb-3">
            現在のオープンポジション ({allOpen.length}件)
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left py-2 pr-3">銘柄</th>
                  <th className="text-left py-2 pr-3">アルゴ</th>
                  <th className="text-right py-2 pr-3">エントリー</th>
                  <th className="text-right py-2 pr-3">株数</th>
                  <th className="text-right py-2 pr-3">保有日数</th>
                  <th className="text-right py-2 pr-3">含損益 (簡易)</th>
                  <th className="text-right py-2">スコア</th>
                </tr>
              </thead>
              <tbody>
                {allOpen.map(({ algoId, algoName, ticker, pos }) => {
                  const days = daysSince(pos.entry_date);
                  return (
                    <tr key={`${algoId}-${ticker}`} className="border-b border-gray-800/50 hover:bg-gray-900">
                      <td className="py-2 pr-3">
                        <a
                          href={`https://finance.yahoo.com/quote/${ticker}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-400 hover:underline font-mono"
                        >
                          {ticker}
                        </a>
                      </td>
                      <td className="py-2 pr-3 text-gray-400">#{algoId} {algoName}</td>
                      <td className="py-2 pr-3 text-right font-mono">${pos.entry_price.toFixed(2)}</td>
                      <td className="py-2 pr-3 text-right font-mono">{pos.shares}</td>
                      <td className="py-2 pr-3 text-right text-gray-400">{days}日</td>
                      <td className="py-2 pr-3 text-right text-gray-500">—</td>
                      <td className="py-2 text-right text-yellow-400">{pos.score.toFixed(1)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Trade history */}
      <div>
        <h2 className="text-sm font-semibold text-gray-300 mb-3">
          取引履歴 (直近{recentLogs.length}件)
        </h2>
        {recentLogs.length === 0 ? (
          <div className="text-gray-600 text-xs py-4 text-center">
            取引履歴がありません。GitHub Actions の初回実行をお待ちください。
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left py-2 pr-3">日付</th>
                  <th className="text-left py-2 pr-3">銘柄</th>
                  <th className="text-left py-2 pr-3">売買</th>
                  <th className="text-left py-2 pr-3">アルゴ</th>
                  <th className="text-right py-2 pr-3">価格</th>
                  <th className="text-right py-2 pr-3">損益</th>
                  <th className="text-right py-2">リターン</th>
                </tr>
              </thead>
              <tbody>
                {recentLogs.map((log, i) => (
                  <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-900">
                    <td className="py-2 pr-3 text-gray-500 font-mono">{log.date}</td>
                    <td className="py-2 pr-3">
                      <a
                        href={`https://finance.yahoo.com/quote/${log.ticker}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:underline font-mono"
                      >
                        {log.ticker}
                      </a>
                    </td>
                    <td className="py-2 pr-3">
                      <span className={log.action === "BUY" ? "text-green-400" : "text-red-400"}>
                        {log.action === "BUY" ? "買" : "売"}
                      </span>
                    </td>
                    <td className="py-2 pr-3 text-gray-500">#{log.algo_id} {log.algo_name}</td>
                    <td className="py-2 pr-3 text-right font-mono">${log.price.toFixed(2)}</td>
                    <td className={`py-2 pr-3 text-right font-mono ${log.pnl !== null ? pctClass(log.pnl) : "text-gray-600"}`}>
                      {log.pnl !== null ? `${log.pnl >= 0 ? "+" : ""}$${Math.abs(log.pnl).toFixed(0)}` : "—"}
                    </td>
                    <td className={`py-2 text-right font-mono ${log.return_pct !== null ? pctClass(log.return_pct) : "text-gray-600"}`}>
                      {log.return_pct !== null ? `${log.return_pct >= 0 ? "+" : ""}${log.return_pct.toFixed(1)}%` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Footer note */}
      <div className="text-xs text-gray-600 border-t border-gray-800 pt-4">
        ※ 純粋シミュレーション — 実際の注文は行いません。価格は yfinance 終値。手数料 0.16%。
        初期資金 $100,000 × 10アルゴ = $1,000,000 仮想資金。
      </div>
    </div>
  );
}
