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
  multiplier: number;
  currency: "USD" | "JPY";
  target_pct: number | null;
  stop_pct: number | null;
  trail_pct: number | null;
  max_price: number;
};

type AlgoState = {
  capital_usd: number;
  capital_jpy: number;
  initial_capital_usd: number;
  initial_capital_jpy: number;
  open: Record<string, OpenPosition>;
  closed_pnl_usd: number;
  closed_pnl_jpy: number;
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
  multiplier: number | null;
  currency: "USD" | "JPY";
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

const USDJPY_APPROX = 150.0; // 表示用の固定レート (概算)

function fmtUSD(v: number) {
  return v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}
function fmtJPY(v: number) {
  return `¥${Math.round(v).toLocaleString("ja-JP")}`;
}
function fmtPct(v: number) {
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
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

function StatCard({ label, value, sub, valueClass }: {
  label: string; value: string; sub?: string; valueClass?: string
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-base font-bold ${valueClass ?? "text-white"}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function AlgoCard({ algo, state }: { algo: Algorithm; state: AlgoState }) {
  const usdOpenVal = Object.values(state.open)
    .filter(p => p.currency === "USD")
    .reduce((s, p) => s + p.entry_price * p.shares, 0);
  const jpyOpenVal = Object.values(state.open)
    .filter(p => p.currency !== "USD")
    .reduce((s, p) => s + p.entry_price * p.shares, 0);

  const usdEquity = state.capital_usd + usdOpenVal;
  const jpyEquity = state.capital_jpy + jpyOpenVal;
  const usdPnl = usdEquity - state.initial_capital_usd;
  const jpyPnl = jpyEquity - state.initial_capital_jpy;
  const usdPnlPct = (usdPnl / state.initial_capital_usd) * 100;
  const jpyPnlPct = (jpyPnl / state.initial_capital_jpy) * 100;

  const winRate = state.total_trades > 0
    ? (state.winning_trades / state.total_trades) * 100
    : null;

  const sellLabel = algo.target_pct
    ? `+${(algo.target_pct * 100).toFixed(0)}% / -${((algo.stop_pct ?? 0) * 100).toFixed(0)}%`
    : algo.trail_pct
    ? `トレール ${(algo.trail_pct * 100).toFixed(0)}%`
    : algo.sell_rule;

  const jpOpen = Object.values(state.open).filter(p => p.currency !== "USD").length;
  const usOpen = Object.values(state.open).filter(p => p.currency === "USD").length;

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
          <span className="text-gray-500">USD損益</span>
          <div className={`font-mono font-bold ${pctClass(usdPnl)}`}>
            {usdPnl >= 0 ? "+" : ""}{fmtUSD(usdPnl)}
            <span className="text-xs font-normal ml-1">({fmtPct(usdPnlPct)})</span>
          </div>
        </div>
        <div>
          <span className="text-gray-500">JPY損益</span>
          <div className={`font-mono font-bold ${pctClass(jpyPnl)}`}>
            {jpyPnl >= 0 ? "+" : ""}{fmtJPY(jpyPnl)}
            <span className="text-xs font-normal ml-1">({fmtPct(jpyPnlPct)})</span>
          </div>
        </div>
        <div>
          <span className="text-gray-500">ポジション</span>
          <div className="text-white">
            US:{usOpen} JP:{jpOpen}
            <span className="text-gray-500 ml-1">/ 5</span>
          </div>
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

  // Overall USD totals
  const totalUsdEquity = algorithms.reduce((sum, algo) => {
    const st = positions[String(algo.id)];
    const openVal = Object.values(st.open)
      .filter(p => p.currency === "USD")
      .reduce((s, p) => s + p.entry_price * p.shares, 0);
    return sum + st.capital_usd + openVal;
  }, 0);
  const totalUsdInitial = algorithms.reduce(
    (s, a) => s + positions[String(a.id)].initial_capital_usd, 0
  );
  const totalUsdPnl = totalUsdEquity - totalUsdInitial;

  // Overall JPY totals
  const totalJpyEquity = algorithms.reduce((sum, algo) => {
    const st = positions[String(algo.id)];
    const openVal = Object.values(st.open)
      .filter(p => p.currency !== "USD")
      .reduce((s, p) => s + p.entry_price * p.shares, 0);
    return sum + st.capital_jpy + openVal;
  }, 0);
  const totalJpyInitial = algorithms.reduce(
    (s, a) => s + positions[String(a.id)].initial_capital_jpy, 0
  );
  const totalJpyPnl = totalJpyEquity - totalJpyInitial;

  const totalTrades = algorithms.reduce((s, a) => s + positions[String(a.id)].total_trades, 0);
  const totalWins = algorithms.reduce((s, a) => s + positions[String(a.id)].winning_trades, 0);
  const overallWinRate = totalTrades > 0 ? (totalWins / totalTrades) * 100 : null;

  const allOpen: Array<{ algoId: number; algoName: string; ticker: string; pos: OpenPosition }> = [];
  for (const algo of algorithms) {
    for (const [ticker, pos] of Object.entries(positions[String(algo.id)].open)) {
      allOpen.push({ algoId: algo.id, algoName: algo.name, ticker, pos });
    }
  }

  const recentLogs = [...logs].reverse().slice(0, 60);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">ペーパートレード</h1>
        <p className="text-xs text-gray-500 mt-1">
          10アルゴリズム並行仮想売買 — US37銘柄 + JP10銘柄 — 毎営業日 UTC 23:00 自動更新
        </p>
      </div>

      {/* Overall summary */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard
          label="USD合計損益 (10アルゴ)"
          value={`${totalUsdPnl >= 0 ? "+" : ""}${fmtUSD(totalUsdPnl)}`}
          sub={`資産: ${fmtUSD(totalUsdEquity)} / 初期: ${fmtUSD(totalUsdInitial)}`}
          valueClass={pctClass(totalUsdPnl)}
        />
        <StatCard
          label="JPY合計損益 (10アルゴ)"
          value={`${totalJpyPnl >= 0 ? "+" : ""}${fmtJPY(totalJpyPnl)}`}
          sub={`資産: ${fmtJPY(totalJpyEquity)} / 初期: ${fmtJPY(totalJpyInitial)}`}
          valueClass={pctClass(totalJpyPnl)}
        />
        <StatCard
          label="全取引数 / 勝率"
          value={totalTrades > 0 ? `${totalTrades}件` : "—"}
          sub={overallWinRate !== null ? `勝率 ${overallWinRate.toFixed(0)}%` : "取引なし"}
        />
        <StatCard
          label="オープンポジション"
          value={`${allOpen.length}件`}
          sub={`US${allOpen.filter(p => p.pos.currency === "USD").length} / JP${allOpen.filter(p => p.pos.currency !== "USD").length}`}
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
                  <th className="text-right py-2 pr-3">評価額</th>
                  <th className="text-right py-2 pr-3">保有日</th>
                  <th className="text-right py-2 pr-3">スコア</th>
                  <th className="text-right py-2">倍率</th>
                </tr>
              </thead>
              <tbody>
                {allOpen.map(({ algoId, algoName, ticker, pos }) => {
                  const days = daysSince(pos.entry_date);
                  const isJp = pos.currency === "JPY";
                  const posVal = pos.entry_price * pos.shares;
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
                        <span className={`ml-1 text-xs ${isJp ? "text-orange-400" : "text-gray-600"}`}>
                          {isJp ? "JP" : "US"}
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-gray-400">#{algoId}</td>
                      <td className="py-2 pr-3 text-right font-mono">
                        {isJp ? fmtJPY(pos.entry_price) : `$${pos.entry_price.toFixed(2)}`}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono">{pos.shares}</td>
                      <td className="py-2 pr-3 text-right font-mono text-gray-400">
                        {isJp ? fmtJPY(posVal) : fmtUSD(posVal)}
                      </td>
                      <td className="py-2 pr-3 text-right text-gray-400">{days}日</td>
                      <td className="py-2 pr-3 text-right text-yellow-400">{pos.score.toFixed(1)}</td>
                      <td className="py-2 text-right text-blue-400">{pos.multiplier?.toFixed(1)}x</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Position sizing guide */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-3 text-xs text-gray-500">
        <span className="text-gray-400 font-semibold">ポジションサイズ計算式: </span>
        閾値スコア → 0.5x（最小）、スコア100 → 2.0x（最大）の線形スケール。
        US基本 $2,000 × 倍率 / JP基本 ¥300,000 × 倍率。最大5ポジション（うちJP最大2）。
      </div>

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
                  <th className="text-right py-2 pr-3">倍率</th>
                  <th className="text-right py-2 pr-3">損益</th>
                  <th className="text-right py-2">リターン</th>
                </tr>
              </thead>
              <tbody>
                {recentLogs.map((log, i) => {
                  const isJp = log.currency === "JPY";
                  return (
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
                        <span className={`ml-1 ${isJp ? "text-orange-400" : "text-gray-600"}`}>
                          {isJp ? "JP" : "US"}
                        </span>
                      </td>
                      <td className="py-2 pr-3">
                        <span className={log.action === "BUY" ? "text-green-400" : "text-red-400"}>
                          {log.action === "BUY" ? "買" : "売"}
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-gray-500">#{log.algo_id}</td>
                      <td className="py-2 pr-3 text-right font-mono">
                        {isJp ? fmtJPY(log.price) : `$${log.price.toFixed(2)}`}
                      </td>
                      <td className="py-2 pr-3 text-right text-blue-400">
                        {log.multiplier != null ? `${log.multiplier.toFixed(1)}x` : "—"}
                      </td>
                      <td className={`py-2 pr-3 text-right font-mono ${log.pnl !== null ? pctClass(log.pnl) : "text-gray-600"}`}>
                        {log.pnl !== null
                          ? isJp
                            ? `${log.pnl >= 0 ? "+" : ""}${fmtJPY(log.pnl)}`
                            : `${log.pnl >= 0 ? "+" : ""}${fmtUSD(log.pnl)}`
                          : "—"}
                      </td>
                      <td className={`py-2 text-right font-mono ${log.return_pct !== null ? pctClass(log.return_pct) : "text-gray-600"}`}>
                        {log.return_pct !== null ? fmtPct(log.return_pct) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="text-xs text-gray-600 border-t border-gray-800 pt-4 space-y-1">
        <div>※ 純粋シミュレーション — 実際の注文は行いません。価格は yfinance 終値。</div>
        <div>※ 手数料: US 0.16% / JP 0.30%（往復）。初期資金: $20,000 + ¥3,000,000 × 10アルゴ。</div>
        <div>※ USD/JPY合算表示は参考値（¥{USDJPY_APPROX}固定換算）。</div>
      </div>
    </div>
  );
}
