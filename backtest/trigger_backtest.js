// ============================================================
// Scriptable スクリプト — バックテスト起動 (デイ/スイング選択)
// ============================================================
// 【設定方法】
// 1. GitHub で Personal Access Token を発行
//    https://github.com/settings/tokens
//    → "Generate new token (classic)"
//    → スコープ: repo (または workflow)
//    → 生成されたトークンをコピー
//
// 2. 下の GITHUB_TOKEN に貼り付ける
//
// 3. Scriptable にこのスクリプトを追加して実行
// ============================================================

const GITHUB_TOKEN = "ghp_ここにトークンを貼り付ける";   // ← 要変更
const REPO         = "yudai116/stock_watch";
const WORKFLOW     = "backtest.yml";
const BRANCH       = "claude/stock-price-dashboard-QrhiS";

// ── トークン未設定チェック ───────────────────────────────────
if (GITHUB_TOKEN.includes("ここに")) {
  const a = new Alert();
  a.title = "設定が必要です";
  a.message = "スクリプト内の GITHUB_TOKEN に\nGitHub Personal Access Token を貼り付けてください。\n\nhttps://github.com/settings/tokens";
  a.addAction("OK");
  await a.present();
  Script.complete();
  return;
}

// ── モード選択ダイアログ ──────────────────────────────────────
const modeAlert = new Alert();
modeAlert.title = "バックテストモード";
modeAlert.message = "実行するモードを選んでください\n\n• スイング: 数日〜数週間保有の戦略最適化\n• デイ: 当日内 2〜8h 保有の戦略最適化\n• 両方: 約20〜40分かかります";
modeAlert.addAction("スイング (推奨)");
modeAlert.addAction("デイトレード");
modeAlert.addAction("両方");
modeAlert.addCancelAction("キャンセル");

const modeIdx = await modeAlert.present();
if (modeIdx === -1) { Script.complete(); return; }

const modeMap = { 0: "swing", 1: "day", 2: "both" };
const mode = modeMap[modeIdx];
const modeJa = { swing: "スイング", day: "デイ", both: "スイング + デイ" }[mode];

// ── ワークフロー起動 API リクエスト ────────────────────────────
const url = `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`;

const req = new Request(url);
req.method  = "POST";
req.headers = {
  "Authorization": `token ${GITHUB_TOKEN}`,
  "Accept":        "application/vnd.github.v3+json",
  "Content-Type":  "application/json",
};
req.body = JSON.stringify({
  ref: BRANCH,
  inputs: { mode: mode },
});

try {
  await req.load();
  const status = req.response.statusCode;

  if (status === 204) {
    const a = new Alert();
    a.title   = `${modeJa} バックテスト開始 ✓`;
    a.message = `GitHub Actions で「${modeJa}」バックテストが始まりました。\n\n` +
                `実験計画法 (DOE) → 遺伝的アルゴリズム → Walk-Forward 検証\n\n` +
                `完了まで約 ${mode === "both" ? "20〜40" : "10〜20"}分。\n` +
                `結果は自動でサイトに反映されます。`;
    a.addAction("OK");
    await a.present();

  } else if (status === 401) {
    throw new Error("トークンが無効です。\nGitHub でトークンを再発行してください。");
  } else if (status === 404) {
    throw new Error("リポジトリまたはワークフローが見つかりません。");
  } else {
    throw new Error(`HTTPエラー: ${status}`);
  }

} catch (e) {
  const a = new Alert();
  a.title   = "エラー";
  a.message = e.message;
  a.addAction("OK");
  await a.present();
}

Script.complete();
