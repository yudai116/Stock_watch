// ============================================================
// Scriptable スクリプト — バックテスト起動
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

// ── ワークフロー起動 API リクエスト ────────────────────────────
const url = `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW}/dispatches`;

const req = new Request(url);
req.method  = "POST";
req.headers = {
  "Authorization": `token ${GITHUB_TOKEN}`,
  "Accept":        "application/vnd.github.v3+json",
  "Content-Type":  "application/json",
};
req.body = JSON.stringify({ ref: BRANCH });

try {
  await req.load();
  const status = req.response.statusCode;

  if (status === 204) {
    // 起動成功
    const a = new Alert();
    a.title   = "バックテスト開始 ✓";
    a.message = "GitHub Actions でバックテストが始まりました。\n\n約10〜20分後に完了し、\n結果が自動でサイトに反映されます。\n\n進捗は GitHub の Actions タブで確認できます。";
    a.addAction("OK");
    await a.present();

  } else if (status === 401) {
    throw new Error("トークンが無効です。\nGitHub でトークンを再発行してください。");

  } else if (status === 404) {
    throw new Error("リポジトリまたはワークフローが見つかりません。\nREPO / WORKFLOW の設定を確認してください。");

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
