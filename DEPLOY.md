# デプロイ手順 — Vercelで完全無料

## 構成（サーバー1つだけ）
- **Vercel のみ** — フロントエンド + バックエンドAPI の両方（完全無料）
- Python サーバー不要。Next.js の API Routes が Yahoo Finance から直接データ取得

---

## ステップ1: GitHubにリポジトリを公開する

1. [GitHub](https://github.com) でアカウントを作成（未登録の場合）
2. 右上「+」→「New repository」→ 名前: `stock-watch`、Public で作成
3. ローカルPCのターミナルで実行:
   ```bash
   cd Stock_watch
   git remote set-url origin https://github.com/あなたのユーザー名/stock-watch.git
   git push -u origin main
   ```
   ※ ブランチ名が `claude/stock-price-dashboard-QrhiS` の場合は `main` に変更してプッシュ

---

## ステップ2: Vercelにデプロイ

1. [vercel.com](https://vercel.com) にアクセス → 「Sign Up」→ GitHub でサインアップ
2. 「Add New...」→「Project」→ `stock-watch` リポジトリを選択
3. 設定画面で以下を変更:
   - **Root Directory**: `frontend` と入力
   - それ以外はデフォルトのまま
4. 「Deploy」をクリック
5. 2〜3分後にデプロイ完了 → URLが発行される（例: `https://stock-watch.vercel.app`）

---

## ステップ3: スマホでアクセス

VercelのURL をスマホのブラウザで開くだけ！

**ホーム画面に追加する方法:**
- iPhone: Safari で開く → 共有ボタン → 「ホーム画面に追加」
- Android: Chrome で開く → メニュー → 「ホーム画面に追加」

---

## ウォッチリストのカスタマイズ

デフォルト銘柄を変更したい場合:
1. Vercel の「Settings」→「Environment Variables」
2. 以下を追加:

   | 変数名 | 値（カンマ区切りで銘柄コード）|
   |--------|-------------------------------|
   | `WATCHLIST` | `7203.T,6758.T,AAPL,MSFT,NVDA` |

3. 「Redeploy」で反映

---

## 費用

| サービス | 費用 |
|--------|------|
| Vercel | **完全無料**（個人利用） |
| GitHub | **完全無料** |
| Yahoo Finance API | **完全無料**（利用制限なし） |

**合計: 0円**
