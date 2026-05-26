# デプロイ手順 — スマホからいつでもアクセスできるようにする

## 構成
- **バックエンド**: Railway（無料枠 $5/月）
- **フロントエンド**: Vercel（完全無料）

---

## ステップ1: GitHubにリポジトリを公開する

1. [GitHub](https://github.com) でアカウントを作成（未登録の場合）
2. 新しいリポジトリを作成（例: `stock-watch`）
3. ローカルでコマンドを実行:
   ```bash
   git remote set-url origin https://github.com/あなたのユーザー名/stock-watch.git
   git push -u origin main
   ```

---

## ステップ2: バックエンドをRailwayにデプロイ

1. [railway.app](https://railway.app) にアクセス → GitHubアカウントでサインアップ
2. 「New Project」→「Deploy from GitHub repo」→ リポジトリを選択
3. **Root Directory** に `backend` と入力
4. デプロイ後、「Settings」→「Environment Variables」で以下を設定:

   | 変数名 | 値 |
   |--------|----|
   | `CORS_ALLOW_ALL` | `true` |
   | `WATCHLIST_FILE` | `data/watchlist.json` |

5. 「Settings」→「Domains」→「Generate Domain」でURLを生成
   - 例: `https://stock-watch-backend.up.railway.app`
   - **このURLをメモする**

---

## ステップ3: フロントエンドをVercelにデプロイ

1. [vercel.com](https://vercel.com) にアクセス → GitHubアカウントでサインアップ
2. 「Add New Project」→ リポジトリを選択
3. **Root Directory** に `frontend` と入力
4. 「Environment Variables」に以下を追加:

   | 変数名 | 値 |
   |--------|----|
   | `NEXT_PUBLIC_API_URL` | ステップ2でメモしたRailwayのURL |

5. 「Deploy」をクリック
6. デプロイ完了後、VercelがURLを発行 (例: `https://stock-watch.vercel.app`)

---

## ステップ4: スマホでアクセス

VercelのURL（例: `https://stock-watch.vercel.app`）をスマホのブラウザで開くだけ。
ホーム画面に追加すれば、アプリのように使えます。

---

## 注意事項

- Railwayの無料枠は月$5のクレジット。通常の使い方では数ヶ月は無料で使えます
- Vercelは個人利用は完全無料
- ウォッチリスト（銘柄一覧）はRailwayサーバー上に保存されます
