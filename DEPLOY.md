# デプロイ手順 — Vercelで完全無料

## 構成（サーバー1つだけ）
- **Vercel のみ** — フロントエンド + バックエンドAPI の両方（完全無料）
- Python サーバー不要。Next.js の API Routes が Yahoo Finance から直接データ取得

---

## 前提
コードはすでに GitHub の `yudai116/Stock_watch` リポジトリに入っています。

---

## ステップ1: Vercelにデプロイ

1. [vercel.com](https://vercel.com) にアクセス → 「Sign Up」→ **GitHub でサインアップ**
2. 「Add New...」→「Project」をクリック
3. リポジトリ一覧から **`yudai116/Stock_watch`** を選択
4. 設定画面はデフォルトのままで「**Deploy**」をクリック
   - Root Directory の変更は不要
5. 2〜3分後にデプロイ完了 → URLが発行される（例: `https://stock-watch-green.vercel.app`）

---

## ステップ2: スマホでアクセス

VercelのURL をスマホのブラウザで開くだけ！

**ホーム画面に追加する方法:**
- **iPhone**: Safari で開く → 共有ボタン → 「ホーム画面に追加」
- **Android**: Chrome で開く → メニュー → 「ホーム画面に追加」

アプリのように使えます。

---

## ウォッチリストのカスタマイズ（オプション）

デフォルト銘柄（トヨタ・ソニー・任天堂・ソフトバンク・AAPL・MSFT・NVDA・TSLA）を変更したい場合:

1. Vercel のプロジェクトページ → 「Settings」→「Environment Variables」
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
| Yahoo Finance API | **完全無料** |

**合計: 0円**
