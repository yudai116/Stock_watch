#!/usr/bin/env python3
"""
PDCAループ 検証役・チェッカー自動実行エージェント
使用モデル: claude-opus-4-8 【固定・変更禁止】

使い方:
  python3 backtest/pdca_verify_agent.py --mode check    # チェッカー実行
  python3 backtest/pdca_verify_agent.py --mode improve  # 検証役実行（改善実装）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anthropic

# ── 設定 ─────────────────────────────────────────────────────────────────────
MODEL       = "claude-opus-4-8"   # 変更禁止
MAX_TOKENS  = 8096
MAX_TURNS   = 30
REPO_ROOT   = Path(__file__).parent.parent

HERE        = Path(__file__).parent
SPEC_F      = HERE / "VERIFICATION_SPEC.md"
CHECKER_F   = HERE / "VERIFICATION_CHECKER.md"
STATE_F     = HERE / "pdca_state.json"
SWING_F     = HERE / "strategy_results_swing.json"
DAY_F       = HERE / "strategy_results_day.json"
STRATEGY_F  = HERE / "strategy_search.py"


# ── ツール定義 ─────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "read_file",
        "description": "ファイルを読み込む",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "ファイルパス（絶対パスまたはリポジトリルートからの相対パス）"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "edit_file",
        "description": "ファイルの特定テキストを置換する（安全な部分編集）",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":     {"type": "string", "description": "編集するファイルパス"},
                "old_text": {"type": "string", "description": "置換対象のテキスト（完全一致）"},
                "new_text": {"type": "string", "description": "置換後のテキスト"}
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
    {
        "name": "write_file",
        "description": "ファイルを丸ごと書き換える（edit_file で対応できない場合のみ使用）",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_command",
        "description": "シェルコマンドを実行する（Pythonスクリプト実行・git diff確認等）",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "実行するコマンド"}
            },
            "required": ["command"]
        }
    }
]


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def execute_tool(name: str, inp: dict) -> str:
    try:
        if name == "read_file":
            p = _resolve(inp["path"])
            if not p.exists():
                return f"[ERROR] ファイルが見つかりません: {p}"
            content = p.read_text(encoding="utf-8")
            # 長すぎる場合は先頭部分のみ
            if len(content) > 60000:
                return content[:60000] + f"\n\n... (省略: 全{len(content)}文字)"
            return content

        elif name == "edit_file":
            p = _resolve(inp["path"])
            if not p.exists():
                return f"[ERROR] ファイルが見つかりません: {p}"
            content = p.read_text(encoding="utf-8")
            old = inp["old_text"]
            new = inp["new_text"]
            if old not in content:
                return f"[ERROR] 置換対象テキストが見つかりません:\n{old[:200]}"
            count = content.count(old)
            if count > 1:
                return f"[ERROR] 置換対象が{count}箇所あります。より具体的なテキストを指定してください"
            p.write_text(content.replace(old, new, 1), encoding="utf-8")
            return f"[OK] 編集完了: {p}"

        elif name == "write_file":
            p = _resolve(inp["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inp["content"], encoding="utf-8")
            return f"[OK] 書き込み完了: {p}"

        elif name == "run_command":
            result = subprocess.run(
                inp["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(REPO_ROOT)
            )
            out = (result.stdout or "") + (result.stderr or "")
            if len(out) > 8000:
                out = out[:8000] + "\n... (省略)"
            return out if out else "(出力なし)"

    except Exception as e:
        return f"[ERROR] {name}: {e}"

    return "[ERROR] 不明なツール"


def run_agent(mode: str) -> str:
    """エージェントを実行して最終テキストを返す"""

    client = anthropic.Anthropic()

    # ── プロンプト構築 ──────────────────────────────────────────────────────
    if mode == "check":
        spec = CHECKER_F.read_text(encoding="utf-8")
        system_prompt = (
            "あなたは検証チェッカーエージェントです。"
            "VERIFICATION_CHECKER.md の仕様に従って実装を確認し、詳細なレポートを出力してください。"
            "コードは変更しないこと。"
        )
        user_message = spec

    elif mode == "improve":
        spec     = SPEC_F.read_text(encoding="utf-8")
        state_txt = STATE_F.read_text(encoding="utf-8") if STATE_F.exists() else "{}"
        swing_txt = SWING_F.read_text(encoding="utf-8") if SWING_F.exists() else "null"
        day_txt   = DAY_F.read_text(encoding="utf-8")   if DAY_F.exists() else "null"

        system_prompt = (
            "あなたは検証役エージェントです。"
            "VERIFICATION_SPEC.md の仕様に従って、アルゴリズムの改善を提案し、"
            "backtest/strategy_search.py に直接実装してください。"
            "提案だけでは不十分です。必ずコードを編集してください。"
        )
        user_message = f"""{spec}

---

## 今回の実験データ

### pdca_state.json（実験履歴）
```json
{state_txt}
```

### strategy_results_swing.json（スイング結果）
```json
{swing_txt[:3000]}
```

### strategy_results_day.json（デイトレ結果）
```json
{day_txt[:3000]}
```

---

## 実行指示

上記の仕様書（VERIFICATION_SPEC.md）と実験データをもとに、以下の順番で作業してください:

1. `backtest/strategy_search.py` を read_file ツールで読む
2. Step 1: 現状コードの問題点を特定する
3. Step 2: 実験結果と照合して原因を特定する
4. Step 3: VERIFICATION_SPEC.md の「改善案フォーマット」に従って改善案を立案する
5. 改善案を `edit_file` ツールを使って `backtest/strategy_search.py` に実装する
6. 実装完了後、`run_command` で `git diff backtest/strategy_search.py` を実行して変更内容を確認する
7. 最終レポートを出力する（Step 3 フォーマット準拠）
"""
    else:
        print(f"[ERROR] 不明なモード: {mode}", file=sys.stderr)
        sys.exit(1)

    # ── エージェントループ ─────────────────────────────────────────────────
    messages: list[dict] = [{"role": "user", "content": user_message}]
    final_text = ""

    print(f"[pdca_verify_agent] モード={mode} モデル={MODEL} 開始", file=sys.stderr)

    for turn in range(MAX_TURNS):
        print(f"[turn {turn+1}/{MAX_TURNS}] API呼び出し中...", file=sys.stderr)

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )
                break
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                print(f"  レート制限 → {wait}秒待機...", file=sys.stderr)
                time.sleep(wait)
            except anthropic.APIError as e:
                print(f"  APIエラー: {e}", file=sys.stderr)
                if attempt == 2:
                    raise
                time.sleep(10)

        # アシスタント応答を messages に追加
        messages.append({"role": "assistant", "content": response.content})

        # テキスト部分を収集
        for block in response.content:
            if hasattr(block, "text") and block.text:
                final_text += block.text + "\n"
                print(block.text, flush=True)

        # ツール呼び出しがある場合は実行
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [ツール] {block.name}({json.dumps(block.input, ensure_ascii=False)[:100]})", file=sys.stderr)
                    result = execute_tool(block.name, block.input)
                    print(f"  [結果] {result[:200]}", file=sys.stderr)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # end_turn → 完了
        if response.stop_reason == "end_turn":
            print(f"[pdca_verify_agent] 完了 ({turn+1}ターン)", file=sys.stderr)
            break

    return final_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "improve"], required=True)
    parser.add_argument("--output", type=str, default=None, help="結果を保存するファイルパス")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ERROR] ANTHROPIC_API_KEY が設定されていません", file=sys.stderr)
        sys.exit(1)

    result = run_agent(args.mode)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"[pdca_verify_agent] 結果を {args.output} に保存しました", file=sys.stderr)

    # チェッカーモードの場合は終了コードで PASS/FAIL を伝える
    if args.mode == "check":
        if "RESULT: FAIL" in result:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
