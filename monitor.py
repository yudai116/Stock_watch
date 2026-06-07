"""
monitor.py — バックテスト進捗モニター
使い方: python monitor.py [--log pipeline_full.log] [--pid 18440] [--interval 5]
別ターミナルで実行して Ctrl+C で終了。
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows cp932 対策: stdout を UTF-8 に強制
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent

# ── ANSI カラー (Windows 10+/Windows Terminal 対応) ─────────────────────────
def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def cyan(t):    return _ansi("96", t)
def green(t):   return _ansi("92", t)
def yellow(t):  return _ansi("93", t)
def red(t):     return _ansi("91", t)
def bold(t):    return _ansi("1",  t)
def dim(t):     return _ansi("2",  t)


def bar(done: int, total: int, width: int = 28, color_fn=green) -> str:
    """ASCII プログレスバー (cp932対応)。"""
    if total <= 0:
        return dim("[" + "?" * width + "]")
    pct  = min(done / total, 1.0)
    fill = int(pct * width)
    b    = color_fn("#" * fill) + dim("." * (width - fill))
    return f"[{b}] {pct:5.1%}"


# ── ログパーサー ────────────────────────────────────────────────────────────

class PipelineState:
    def __init__(self):
        self.mode: str = "?"                  # swing / day
        self.phase: str = "起動中"            # fetch / hmm / strategy / done
        self.agent: str = "executor"          # architect / executor / checker / reviewer
        self.fold_current: int = 0
        self.fold_total: int   = 0
        self.rule_current: int = 0
        self.rule_total: int   = 0
        self.rule_name: str    = ""
        self.gen_current: int  = 0
        self.gen_total: int    = 0
        self.best_fit: float | None = None
        self.oos_folds: list[tuple[int, str]] = []   # [(fold_id, sharpe_str)]
        self.train_sharpe: float | None = None
        self.holdout_sharpe: float | None = None
        self.step: str = ""
        self.t_min: int = 0
        self.n_tickers: int = 0

    def parse(self, log_path: Path) -> None:
        if not log_path.exists():
            self.phase = "ログなし"
            return
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return
        self._parse_lines(lines)

    def _parse_lines(self, lines: list[str]) -> None:
        for line in lines:
            # ── フェーズ判定 ───────────────────────────────────────────
            if "[Step 1]" in line and "マクロ" in line:
                self.phase = "マクロデータ取得中"; self.agent = "executor"
            elif "[Step 1]" in line and "価格" in line:
                m = re.search(r"\((\w+)\)", line)
                self.mode  = m.group(1) if m else self.mode
                self.phase = f"Polygon価格取得 ({self.mode})"
                self.agent = "executor"
            elif "[Step 2]" in line:
                self.phase = "HMMレジーム検出中"; self.agent = "executor"
            elif "[Step 3]" in line:
                m = re.search(r"\((\w+)\)", line)
                self.mode  = m.group(1) if m else self.mode
                self.phase = f"GA最適化 ({self.mode.upper()})"
                self.agent = "executor"
            elif "保存]" in line:
                self.phase = "完了 (結果保存済)"
                self.agent = "reviewer"
            elif "[ERROR] レジーム検出失敗" in line:
                self.phase = "HMM失敗 → GA続行"
            # ── データロード ───────────────────────────────────────────
            elif "build_strategy_data:" in line:
                m = re.search(r"(\d+) 銘柄完了", line)
                if m: self.n_tickers = int(m.group(1))
            elif "T_min=" in line:
                m = re.search(r"T_min=(\d+)", line)
                if m: self.t_min = int(m.group(1))
            # ── WF フォールド ──────────────────────────────────────────
            elif "[WF]" in line:
                m = re.search(r"(\d+) フォールド", line)
                if m: self.fold_total = int(m.group(1))
            elif "── Fold" in line:
                m = re.search(r"Fold (\d+)", line)
                if m: self.fold_current = int(m.group(1))
                self.rule_current = 0; self.rule_name = ""
                self.gen_current  = 0; self.best_fit  = None
            # ── 売りルール ─────────────────────────────────────────────
            elif re.match(r"\s+\[(\d+)/(\d+)\]", line):
                m = re.match(r"\s+\[(\d+)/(\d+)\]\s+(\S+)", line)
                if m:
                    self.rule_current = int(m.group(1))
                    self.rule_total   = int(m.group(2))
                    self.rule_name    = m.group(3)
                    self.gen_current  = 0; self.best_fit = None
            # ── GA 世代 ────────────────────────────────────────────────
            elif "[GA gen" in line:
                m = re.search(r"gen\s+(\d+)/(\d+)\]\s+best_fit=([+-]?\d+\.\d+)", line)
                if m:
                    self.gen_current = int(m.group(1))
                    self.gen_total   = int(m.group(2))
                    self.best_fit    = float(m.group(3))
            elif "早期停止" in line:
                m = re.search(r"gen=(\d+)", line)
                if m:
                    self.gen_current = int(m.group(1))
            # ── OOS 結果 ───────────────────────────────────────────────
            elif "train_sharpe=" in line and "oos_sharpe=" in line:
                m = re.search(r"oos_sharpe=([^\s]+)", line)
                oos_str = m.group(1) if m else "?"
                self.oos_folds.append((self.fold_current, oos_str))
            # ── 最終ホールドアウト ──────────────────────────────────────
            elif "最終ホールドアウト評価" in line:
                self.fold_current = self.fold_total  # 全フォールド完了
                self.agent = "reviewer"
            elif "train_sharpe=" in line and "holdout:" not in line and "oos_sharpe" not in line:
                m = re.search(r"train_sharpe=([+-]?\d+\.\d+)", line)
                if m: self.train_sharpe = float(m.group(1))
            elif "holdout: sharpe=" in line:
                m = re.search(r"sharpe=([+-]?\d+\.\d+)", line)
                if m: self.holdout_sharpe = float(m.group(1))


# ── プロセス情報 ──────────────────────────────────────────────────────────────

def get_process_info(pid: int) -> dict:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-Process -Id {pid} -EA SilentlyContinue | "
             f"Select-Object CPU,WorkingSet64 | ConvertTo-Json"],
            capture_output=True, text=True, timeout=5,
        )
        import json
        d = json.loads(result.stdout.strip())
        return {"cpu": d.get("CPU", 0), "ram_mb": d.get("WorkingSet64", 0) // 1024 // 1024,
                "alive": True}
    except Exception:
        return {"cpu": 0, "ram_mb": 0, "alive": False}


# ── エージェント状態表示 ───────────────────────────────────────────────────────

AGENT_DESC = {
    "architect": ("Arch", "Architect", "次の改善案を設計中"),
    "executor":  ("Exec", "Executor",  "実装・GA最適化を実行中"),
    "checker":   ("Chck", "Checker",   "コード配線・結果を検証中"),
    "reviewer":  ("Rvwr", "Reviewer",  "統計的妥当性を評価中"),
}


def render(state: PipelineState, proc: dict, log_path: Path, interval: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    W   = 60
    sep = bold("=" * W)

    icon, aname, adesc = AGENT_DESC.get(state.agent, (">>", "?", "?"))
    agent_line = f"  [{icon}] {bold(cyan(aname))}  {dim(adesc)}"

    # プロセス状態
    if proc["alive"]:
        cpu_h = int(proc["cpu"]) // 3600
        cpu_m = (int(proc["cpu"]) % 3600) // 60
        proc_line = green(f"  PID {pid_arg}  稼働中  CPU {cpu_h}h{cpu_m:02d}m  RAM {proc['ram_mb']} MB")
    else:
        proc_line = red(f"  PID {pid_arg}  プロセス終了")

    lines = [
        sep,
        bold(f"  Stock Watch バックテスト進捗モニター   {dim(now)}"),
        sep,
        "",
        agent_line,
        "",
        f"  フェーズ:  {yellow(state.phase)}",
        f"  モード:    {state.mode.upper()}  ({state.n_tickers} 銘柄  T_min={state.t_min})",
        "",
    ]

    # フォールド進捗
    if state.fold_total > 0:
        lines += [
            bold("  -- Walk-Forward フォールド ---------------------"),
            f"  {bar(state.fold_current, state.fold_total, 30, green)}  "
            f"{state.fold_current}/{state.fold_total} 完了",
            "",
        ]

    # 売りルール進捗
    if state.rule_total > 0:
        lines += [
            bold("  -- 現在のフォールド: 売りルール ----------------"),
            f"  {bar(state.rule_current, state.rule_total, 30, cyan)}  "
            f"{state.rule_current}/{state.rule_total}  {dim(state.rule_name)}",
            "",
        ]

    # GA世代進捗
    if state.gen_total > 0:
        fit_str = f"  best_fit={state.best_fit:.3f}" if state.best_fit is not None else ""
        lines += [
            bold("  -- GA 世代 -------------------------------------"),
            f"  {bar(state.gen_current, state.gen_total, 30, yellow)}  "
            f"gen {state.gen_current}/{state.gen_total}{fit_str}",
            "",
        ]

    # OOS Sharpe 結果
    if state.oos_folds:
        lines.append(bold("  -- OOS Sharpe (完了フォールド) -----------------"))
        for fid, shp in state.oos_folds[-4:]:
            try:
                shp_f = float(shp) if shp not in ("None", "nan", "") else 0.0
                color = green if shp_f > 0 else red
            except ValueError:
                color = dim
            lines.append(f"    Fold {fid}: {color(shp)}")
        lines.append("")

    # 最終結果
    if state.holdout_sharpe is not None:
        col = green if state.holdout_sharpe > 0.5 else (yellow if state.holdout_sharpe > 0 else red)
        lines.append(f"  ホールドアウト Sharpe: {col(bold(f'{state.holdout_sharpe:.3f}'))}")
        lines.append("")

    lines += [
        sep,
        proc_line,
        dim(f"  ログ: {log_path.name}   更新間隔: {interval}秒   Ctrl+C で終了"),
        sep,
    ]
    return "\n".join(lines)


# ── メイン ────────────────────────────────────────────────────────────────────

pid_arg = 18440  # グローバル (render() から参照)

def main():
    global pid_arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",      default="pipeline_full.log")
    parser.add_argument("--pid",      type=int, default=18440)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--once",     action="store_true", help="1回表示して終了")
    args = parser.parse_args()

    pid_arg  = args.pid
    log_path = HERE / args.log
    interval = args.interval

    # Windows ANSIカラー有効化
    if sys.platform == "win32":
        os.system("color")
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetConsoleMode(k32.GetStdHandle(-11), 7)
        except Exception:
            pass

    print(f"Monitor: {log_path}  PID={pid_arg}  interval={interval}s")
    try:
        while True:
            state = PipelineState()
            state.parse(log_path)
            proc  = get_process_info(pid_arg)

            if not args.once:
                os.system("cls" if sys.platform == "win32" else "clear")
            print(render(state, proc, log_path, interval))
            sys.stdout.flush()

            if args.once:
                break

            if not proc["alive"] and state.holdout_sharpe is not None:
                print(green("\n  Pipeline complete! Check results files."))
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped")


if __name__ == "__main__":
    main()
