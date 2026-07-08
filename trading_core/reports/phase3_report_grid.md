# Phase 3 Report — phase3-20260708-130127
grid: True
universe: PIT quarterly (38 quarters)
regime: simple

## Variant 3b
holdout reserved from: 2025-01-06 (untouched; final one-shot eval only)
OOS folds: [('2020-10-06', '2021-10-26'), ('2021-10-27', '2022-11-16'), ('2022-11-17', '2023-12-09'), ('2023-12-12', '2025-01-03')]
| metric | OOS | pass line |
|---|---|---|
| Sharpe | 0.15 | >= 0.6 |
| CAGR | 0.4% | >= 10% |
| MaxDD | -5.4% | >= -25.0% |
| Calmar | 0.28 | - |
| WF efficiency | 0.60 | >= 0.5 |
| DSR | 0.001 (N=897) | > 0.95 |

QQQ gate: **FAIL** (strategy Sharpe 0.15 vs QQQ 1.05; MaxDD -5.4% vs -35.1%)
Overfit alert (IS Sharpe): no
pass lines (R5): sharpe NG / cagr NG / max_dd OK / wf_eff OK / dsr@95 NG
**Branch: [C]**
