# Phase 3 Report — phase3-20260708-125821
grid: False
universe: PIT quarterly (38 quarters)
regime: hmm

## Variant 3d
holdout reserved from: 2025-01-06 (untouched; final one-shot eval only)
OOS folds: [('2020-10-06', '2021-10-26'), ('2021-10-27', '2022-11-16'), ('2022-11-17', '2023-12-09'), ('2023-12-12', '2025-01-03')]
| metric | OOS | pass line |
|---|---|---|
| Sharpe | 0.18 | >= 0.6 |
| CAGR | 2.2% | >= 10% |
| MaxDD | -11.8% | >= -25.0% |
| Calmar | 0.26 | - |
| WF efficiency | 0.85 | >= 0.5 |
| DSR | 0.005 (N=573) | > 0.95 |

QQQ gate: **FAIL** (strategy Sharpe 0.18 vs QQQ 1.05; MaxDD -11.8% vs -35.1%)
Overfit alert (IS Sharpe): no
pass lines (R5): sharpe NG / cagr NG / max_dd OK / wf_eff OK / dsr@95 NG
**Branch: [C]**
