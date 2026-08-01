"""Trial log: EVERY evaluated configuration goes to SQLite (SPEC §8.3 rule 4).

The row count per experiment family is the N used to deflate the Sharpe
ratio (validation/dsr.py). GA generations, ablation runs, and manual
threshold tweaks all count.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from data.config_loader import data_root

UTC = timezone.utc


class TrialLogger:
    def __init__(self, root: Path | None = None):
        self.db = sqlite3.connect((root or data_root()) / "trials.sqlite")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                generation INTEGER,
                params TEXT NOT NULL,
                metrics TEXT NOT NULL,
                logged_at TEXT NOT NULL)"""
        )
        self.db.commit()

    def log(self, run_id: str, phase: str, params: dict, metrics: dict,
            generation: int | None = None) -> None:
        self.db.execute(
            "INSERT INTO trials (run_id, phase, generation, params, metrics, logged_at)"
            " VALUES (?,?,?,?,?,?)",
            (run_id, phase, generation,
             json.dumps(params, default=str), json.dumps(metrics, default=str),
             datetime.now(UTC).isoformat()),
        )
        self.db.commit()

    def count_trials(self, run_id: str | None = None) -> int:
        """DSR denominator. Without run_id counts EVERYTHING ever logged."""
        if run_id:
            row = self.db.execute(
                "SELECT COUNT(*) FROM trials WHERE run_id=?", (run_id,)).fetchone()
        else:
            row = self.db.execute("SELECT COUNT(*) FROM trials").fetchone()
        return int(row[0])

    def sharpe_variance(self, run_id: str) -> float | None:
        """Cross-trial variance of per-period Sharpe (input to E[max SR])."""
        rows = self.db.execute(
            "SELECT metrics FROM trials WHERE run_id=?", (run_id,)).fetchall()
        srs = []
        for (m,) in rows:
            d = json.loads(m)
            if "sharpe_per_bar" in d:
                srs.append(float(d["sharpe_per_bar"]))
        if len(srs) < 2:
            return None
        import numpy as np
        return float(np.var(srs, ddof=1))
