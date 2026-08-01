"""CSV export: files + manifest written, contents round-trip."""

import pandas as pd

from data.bitemporal_store import BitemporalStore
from data.export_bars import export_all


def _bars(n=5):
    ts = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"ts": ts, "open": [100.0] * n, "high": [101.0] * n,
                         "low": [99.0] * n, "close": [100.5] * n,
                         "volume": [1000.0] * n})


def test_export_writes_csv_and_manifest(tmp_path):
    store = BitemporalStore(tmp_path / "ds")
    store.put_bars("NVDA", "1d", _bars())
    store.put_bars("AMD", "1d", _bars(3))

    out = tmp_path / "exports"
    manifest = export_all(store, "1d", out_dir=out)
    assert set(manifest["symbol"]) == {"NVDA", "AMD"}
    assert (out / "NVDA.csv").exists() and (out / "_manifest.csv").exists()

    back = pd.read_csv(out / "NVDA.csv")
    assert len(back) == 5
    assert list(back.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert back["close"].iloc[0] == 100.5


def test_export_empty_store(tmp_path):
    manifest = export_all(BitemporalStore(tmp_path / "ds"), "1d",
                          out_dir=tmp_path / "exports")
    assert manifest.empty
