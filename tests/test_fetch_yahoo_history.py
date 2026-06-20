from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from fetch_yahoo_history import (
    merge_histories,
    _normalize_history_df,
    read_existing_history,
    write_history_atomic,
)


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    index = pd.to_datetime([row[0] for row in rows])
    return pd.DataFrame({"Close": [row[1] for row in rows]}, index=index)


class FetchYahooHistoryTests(unittest.TestCase):
    def test_merge_histories_appends_new_dates(self) -> None:
        existing = _frame([("2026-06-01", 100.0), ("2026-06-02", 101.0)])
        new = _frame([("2026-06-03", 102.0)])
        merged = merge_histories(existing, new)
        self.assertEqual(len(merged), 3)
        self.assertAlmostEqual(float(merged.loc["2026-06-03", "Close"]), 102.0)

    def test_merge_histories_replaces_duplicate_date(self) -> None:
        existing = _frame([("2026-06-01", 100.0)])
        new = _frame([("2026-06-01", 105.0), ("2026-06-02", 106.0)])
        merged = merge_histories(existing, new)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(float(merged.loc["2026-06-01", "Close"]), 105.0)

    def test_atomic_write_and_read_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.csv"
            df = _frame([("2026-06-01", 100.0), ("2026-06-02", 101.0)])
            df["Open"] = df["Close"]
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            df["Volume"] = 0
            write_history_atomic(path, df)
            loaded = read_existing_history(path)
            assert loaded is not None
            self.assertEqual(len(loaded), 2)
            self.assertEqual(pd.Timestamp(loaded.index[-1]).date(), date(2026, 6, 2))

    def test_normalize_history_preserves_adjusted_close(self) -> None:
        df = _frame([("2026-06-01", 100.0)])
        df["Adj Close"] = [99.5]
        out = _normalize_history_df(df)

        self.assertIn("Adj Close", out.columns)
        self.assertAlmostEqual(float(out.iloc[0]["Adj Close"]), 99.5)
