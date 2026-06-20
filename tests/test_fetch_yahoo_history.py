from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from fetch_yahoo_history import (
    apply_adj_close_restatement,
    infer_adj_close_restatement_factor,
    merge_histories,
    restate_and_merge_histories,
    _normalize_history_df,
    read_existing_history,
    write_history_atomic,
)


def _ohlc_frame(
    rows: list[tuple[str, float, float | None]],
) -> pd.DataFrame:
    index = pd.to_datetime([row[0] for row in rows])
    df = pd.DataFrame({"Close": [row[1] for row in rows]}, index=index)
    if any(row[2] is not None for row in rows):
        df["Adj Close"] = [row[2] for row in rows]
    return df


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return _ohlc_frame([(row[0], row[1], None) for row in rows])


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

    def test_infer_adj_close_restatement_factor_on_overlap(self) -> None:
        existing = _ohlc_frame(
            [
                ("2026-06-01", 100.0, 100.0),
                ("2026-06-02", 101.0, 101.0),
            ]
        )
        new = _ohlc_frame(
            [
                ("2026-06-02", 101.0, 105.0),
                ("2026-06-03", 102.0, 106.05),
            ]
        )
        factor = infer_adj_close_restatement_factor(existing, new)
        self.assertAlmostEqual(factor or 0.0, 105.0 / 101.0, places=9)

    def test_infer_adj_close_restatement_factor_unchanged(self) -> None:
        existing = _ohlc_frame([("2026-06-02", 101.0, 101.0)])
        new = _ohlc_frame([("2026-06-02", 101.0, 101.0), ("2026-06-03", 102.0, 102.0)])
        self.assertIsNone(infer_adj_close_restatement_factor(existing, new))

    def test_infer_adj_close_restatement_factor_rejects_close_drift(self) -> None:
        existing = _ohlc_frame([("2026-06-02", 101.0, 101.0)])
        new = _ohlc_frame([("2026-06-02", 99.0, 105.0)])
        self.assertIsNone(infer_adj_close_restatement_factor(existing, new))

    def test_restate_and_merge_histories_rescales_history(self) -> None:
        existing = _ohlc_frame(
            [
                ("2026-06-01", 100.0, 100.0),
                ("2026-06-02", 101.0, 101.0),
            ]
        )
        new = _ohlc_frame(
            [
                ("2026-06-02", 101.0, 105.0),
                ("2026-06-03", 102.0, 106.05),
            ]
        )
        merged, factor = restate_and_merge_histories(existing, new)
        self.assertAlmostEqual(factor or 0.0, 105.0 / 101.0, places=9)
        self.assertAlmostEqual(float(merged.loc["2026-06-01", "Adj Close"]), 100.0 * 105.0 / 101.0)
        self.assertAlmostEqual(float(merged.loc["2026-06-02", "Adj Close"]), 105.0)
        self.assertAlmostEqual(float(merged.loc["2026-06-03", "Adj Close"]), 106.05)
        self.assertEqual(len(merged), 3)

    def test_apply_adj_close_restatement_leaves_close_unchanged(self) -> None:
        df = _ohlc_frame([("2026-06-01", 100.0, 100.0)])
        restated = apply_adj_close_restatement(df, 1.05)
        self.assertAlmostEqual(float(restated.loc["2026-06-01", "Close"]), 100.0)
        self.assertAlmostEqual(float(restated.loc["2026-06-01", "Adj Close"]), 105.0)
