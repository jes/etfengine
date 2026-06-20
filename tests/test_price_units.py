from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from strategy.data import (
    UNIT_SCALE_BOTH_DIRECTIONS,
    _bars_from_csv,
    _factor_entering_wrong_scale,
    patch_price_unit_series,
)


class PriceUnitPatchTests(unittest.TestCase):
    def test_exact_downward_hundred_x_uses_fixed_factor(self) -> None:
        factor = _factor_entering_wrong_scale(500.0, 5.0, 500.0)
        self.assertAlmostEqual(factor, 100.0)

    def test_exact_upward_hundred_x_uses_fixed_factor(self) -> None:
        factor = _factor_entering_wrong_scale(5.0, 500.0, 5.0)
        self.assertAlmostEqual(factor, 0.01)

    def test_non_exact_flip_restores_continuity_on_first_bad_day(self) -> None:
        factor = _factor_entering_wrong_scale(19.362, 1483.8, 19.362)
        self.assertAlmostEqual(1483.8 * factor, 19.362, places=6)

    def test_exits_wrong_scale_without_rescaling_return(self) -> None:
        df = pd.DataFrame(
            {"Open": [100, 100, 8, 8, 102], "High": [100, 100, 8, 8, 102],
             "Low": [100, 100, 8, 8, 102], "Close": [100, 100, 8, 8, 102]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        patched = patch_price_unit_series(df)
        self.assertAlmostEqual(patched.iloc[-1]["Close"], 102.0, places=6)
        self.assertAlmostEqual(patched["Close"].pct_change().iloc[-1], 0.02, places=6)

    def test_walk_forward_pence_run(self) -> None:
        df = pd.DataFrame(
            {
                "Open": [500.0, 505.0, 5.0, 5.1],
                "High": [510.0, 510.0, 5.2, 5.2],
                "Low": [495.0, 500.0, 4.8, 5.0],
                "Close": [505.0, 505.0, 5.0, 5.1],
                "Adj Close": [505.0, 505.0, 5.0, 5.1],
            },
            index=pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"]),
        )
        patched = patch_price_unit_series(df)
        self.assertEqual(patched.iloc[2]["Close"], 500.0)
        self.assertAlmostEqual(patched.iloc[3]["Close"], 510.0, places=6)
        self.assertEqual(patched.iloc[2]["Adj Close"], 500.0)
        self.assertAlmostEqual(patched.iloc[3]["Adj Close"], 510.0, places=6)

    def test_warns_on_both_scale_directions(self) -> None:
        df = pd.DataFrame(
            {"Open": [100, 100, 10000, 10000, 100, 1],
             "High": [100, 100, 10000, 10000, 100, 1],
             "Low": [100, 100, 10000, 10000, 100, 1],
             "Close": [100, 100, 10000, 10000, 100, 1]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]),
        )
        warnings: list[str] = []
        patch_price_unit_series(df, warnings=warnings)
        self.assertIn(UNIT_SCALE_BOTH_DIRECTIONS, warnings)

    def test_bars_from_csv_uses_adjusted_close_for_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fund.csv"
            df = pd.DataFrame(
                {
                    "Date": ["2020-01-03", "2020-01-10", "2020-01-17"],
                    "Open": [100.0, 100.0, 100.0],
                    "High": [100.0, 100.0, 100.0],
                    "Low": [100.0, 100.0, 100.0],
                    "Close": [100.0, 100.0, 100.0],
                    "Adj Close": [100.0, 101.0, 103.0],
                    "Volume": [0, 0, 0],
                }
            )
            df.to_csv(path, index=False)

            returns, daily_returns, ohlc, _ = _bars_from_csv(path)

        self.assertAlmostEqual(returns["2020-01-10"], 0.01)
        self.assertAlmostEqual(returns["2020-01-17"], 103.0 / 101.0 - 1.0)
        self.assertAlmostEqual(daily_returns["2020-01-10"], 0.01)
        self.assertAlmostEqual(ohlc["2020-01-17"].close, 100.0)


if __name__ == "__main__":
    unittest.main()
