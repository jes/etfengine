from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from site_builder.etf_data import (
    SummaryStats,
    allocation_rows,
    drawdown_snapshot,
    period_returns,
    rebased_equity,
    summary_stats,
    tracking_anchor_index,
)
from site_builder.etf_html import build_index_html
from strategy.data import Asset, Universe


@dataclass(frozen=True)
class FakePoint:
    iso_date: str
    equity: float
    weekly_return: float
    invested_weight: float
    cash_weight: float
    effective_weights: dict[str, float]
    target_weights: dict[str, float]


def _fake_universe() -> Universe:
    asset = Asset(
        market_id="ie00bk5bqt80",
        name="VWRP",
        yahoo_ticker="VWRP.L",
        returns_by_date={"2026-01-01": 0.01, "2026-06-01": 0.02},
        daily_returns_by_date={},
        ohlc_by_date={},
        first_date="2026-01-01",
    )
    return Universe(
        assets={"ie00bk5bqt80": asset},
        weekly_dates=["2026-01-01", "2026-06-01"],
        market_names={"ie00bk5bqt80": "VWRP"},
        market_categories={"ie00bk5bqt80": "ETF"},
        spread_fraction={"ie00bk5bqt80": 0.001},
    )


class EtfSiteBuilderTests(unittest.TestCase):
    def test_tracking_anchor_index(self) -> None:
        dates = ["2025-01-01", "2026-01-01", "2026-06-01"]
        self.assertEqual(tracking_anchor_index(dates, "2026-06-20"), 2)
        self.assertEqual(tracking_anchor_index(dates, "2025-06-01"), 0)

    def test_rebased_equity(self) -> None:
        values = [1.0, 1.1, 1.2]
        self.assertEqual(rebased_equity(values, 1), [1.0 / 1.1, 1.0, 1.2 / 1.1])

    def test_period_returns_since_tracking_start(self) -> None:
        points = [
            FakePoint("2026-01-01", 1.0, 0.0, 1.0, 0.0, {"a": 1.0}, {"a": 1.0}),
            FakePoint("2026-06-01", 1.1, 0.1, 1.0, 0.0, {"a": 1.0}, {"a": 1.0}),
        ]
        rows = period_returns(points, [0.0, 0.1], tracking_start="2026-01-01")
        since = next(row for row in rows if row.label == "Since tracking start")
        self.assertAlmostEqual(since.return_pct, 0.1)

    def test_period_returns_past_days_ignore_future_tracking_start(self) -> None:
        points = [
            FakePoint("2026-05-01", 1.0, 0.0, 1.0, 0.0, {"a": 1.0}, {"a": 1.0}),
            FakePoint("2026-06-19", 1.1, 0.1, 1.0, 0.0, {"a": 1.0}, {"a": 1.0}),
        ]
        rows = period_returns(points, [0.0, 0.1], tracking_start="2026-06-20")
        past_30 = next(row for row in rows if row.label == "Past 30 days")
        since = next(row for row in rows if row.label == "Since tracking start")
        self.assertAlmostEqual(past_30.return_pct, 0.1)
        self.assertTrue(math.isnan(since.return_pct))

    def test_drawdown_snapshot(self) -> None:
        points = [
            FakePoint("2026-01-01", 1.0, 0.0, 1.0, 0.0, {}, {}),
            FakePoint("2026-06-01", 0.9, -0.1, 1.0, 0.0, {}, {}),
        ]
        snap = drawdown_snapshot(points)
        self.assertAlmostEqual(snap.drawdown_pct or 0.0, -0.1)

    def test_build_index_html_writes_file(self) -> None:
        universe = _fake_universe()
        point = FakePoint(
            "2026-06-01",
            1.1,
            0.02,
            0.95,
            0.05,
            {"ie00bk5bqt80": 0.95},
            {"ie00bk5bqt80": 1.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "index.html"
            build_index_html(
                output=out,
                universe=universe,
                generated_at="2026-06-20 12:00:00",
                tracking_start="2026-06-20",
                as_of_date="2026-06-01",
                strat_stats=SummaryStats(0.1, 0.2, 0.5, 0.08),
                drawdown=drawdown_snapshot([point]),
                period_returns=period_returns([point], [0.02], tracking_start="2026-06-20"),
                allocations=allocation_rows(
                    universe,
                    point,
                    yahoo_dir=Path("/nonexistent"),
                    spark_dir=Path(tmp) / "sparklines",
                    as_of=date(2026, 6, 1),
                ),
                invested_weight=0.95,
                cash_weight=0.05,
                sharpe_1y=0.6,
                portfolio_url="https://investengine.com/share/portfolio/example/",
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("ETF Engine", text)
            self.assertIn("InvestEngine portfolio", text)
            self.assertIn("https://investengine.com/share/portfolio/example/", text)
            self.assertIn("Portfolio weights", text)
            self.assertIn("Backtest weight", text)
            self.assertIn("tracking from 2026-06-20", text)
            self.assertNotIn("VWRP CAGR", text)


if __name__ == "__main__":
    unittest.main()
