from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from site_builder.etf_data import (
    SummaryStats,
    RegimeReturnSeries,
    allocation_rows,
    ath_snapshot,
    drawdown_snapshot,
    period_returns,
    point_at_or_before,
    rebased_equity,
    regime_return_series,
    regime_unanimous_bearish_spans,
    regime_vote_rows,
    summary_stats,
    tracking_anchor_index,
)
from site_builder.metrics import BenchmarkRegressionStats, benchmark_regression_stats, days_since_ath_series, max_drawdown
from site_builder.etf_html import build_index_html
from site_builder.etf_plots import (
    _format_vol_cap_label,
    plot_etf_equity_since_tracking,
    plot_etf_vol_cap_equity,
)
from strategy.data import Asset, Universe


@dataclass(frozen=True)
class FakeVolCapCurve:
    vol_cap: float
    equity: list[float]
    sharpe: float
    cagr: float
    vol_ann: float
    max_drawdown: float


@dataclass(frozen=True)
class FakePoint:
    iso_date: str
    equity: float
    weekly_return: float
    invested_weight: float
    cash_weight: float
    effective_weights: dict[str, float]
    target_weights: dict[str, float]
    shadow_weekly_return: float = 0.0
    regime_votes: tuple[tuple[int, str], ...] = ()
    in_regime_cash: bool = False


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
    def test_max_drawdown(self) -> None:
        self.assertAlmostEqual(max_drawdown([1.0, 1.2, 0.9, 1.1]), -0.25)
        self.assertTrue(math.isnan(max_drawdown([])))

    def test_format_vol_cap_label(self) -> None:
        curve = FakeVolCapCurve(
            vol_cap=0.25,
            equity=[1.0],
            sharpe=1.23,
            cagr=0.118,
            vol_ann=0.241,
            max_drawdown=-0.182,
        )
        label = _format_vol_cap_label(curve)
        self.assertIn("25% cap", label)
        self.assertIn("Sharpe 1.23", label)
        self.assertIn("CAGR 11.8%", label)
        self.assertIn("vol 24.1%", label)
        self.assertIn("max DD -18.2%", label)

    def test_plot_etf_vol_cap_equity_writes_png(self) -> None:
        curves = [
            FakeVolCapCurve(
                vol_cap=0.10,
                equity=[1.0, 1.05, 1.08],
                sharpe=0.8,
                cagr=0.05,
                vol_ann=0.10,
                max_drawdown=-0.05,
            ),
            FakeVolCapCurve(
                vol_cap=0.25,
                equity=[1.0, 1.10, 1.15],
                sharpe=1.1,
                cagr=0.09,
                vol_ann=0.24,
                max_drawdown=-0.12,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "equity_vol_caps.png"
            plot_etf_vol_cap_equity(
                trade_dates=["2026-01-01", "2026-02-01", "2026-03-01"],
                curves=curves,
                tracking_start="2026-01-01",
                highlight_vol_cap=0.25,
                output=output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_etf_equity_since_tracking_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "equity_tracking.png"
            plot_etf_equity_since_tracking(
                trade_dates=["2026-01-01", "2026-02-01", "2026-03-01", "2026-06-01"],
                strat_equity=[1.0, 1.05, 1.08, 1.12],
                bench_equity=[1.0, 1.02, 1.03, 1.04],
                bench_label="VWRP",
                tracking_start="2026-02-01",
                output=output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

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

    def test_days_since_ath_series(self) -> None:
        dates = ["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22"]
        equities = [1.0, 1.1, 1.05, 1.12]
        self.assertEqual(days_since_ath_series(dates, equities), [0, 0, 7, 0])

    def test_ath_snapshot(self) -> None:
        points = [
            FakePoint("2026-01-01", 1.0, 0.0, 1.0, 0.0, {}, {}),
            FakePoint("2026-01-08", 1.1, 0.1, 1.0, 0.0, {}, {}),
            FakePoint("2026-01-15", 1.05, -0.05, 1.0, 0.0, {}, {}),
        ]
        snap = ath_snapshot(points)
        self.assertEqual(snap.days_since_ath, 7)
        self.assertAlmostEqual(snap.backtest_time_fraction_pct or 0.0, 100.0 / 3.0)

    def test_benchmark_regression_stats(self) -> None:
        benchmark = [0.01, 0.02, -0.01, 0.0]
        strategy = [value * 2.0 for value in benchmark]
        stats = benchmark_regression_stats(strategy, benchmark)
        self.assertAlmostEqual(stats.beta, 2.0)
        self.assertAlmostEqual(stats.alpha_ann, 0.0, places=9)
        self.assertAlmostEqual(stats.residual_vol_ann, 0.0, places=9)

    def test_point_at_or_before(self) -> None:
        points = [
            FakePoint("2025-01-01", 1.0, 0.0, 1.0, 0.0, {"a": 1.0}, {"a": 1.0}),
            FakePoint("2026-01-01", 1.1, 0.1, 1.0, 0.0, {"a": 0.8, "b": 0.2}, {"a": 1.0}),
        ]
        self.assertEqual(point_at_or_before(points, "2025-06-01").iso_date, "2025-01-01")
        self.assertEqual(point_at_or_before(points, "2026-06-01").iso_date, "2026-01-01")
        self.assertIsNone(point_at_or_before(points, "2024-01-01"))

    def test_allocation_rows_includes_weight_change_1y(self) -> None:
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
        point_1y_ago = FakePoint(
            "2025-06-01",
            1.0,
            0.0,
            0.82,
            0.18,
            {"ie00bk5bqt80": 0.82},
            {"ie00bk5bqt80": 1.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            rows = allocation_rows(
                universe,
                point,
                yahoo_dir=Path("/nonexistent"),
                spark_dir=Path(tmp) / "sparklines",
                as_of=date(2026, 6, 1),
                point_1y_ago=point_1y_ago,
            )
        self.assertEqual(len(rows), 2)
        etf_row = next(row for row in rows if row.market_id == "ie00bk5bqt80")
        cash_row = next(row for row in rows if row.market_id == "__cash__")
        self.assertAlmostEqual(etf_row.weight_change_1y or 0.0, 0.13)
        self.assertAlmostEqual(cash_row.weight_change_1y or 0.0, -0.13)

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
            shadow_weekly_return=0.02,
            regime_votes=((3, "invested"), (6, "cash"), (12, "invested")),
        )
        point_1y_ago = FakePoint(
            "2025-06-01",
            1.0,
            0.0,
            0.82,
            0.18,
            {"ie00bk5bqt80": 0.82},
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
                bench_regression=BenchmarkRegressionStats(0.01, 0.85, 0.06),
                bench_label="VWRP",
                drawdown=drawdown_snapshot([point]),
                ath=ath_snapshot([point]),
                period_returns=period_returns([point], [0.02], tracking_start="2026-06-20"),
                allocations=allocation_rows(
                    universe,
                    point,
                    yahoo_dir=Path("/nonexistent"),
                    spark_dir=Path(tmp) / "sparklines",
                    as_of=date(2026, 6, 1),
                    point_1y_ago=point_1y_ago,
                ),
                invested_weight=0.95,
                cash_weight=0.05,
                sharpe_1y=0.6,
                portfolio_url="https://investengine.com/share/portfolio/example/",
                regime_rows=[(3, 0.12, "bullish"), (6, -0.03, "bearish"), (12, 0.05, "bullish")],
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("ETF Engine", text)
            self.assertIn("InvestEngine portfolio", text)
            self.assertIn("https://investengine.com/share/portfolio/example/", text)
            self.assertIn("Portfolio weights", text)
            self.assertIn("Backtest weight", text)
            self.assertIn("(+13.00pp since 1y ago)", text)
            self.assertIn('style="color: green"', text)
            self.assertIn("tracking from 2026-06-20", text)
            self.assertIn("Days since ATH", text)
            self.assertIn("Beta vs VWRP", text)
            self.assertIn("equity_vol_caps.png", text)
            self.assertIn("equity_tracking.png", text)
            self.assertIn("Alpha vs VWRP", text)
            self.assertIn("Regime votes", text)
            self.assertIn('class="chart" src="regime_returns.png"', text)
            self.assertIn("3m", text)
            self.assertIn("bearish", text)
            self.assertIn("+12.00%", text)
            self.assertIn("-3.00%", text)
            self.assertNotIn("VWRP CAGR", text)

    def test_regime_unanimous_bearish_spans_merges_contiguous_weeks(self) -> None:
        series = [
            RegimeReturnSeries(
                months=3,
                dates=[
                    date(2024, 1, 1),
                    date(2024, 1, 8),
                    date(2024, 1, 15),
                    date(2024, 1, 22),
                ],
                values=[0.01, -0.02, -0.03, 0.01],
            ),
            RegimeReturnSeries(
                months=6,
                dates=[
                    date(2024, 1, 1),
                    date(2024, 1, 8),
                    date(2024, 1, 15),
                    date(2024, 1, 22),
                ],
                values=[0.02, -0.01, -0.04, -0.02],
            ),
            RegimeReturnSeries(
                months=12,
                dates=[
                    date(2024, 1, 1),
                    date(2024, 1, 8),
                    date(2024, 1, 15),
                    date(2024, 1, 22),
                ],
                values=[0.03, -0.02, -0.05, -0.01],
            ),
        ]
        spans = regime_unanimous_bearish_spans(series)
        self.assertEqual(len(spans), 1)
        self.assertLess(spans[0][0], date(2024, 1, 8))
        self.assertGreater(spans[0][1], date(2024, 1, 15))

    def test_regime_vote_rows_uses_shadow_returns(self) -> None:
        points = [
            FakePoint(
                "2025-01-01",
                1.0,
                0.0,
                1.0,
                0.0,
                {"a": 1.0},
                {"a": 1.0},
                shadow_weekly_return=0.0,
            ),
            FakePoint(
                "2026-06-01",
                0.9,
                -0.1,
                1.0,
                0.0,
                {"a": 1.0},
                {"a": 1.0},
                shadow_weekly_return=-0.1,
                regime_votes=((1, "cash"),),
            ),
        ]
        rows = regime_vote_rows(points, regime_months=(1,))
        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(rows[0][2], "bearish")


if __name__ == "__main__":
    unittest.main()
