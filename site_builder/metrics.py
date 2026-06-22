"""Shared metrics and chart data types for the ETF static site."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from strategy.constants import WEEKS_PER_YEAR


@dataclass(frozen=True)
class ReturnDistributionStats:
    count: int
    mean: float
    stdev: float
    skew: float
    min_return: float
    max_return: float
    ann_vol: float


@dataclass(frozen=True)
class MetricLineSeries:
    dates: list[date]
    values: list[float]


@dataclass(frozen=True)
class MetricScatterSeries:
    x: list[float]
    y: list[float]
    dates: list[date]


@dataclass(frozen=True)
class RollingMetricChart:
    slug: str
    title: str
    ylabel: str
    percent: bool
    us500: MetricLineSeries
    optimised: MetricLineSeries
    backtest: MetricLineSeries
    live: MetricLineSeries
    backtest_scatter: MetricScatterSeries
    live_scatter: MetricScatterSeries


@dataclass(frozen=True)
class BenchmarkRegressionStats:
    alpha_ann: float
    beta: float
    residual_vol_ann: float


def parse_timestamp(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def percentile_of(value: float, distribution: list[float]) -> float | None:
    if not distribution:
        return None
    below = sum(1 for sample in distribution if sample < value)
    return 100.0 * below / len(distribution)


def rolling_compound_returns(weekly_returns: list[float], window: int) -> list[float]:
    if window <= 0 or len(weekly_returns) < window:
        return []
    out: list[float] = []
    for index in range(window - 1, len(weekly_returns)):
        value = 1.0
        for weekly_return in weekly_returns[index - window + 1 : index + 1]:
            value *= 1.0 + weekly_return
        out.append(value - 1.0)
    return out


def distribution_stats(returns: list[float]) -> ReturnDistributionStats | None:
    if not returns:
        return None
    count = len(returns)
    mean = sum(returns) / count
    if count >= 2:
        variance = sum((value - mean) ** 2 for value in returns) / (count - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0
    if count >= 3 and stdev > 0:
        m3 = sum((value - mean) ** 3 for value in returns) / count
        skew = m3 / (stdev**3)
    else:
        skew = float("nan")
    ann_vol = stdev * math.sqrt(WEEKS_PER_YEAR) if stdev > 0 else 0.0
    return ReturnDistributionStats(
        count=count,
        mean=mean,
        stdev=stdev,
        skew=skew,
        min_return=min(returns),
        max_return=max(returns),
        ann_vol=ann_vol,
    )


def benchmark_regression_stats(
    strategy_returns: list[float],
    benchmark_returns: list[float],
) -> BenchmarkRegressionStats:
    """OLS of weekly strategy returns on benchmark: r_s = alpha + beta * r_b + eps."""
    nan = float("nan")
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("strategy and benchmark return series must have equal length")
    count = len(strategy_returns)
    if count < 2:
        return BenchmarkRegressionStats(alpha_ann=nan, beta=nan, residual_vol_ann=nan)

    mean_strategy = sum(strategy_returns) / count
    mean_benchmark = sum(benchmark_returns) / count
    covariance = sum(
        (strategy - mean_strategy) * (benchmark - mean_benchmark)
        for strategy, benchmark in zip(strategy_returns, benchmark_returns, strict=True)
    ) / (count - 1)
    benchmark_variance = sum(
        (benchmark - mean_benchmark) ** 2 for benchmark in benchmark_returns
    ) / (count - 1)
    if benchmark_variance <= 0:
        return BenchmarkRegressionStats(alpha_ann=nan, beta=nan, residual_vol_ann=nan)

    beta = covariance / benchmark_variance
    alpha_weekly = mean_strategy - beta * mean_benchmark
    residuals = [
        strategy - alpha_weekly - beta * benchmark
        for strategy, benchmark in zip(strategy_returns, benchmark_returns, strict=True)
    ]
    residual_mean = sum(residuals) / count
    residual_variance = sum((value - residual_mean) ** 2 for value in residuals) / (
        count - 2
    )
    residual_vol_weekly = math.sqrt(residual_variance) if residual_variance > 0 else 0.0
    return BenchmarkRegressionStats(
        alpha_ann=alpha_weekly * WEEKS_PER_YEAR,
        beta=beta,
        residual_vol_ann=residual_vol_weekly * math.sqrt(WEEKS_PER_YEAR),
    )


def drawdown_series(equity_curve: list[float]) -> list[float]:
    if not equity_curve:
        return []
    peak = equity_curve[0]
    series: list[float] = []
    for value in equity_curve:
        peak = max(peak, value)
        series.append(value / peak - 1.0 if peak > 0 else 0.0)
    return series


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return float("nan")
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def fraction_same_or_worse(value: float, distribution: list[float]) -> float | None:
    if not distribution:
        return None
    count = sum(1 for sample in distribution if sample <= value)
    return 100.0 * count / len(distribution)


def fraction_at_least(value: float, distribution: list[float]) -> float | None:
    if not distribution:
        return None
    count = sum(1 for sample in distribution if sample >= value)
    return 100.0 * count / len(distribution)


def days_since_ath_series(trade_dates: list[str], equities: list[float]) -> list[int]:
    if not trade_dates:
        return []
    peak_date = date.fromisoformat(trade_dates[0])
    peak_equity = equities[0] if equities else 0.0
    series: list[int] = []
    for iso_date, equity in zip(trade_dates, equities, strict=True):
        point_date = date.fromisoformat(iso_date)
        if equity >= peak_equity:
            peak_equity = equity
            peak_date = point_date
        series.append((point_date - peak_date).days)
    return series


def drawdown_exceedance_curve(
    drawdowns: list[float],
    *,
    steps: int = 100,
) -> tuple[list[float], list[float]]:
    if not drawdowns or steps < 2:
        return [], []
    low = min(drawdowns)
    xs = [low + (0.0 - low) * index / (steps - 1) for index in range(steps)]
    ys = [fraction_same_or_worse(level, drawdowns) or 0.0 for level in xs]
    return xs, ys


def ath_exceedance_curve(
    days_since_ath: list[int],
    *,
    steps: int = 100,
) -> tuple[list[float], list[float]]:
    if not days_since_ath or steps < 2:
        return [], []
    high = float(max(days_since_ath))
    xs = [high * index / (steps - 1) for index in range(steps)]
    floats = [float(value) for value in days_since_ath]
    ys = [fraction_at_least(level, floats) or 0.0 for level in xs]
    return xs, ys


def yahoo_close_prices_last_year(
    market_id: str,
    yahoo_dir: Path,
    *,
    as_of: date,
    days: int = 365,
) -> list[float]:
    path = yahoo_dir / f"{market_id}.csv"
    if not path.is_file():
        return []
    cutoff = as_of - timedelta(days=days)
    series: list[tuple[date, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_date = (row.get("Date") or "").strip()[:10]
            raw_close = (row.get("Close") or "").strip()
            if not raw_date or not raw_close:
                continue
            try:
                point_date = date.fromisoformat(raw_date)
                close = float(raw_close)
            except ValueError:
                continue
            if point_date < cutoff or point_date > as_of:
                continue
            series.append((point_date, close))
    series.sort(key=lambda item: item[0])
    return [close for _, close in series]


def total_return_from_prices(prices: list[float]) -> float | None:
    if len(prices) < 2 or prices[0] <= 0:
        return None
    return prices[-1] / prices[0] - 1.0
