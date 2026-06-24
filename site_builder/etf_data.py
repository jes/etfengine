"""Derived metrics for the ETF static site (backtest-only, no live account)."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

from site_builder.metrics import (
    MetricLineSeries,
    MetricScatterSeries,
    ReturnDistributionStats,
    RollingMetricChart,
    days_since_ath_series,
    distribution_stats,
    drawdown_series,
    fraction_at_least,
    fraction_same_or_worse,
    percentile_of,
    rolling_compound_returns,
    total_return_from_prices,
    yahoo_close_prices_last_year,
)
from strategy.constants import RISK_FREE_ID, WEEKS_PER_YEAR
from strategy.data import Universe
from strategy.regime import (
    REGIME_MONTHS,
    selection_weeks,
    trailing_compound_return,
    vote_label,
)


class WeekPointLike(Protocol):
    iso_date: str
    equity: float
    weekly_return: float
    invested_weight: float
    cash_weight: float
    effective_weights: dict[str, float]
    target_weights: dict[str, float]
    shadow_weekly_return: float
    regime_votes: tuple[tuple[int, str], ...]
    in_regime_cash: bool


@dataclass(frozen=True)
class PeriodReturnRow:
    label: str
    return_pct: float
    percentile: float | None


@dataclass(frozen=True)
class DrawdownSnapshot:
    drawdown_pct: float | None
    backtest_time_fraction_pct: float | None


@dataclass(frozen=True)
class AthSnapshot:
    days_since_ath: int | None
    backtest_time_fraction_pct: float | None


@dataclass(frozen=True)
class SummaryStats:
    mean_ann: float
    vol_ann: float
    sharpe: float
    cagr: float


@dataclass(frozen=True)
class RegimeReturnSeries:
    months: int
    dates: list[date]
    values: list[float]


def regime_unanimous_bearish_spans(
    series: list[RegimeReturnSeries],
) -> list[tuple[date, date]]:
    """Date spans where every regime horizon has a negative trailing return."""
    if len(series) < 3:
        return []
    months_set = {item.months for item in series}
    value_by_date: dict[date, dict[int, float]] = {}
    all_dates: set[date] = set()
    for item in series:
        for d, value in zip(item.dates, item.values):
            all_dates.add(d)
            value_by_date.setdefault(d, {})[item.months] = value
    sorted_all = sorted(all_dates)
    date_index = {d: index for index, d in enumerate(sorted_all)}

    def unanimous_bearish(d: date) -> bool:
        values = value_by_date.get(d, {})
        return (
            len(values) == len(months_set)
            and all(values[months] < 0.0 for months in months_set)
        )

    bearish_dates = [d for d in sorted_all if unanimous_bearish(d)]
    spans: list[tuple[date, date]] = []
    index = 0
    while index < len(bearish_dates):
        end_index = index + 1
        while end_index < len(bearish_dates):
            if date_index[bearish_dates[end_index]] != date_index[bearish_dates[end_index - 1]] + 1:
                break
            end_index += 1
        start = bearish_dates[index]
        end = bearish_dates[end_index - 1]
        start_idx = date_index[start]
        end_idx = date_index[end]
        if start_idx > 0:
            prev = sorted_all[start_idx - 1]
            x0 = prev + timedelta(days=(start - prev).days // 2)
        else:
            x0 = start - timedelta(days=3)
        if end_idx + 1 < len(sorted_all):
            nxt = sorted_all[end_idx + 1]
            x1 = end + timedelta(days=(nxt - end).days // 2)
        else:
            x1 = end + timedelta(days=3)
        spans.append((x0, x1))
        index = end_index
    return spans


def shadow_returns_from_points(points: list[WeekPointLike]) -> list[float]:
    return [point.shadow_weekly_return for point in points]


def regime_return_series(
    points: list[WeekPointLike],
    *,
    regime_months: tuple[int, ...] = REGIME_MONTHS,
) -> list[RegimeReturnSeries]:
    """Trailing shadow-book returns used by regime votes for each backtest week."""
    regime_weeks = selection_weeks(regime_months)
    series_by_months = [
        RegimeReturnSeries(months=months, dates=[], values=[])
        for months in regime_months
    ]
    shadow_returns: list[float] = []
    for index, point in enumerate(points):
        shadow_returns.append(point.shadow_weekly_return)
        vote_months = {months for months, _vote in point.regime_votes}
        for series, months, weeks in zip(
            series_by_months,
            regime_months,
            regime_weeks,
            strict=True,
        ):
            if months not in vote_months:
                continue
            trailing = trailing_compound_return(shadow_returns, weeks)
            if trailing == trailing:
                series.dates.append(date.fromisoformat(point.iso_date))
                series.values.append(trailing)
    return series_by_months


def regime_vote_rows(
    points: list[WeekPointLike],
    *,
    regime_months: tuple[int, ...] = REGIME_MONTHS,
) -> list[tuple[int, float | None, str]] | None:
    if not points:
        return None
    latest = points[-1]
    if not latest.regime_votes:
        return None
    votes = dict(latest.regime_votes)
    shadow_returns = shadow_returns_from_points(points)
    rows: list[tuple[int, float | None, str]] = []
    for months, weeks in zip(
        regime_months,
        selection_weeks(regime_months),
        strict=True,
    ):
        trailing = trailing_compound_return(shadow_returns, weeks)
        vote = votes.get(months, "")
        trailing_value = None if trailing != trailing else trailing
        label = vote_label(vote) if vote else "unknown"
        rows.append((months, trailing_value, label))
    return rows


@dataclass(frozen=True)
class AllocationRow:
    market_id: str
    label: str
    weight_pct: float
    spark_path: str
    return_1y: float | None
    ie_weight_pct: float | None = None
    icon_path: str = ""
    weight_change_1m: float | None = None
    weight_change_1y: float | None = None


def tracking_anchor_index(dates: list[str], tracking_start: str) -> int:
    anchor = 0
    for index, iso_date in enumerate(dates):
        if iso_date <= tracking_start:
            anchor = index
        else:
            break
    return anchor


def equity_at_or_before(points: list[WeekPointLike], iso_date: str) -> float | None:
    equity: float | None = None
    for point in points:
        if point.iso_date <= iso_date:
            equity = point.equity
        else:
            break
    return equity


def point_at_or_before(points: list[WeekPointLike], iso_date: str) -> WeekPointLike | None:
    selected: WeekPointLike | None = None
    for point in points:
        if point.iso_date <= iso_date:
            selected = point
        else:
            break
    return selected


def rebased_equity(values: list[float], anchor_index: int) -> list[float]:
    if not values:
        return []
    anchor_value = values[anchor_index] if 0 <= anchor_index < len(values) else values[0]
    if anchor_value <= 0:
        return list(values)
    scale = 1.0 / anchor_value
    return [value * scale for value in values]


def _cagr(returns: list[float]) -> float:
    if not returns:
        return float("nan")
    value = 1.0
    for period_return in returns:
        value *= 1.0 + period_return
    years = len(returns) / WEEKS_PER_YEAR
    if years <= 0 or value <= 0:
        return float("nan")
    return value ** (1.0 / years) - 1.0


def _ann_vol(returns: list[float]) -> float:
    if len(returns) < 2:
        return float("nan")
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(WEEKS_PER_YEAR)


def _sharpe(returns: list[float], rf_returns: list[float]) -> float:
    if len(returns) < 2:
        return float("nan")
    vol = _ann_vol(returns)
    if vol != vol or vol <= 0:
        return float("nan")
    avg = sum(returns) / len(returns)
    avg_rf = sum(rf_returns) / len(rf_returns) if rf_returns else 0.0
    return ((avg - avg_rf) * WEEKS_PER_YEAR) / vol


def _metric_value(kind: str, returns: list[float], rf_returns: list[float]) -> float:
    if kind == "sharpe":
        return _sharpe(returns, rf_returns)
    if kind == "cagr":
        return _cagr(returns)
    if kind == "vol":
        return _ann_vol(returns)
    raise ValueError(f"unknown metric kind: {kind}")


def _rf_returns_for_dates(universe: Universe, dates: list[str]) -> list[float]:
    rf = universe.assets.get(RISK_FREE_ID)
    if rf is None:
        return []
    rf_dates = sorted(rf.returns_by_date)
    values: list[float] = []
    for iso_date in dates:
        value = rf.returns_by_date.get(iso_date)
        if value is None:
            index = bisect.bisect_right(rf_dates, iso_date) - 1
            if index < 0:
                continue
            value = rf.returns_by_date[rf_dates[index]]
        values.append(value)
    return values


def _window_dates_for(universe: Universe, iso_date: str, weeks: int) -> list[str]:
    end = bisect.bisect_right(universe.weekly_dates, iso_date)
    start = max(0, end - weeks)
    return universe.weekly_dates[start:end]


def _portfolio_weekly_returns(
    weights: dict[str, float],
    universe: Universe,
    history_dates: list[str],
) -> list[float]:
    returns: list[float] = []
    for iso_date in history_dates:
        gross = 0.0
        for market_id, weight in weights.items():
            gross += weight * universe.assets[market_id].returns_by_date.get(iso_date, 0.0)
        returns.append(gross)
    return returns


def _lookback_scatter_from_returns(
    x_values: list[float],
    returns: list[float],
    rf_returns: list[float],
    dates: list[date],
    *,
    kind: str,
    weeks: int,
) -> MetricScatterSeries:
    xs: list[float] = []
    ys: list[float] = []
    out_dates: list[date] = []
    limit = min(
        len(x_values),
        len(returns),
        len(dates),
        len(rf_returns) if rf_returns else len(returns),
    )
    for index in range(weeks - 1, limit):
        x_value = x_values[index]
        if x_value != x_value:
            continue
        window = returns[index - weeks + 1 : index + 1]
        rf_window = rf_returns[index - weeks + 1 : index + 1]
        y_value = _metric_value(kind, window, rf_window)
        if y_value == y_value:
            xs.append(x_value)
            ys.append(y_value)
            out_dates.append(dates[index])
    return MetricScatterSeries(xs, ys, out_dates)


def _rolling_line_from_returns(
    dates: list[date],
    returns: list[float],
    rf_returns: list[float],
    *,
    kind: str,
    weeks: int,
) -> MetricLineSeries:
    out_dates: list[date] = []
    values: list[float] = []
    limit = min(len(dates), len(returns), len(rf_returns) if rf_returns else len(returns))
    for index in range(weeks - 1, limit):
        window = returns[index - weeks + 1 : index + 1]
        rf_window = rf_returns[index - weeks + 1 : index + 1]
        value = _metric_value(kind, window, rf_window)
        if value == value:
            out_dates.append(dates[index])
            values.append(value)
    return MetricLineSeries(out_dates, values)


def summary_stats(
    weekly_returns: list[float],
    rf_returns: list[float],
) -> SummaryStats:
    mean_ann = sum(weekly_returns) / len(weekly_returns) * WEEKS_PER_YEAR if weekly_returns else float("nan")
    return SummaryStats(
        mean_ann=mean_ann,
        vol_ann=_ann_vol(weekly_returns),
        sharpe=_sharpe(weekly_returns, rf_returns),
        cagr=_cagr(weekly_returns),
    )


def drawdown_snapshot(points: list[WeekPointLike]) -> DrawdownSnapshot:
    if not points:
        return DrawdownSnapshot(drawdown_pct=None, backtest_time_fraction_pct=None)
    equities = [point.equity for point in points]
    drawdowns = drawdown_series(equities)
    current = drawdowns[-1]
    return DrawdownSnapshot(
        drawdown_pct=current,
        backtest_time_fraction_pct=fraction_same_or_worse(current, drawdowns),
    )


def ath_snapshot(points: list[WeekPointLike]) -> AthSnapshot:
    if not points:
        return AthSnapshot(days_since_ath=None, backtest_time_fraction_pct=None)
    dates = [point.iso_date for point in points]
    equities = [point.equity for point in points]
    series = days_since_ath_series(dates, equities)
    current = series[-1]
    floats = [float(value) for value in series]
    return AthSnapshot(
        days_since_ath=current,
        backtest_time_fraction_pct=fraction_at_least(float(current), floats),
    )


def period_returns(
    points: list[WeekPointLike],
    weekly_returns: list[float],
    *,
    tracking_start: str,
) -> list[PeriodReturnRow]:
    if not points:
        return []
    end_date = points[-1].iso_date
    end_equity = points[-1].equity
    specs: list[tuple[str, int | None]] = [
        ("Past 7 days", 7),
        ("Past 30 days", 30),
        ("Past 365 days", 365),
        ("Since tracking start", None),
    ]
    results: list[PeriodReturnRow] = []
    end = date.fromisoformat(end_date)
    tracking_start_date = date.fromisoformat(tracking_start)
    for label, days in specs:
        if days is None:
            if tracking_start_date > end:
                results.append(
                    PeriodReturnRow(
                        label=label,
                        return_pct=float("nan"),
                        percentile=None,
                    )
                )
                continue
            start_iso = tracking_start
            week_window = max(1, (end - tracking_start_date).days // 7)
        else:
            start_iso = (end - timedelta(days=days)).isoformat()
            week_window = max(1, days // 7)
        start_equity = equity_at_or_before(points, start_iso)
        if start_equity is None or start_equity <= 0:
            results.append(
                PeriodReturnRow(label=label, return_pct=float("nan"), percentile=None)
            )
            continue
        ret = end_equity / start_equity - 1.0
        dist = rolling_compound_returns(weekly_returns, week_window)
        results.append(
            PeriodReturnRow(
                label=label,
                return_pct=ret,
                percentile=percentile_of(ret, dist),
            )
        )
    return results


def rolling_metric_charts(
    *,
    points: list[WeekPointLike],
    bench_returns: list[float],
    universe: Universe,
    weeks: int = WEEKS_PER_YEAR,
) -> list[RollingMetricChart]:
    point_dates = [date.fromisoformat(point.iso_date) for point in points]
    point_iso_dates = [point.iso_date for point in points]
    strat_returns = [point.weekly_return for point in points]
    strat_rf = _rf_returns_for_dates(universe, point_iso_dates)

    bench_dates: list[date] = []
    bench_series: list[float] = []
    bench_iso_dates: list[str] = []
    for point_date, bench_return in zip(point_iso_dates, bench_returns):
        if bench_return != bench_return:
            continue
        bench_dates.append(date.fromisoformat(point_date))
        bench_series.append(bench_return)
        bench_iso_dates.append(point_date)
    bench_rf = _rf_returns_for_dates(universe, bench_iso_dates)

    optimised_dates: list[date] = []
    optimised_line_by_kind: dict[str, list[float]] = {
        "sharpe": [],
        "cagr": [],
        "vol": [],
    }
    optimised_point_by_kind: dict[str, list[float]] = {
        "sharpe": [],
        "cagr": [],
        "vol": [],
    }
    for point in points:
        history = _window_dates_for(universe, point.iso_date, weeks)
        if len(history) < weeks:
            for kind in optimised_point_by_kind:
                optimised_point_by_kind[kind].append(float("nan"))
            continue
        frozen_returns = _portfolio_weekly_returns(
            point.target_weights,
            universe,
            history,
        )
        if len(frozen_returns) < weeks:
            for kind in optimised_point_by_kind:
                optimised_point_by_kind[kind].append(float("nan"))
            continue
        rf_returns = _rf_returns_for_dates(universe, history)
        optimised_dates.append(date.fromisoformat(point.iso_date))
        for kind in optimised_point_by_kind:
            value = _metric_value(kind, frozen_returns, rf_returns)
            optimised_point_by_kind[kind].append(value)
            optimised_line_by_kind[kind].append(value)

    specs = [
        ("sharpe", "Sharpe Ratio", "Sharpe ratio", False),
        ("cagr", "CAGR", "CAGR", True),
        ("vol", "Vol estimate", "Annualised volatility", True),
    ]
    charts: list[RollingMetricChart] = []
    for kind, title, ylabel, percent in specs:
        charts.append(
            RollingMetricChart(
                slug=kind,
                title=title,
                ylabel=ylabel,
                percent=percent,
                us500=_rolling_line_from_returns(
                    bench_dates,
                    bench_series,
                    bench_rf,
                    kind=kind,
                    weeks=weeks,
                ),
                optimised=MetricLineSeries(
                    optimised_dates,
                    optimised_line_by_kind[kind],
                ),
                backtest=_rolling_line_from_returns(
                    point_dates,
                    strat_returns,
                    strat_rf,
                    kind=kind,
                    weeks=weeks,
                ),
                live=MetricLineSeries([], []),
                backtest_scatter=_lookback_scatter_from_returns(
                    optimised_point_by_kind[kind],
                    strat_returns,
                    strat_rf,
                    point_dates,
                    kind=kind,
                    weeks=weeks,
                ),
                live_scatter=MetricScatterSeries([], [], []),
            )
        )
    return charts


def market_label(universe: Universe, market_id: str) -> str:
    asset = universe.assets.get(market_id)
    if asset is None:
        return market_id
    ticker = (asset.yahoo_ticker or market_id).strip()
    name = (asset.name or ticker).strip()
    if len(name) > 40:
        name = name[:37] + "..."
    return f"{ticker} — {name}" if ticker != market_id else name


def _weight_change_since(
    current: float,
    *,
    past_point: WeekPointLike | None,
    market_id: str,
) -> float | None:
    if past_point is None:
        return None
    past = (
        past_point.cash_weight
        if market_id == "__cash__"
        else past_point.effective_weights.get(market_id, 0.0)
    )
    if current <= 1e-6 and past <= 1e-6:
        return None
    return current - past


def allocation_rows(
    universe: Universe,
    point: WeekPointLike,
    *,
    yahoo_dir: Path,
    spark_dir: Path,
    as_of: date,
    ie_weights_by_market_id: dict[str, float] | None = None,
    ie_icons_by_market_id: dict[str, str] | None = None,
    point_1m_ago: WeekPointLike | None = None,
    point_1y_ago: WeekPointLike | None = None,
) -> list[AllocationRow]:
    spark_dir.mkdir(parents=True, exist_ok=True)
    ie_weights = ie_weights_by_market_id or {}
    ie_icons = ie_icons_by_market_id or {}
    rows: list[AllocationRow] = []
    weights = sorted(
        point.effective_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    for market_id, weight in weights:
        if weight <= 1e-6:
            continue
        prices = yahoo_close_prices_last_year(market_id, yahoo_dir, as_of=as_of)
        spark_name = f"{market_id}.png"
        rows.append(
            AllocationRow(
                market_id=market_id,
                label=market_label(universe, market_id),
                weight_pct=weight,
                spark_path=f"sparklines/{spark_name}",
                return_1y=total_return_from_prices(prices),
                ie_weight_pct=ie_weights.get(market_id),
                icon_path=ie_icons.get(market_id, ""),
                weight_change_1m=_weight_change_since(
                    weight,
                    past_point=point_1m_ago,
                    market_id=market_id,
                ),
                weight_change_1y=_weight_change_since(
                    weight,
                    past_point=point_1y_ago,
                    market_id=market_id,
                ),
            )
        )
    if point.cash_weight > 1e-6:
        rows.append(
            AllocationRow(
                market_id="__cash__",
                label="Cash",
                weight_pct=point.cash_weight,
                spark_path="",
                return_1y=None,
                weight_change_1m=_weight_change_since(
                    point.cash_weight,
                    past_point=point_1m_ago,
                    market_id="__cash__",
                ),
                weight_change_1y=_weight_change_since(
                    point.cash_weight,
                    past_point=point_1y_ago,
                    market_id="__cash__",
                ),
            )
        )
    return rows
