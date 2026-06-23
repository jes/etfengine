"""Matplotlib plots for the ETF static site."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter

from site_builder.metrics import (
    ReturnDistributionStats,
    RollingMetricChart,
    ath_exceedance_curve,
    drawdown_exceedance_curve,
    drawdown_series,
    fraction_at_least,
    fraction_same_or_worse,
)
from site_builder.etf_data import (
    RegimeReturnSeries,
    WeekPointLike,
    rebased_equity,
    regime_unanimous_bearish_spans,
    tracking_anchor_index,
)
from site_builder.plots import (
    RRD_BLACK,
    RRD_BLUE,
    RRD_GRID,
    RRD_GREEN,
    RRD_ORANGE,
    RRD_PIXEL,
    _apply_rrd_axis,
    _rrd_legend,
    _rrd_plot,
    _rrd_subplots,
    _save_rrd,
    plot_sparkline,
    plot_weekly_return_histogram,
)


class VolCapCurveLike(Protocol):
    vol_cap: float
    equity: list[float]
    sharpe: float
    cagr: float
    vol_ann: float
    max_drawdown: float


def _format_vol_cap_label(curve: VolCapCurveLike) -> str:
    cap_pct = curve.vol_cap * 100.0
    cap_label = f"{cap_pct:.0f}%" if abs(cap_pct - round(cap_pct)) < 1e-9 else f"{cap_pct:.1f}%"
    return (
        f"{cap_label} cap  "
        f"Sharpe {curve.sharpe:.2f}  "
        f"CAGR {curve.cagr * 100:.1f}%  "
        f"vol {curve.vol_ann * 100:.1f}%  "
        f"max DD {curve.max_drawdown * 100:.1f}%"
    )


def _equity_log_tick(value: float, _position: int) -> str:
    if value <= 0:
        return ""
    return f"{value:.1f}"


def plot_etf_equity(
    *,
    trade_dates: list[str],
    strat_equity: list[float],
    bench_equity: list[float],
    bench_label: str,
    tracking_start: str,
    output: Path,
) -> None:
    fig, ax = _rrd_subplots(1440, 600)
    _apply_rrd_axis(ax)

    anchor = tracking_anchor_index(trade_dates, tracking_start)
    anchor_date = date.fromisoformat(trade_dates[anchor])
    dates = [date.fromisoformat(iso_date) for iso_date in trade_dates]

    strat_values = rebased_equity(strat_equity, anchor)
    bench_values = rebased_equity(bench_equity, anchor)

    _rrd_plot(ax, dates, strat_values, color=RRD_BLUE, label="Strategy")
    _rrd_plot(
        ax,
        dates,
        bench_values,
        color=RRD_BLACK,
        alpha=0.85,
        label=bench_label,
    )
    ax.axvline(
        anchor_date,
        color=RRD_BLACK,
        linestyle="--",
        linewidth=RRD_PIXEL,
        label="Tracking start",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_equity_log_tick))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Equity (log scale, rebased to 1.0 at tracking start)")
    ax.set_title("Equity")
    _rrd_legend(ax, loc="upper left")
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    _save_rrd(fig, output)


def plot_etf_equity_since_tracking(
    *,
    trade_dates: list[str],
    strat_equity: list[float],
    bench_equity: list[float],
    bench_label: str,
    tracking_start: str,
    output: Path,
) -> None:
    anchor = tracking_anchor_index(trade_dates, tracking_start)
    if anchor >= len(trade_dates):
        raise ValueError(f"tracking start {tracking_start!r} is after last trade date")

    dates = [date.fromisoformat(iso_date) for iso_date in trade_dates[anchor:]]
    strat_values = rebased_equity(strat_equity, anchor)[anchor:]
    bench_values = rebased_equity(bench_equity, anchor)[anchor:]

    fig, ax = _rrd_subplots(1440, 600)
    _apply_rrd_axis(ax)

    _rrd_plot(ax, dates, strat_values, color=RRD_BLUE, label="Strategy")
    _rrd_plot(
        ax,
        dates,
        bench_values,
        color=RRD_BLACK,
        alpha=0.85,
        label=bench_label,
    )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Equity (rebased to 1.0 at tracking start)")
    ax.set_title("Equity since tracking start")
    _rrd_legend(ax, loc="upper left")
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    _save_rrd(fig, output)


def plot_etf_vol_cap_equity(
    *,
    trade_dates: list[str],
    curves: list[VolCapCurveLike],
    tracking_start: str,
    highlight_vol_cap: float | None = None,
    output: Path,
) -> None:
    if not curves:
        raise ValueError("no vol-cap curves to plot")

    fig, ax = _rrd_subplots(1440, 720)
    _apply_rrd_axis(ax)

    anchor = tracking_anchor_index(trade_dates, tracking_start)
    anchor_date = date.fromisoformat(trade_dates[anchor])
    dates = [date.fromisoformat(iso_date) for iso_date in trade_dates]
    cmap = plt.get_cmap("viridis")
    n = len(curves)

    for index, curve in enumerate(curves):
        if len(curve.equity) != len(trade_dates):
            raise ValueError(
                f"plot length mismatch for vol cap {curve.vol_cap}: "
                f"{len(trade_dates)} dates, {len(curve.equity)} equity points"
            )
        color = cmap(index / max(n - 1, 1))
        highlighted = (
            highlight_vol_cap is not None
            and abs(curve.vol_cap - highlight_vol_cap) < 1e-9
        )
        _rrd_plot(
            ax,
            dates,
            rebased_equity(curve.equity, anchor),
            color=color,
            label=_format_vol_cap_label(curve),
            linewidth=2.5 * RRD_PIXEL if highlighted else RRD_PIXEL,
            alpha=1.0 if highlighted else 0.9,
            zorder=3 if highlighted else 2,
        )

    ax.axvline(
        anchor_date,
        color=RRD_BLACK,
        linestyle="--",
        linewidth=RRD_PIXEL,
        label="Tracking start",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_equity_log_tick))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Equity (log scale, rebased to 1.0 at tracking start)")
    ax.set_title("Equity by vol cap")
    _rrd_legend(ax, loc="upper left", fontsize=7)
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.22)
    _save_rrd(fig, output)


def plot_etf_drawdown(
    *,
    trade_dates: list[str],
    strat_equity: list[float],
    tracking_start: str,
    output: Path,
) -> None:
    fig, ax = _rrd_subplots(1440, 480)
    _apply_rrd_axis(ax)

    anchor = tracking_anchor_index(trade_dates, tracking_start)
    dates = [date.fromisoformat(iso_date) for iso_date in trade_dates]
    equities = rebased_equity(strat_equity, anchor)
    drawdowns = drawdown_series(equities)
    _rrd_plot(ax, dates, drawdowns, color=RRD_BLUE, label="Strategy")
    ax.axvline(
        date.fromisoformat(tracking_start),
        color=RRD_BLACK,
        linestyle="--",
        linewidth=RRD_PIXEL,
        label="Tracking start",
    )
    ax.set_ylabel("Drawdown from prior high")
    ax.set_title("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(top=0.0)
    _rrd_legend(ax, loc="lower left")
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.24)
    _save_rrd(fig, output)


def plot_invested_weight(
    *,
    points: list[WeekPointLike],
    tracking_start: str,
    output: Path,
) -> None:
    fig, ax = _rrd_subplots(1440, 420)
    _apply_rrd_axis(ax)
    dates = [date.fromisoformat(point.iso_date) for point in points]
    invested = [point.invested_weight for point in points]
    _rrd_plot(ax, dates, invested, color=RRD_BLUE, label="Invested weight")
    ax.axvline(
        date.fromisoformat(tracking_start),
        color=RRD_BLACK,
        linestyle="--",
        linewidth=RRD_PIXEL,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Invested fraction")
    ax.set_title("Invested weight")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.24)
    _save_rrd(fig, output)


def plot_etf_rolling_metric_chart(
    chart: RollingMetricChart,
    output: Path,
    *,
    tracking_start: str,
    bench_label: str = "VWRP prior 1y",
) -> None:
    from site_builder.plots import plot_rolling_metric_chart

    plot_rolling_metric_chart(
        chart,
        output,
        first_live=f"{tracking_start}T00:00:00Z",
        bench_label=bench_label,
        optimised_label="Optimiser pick prior 1y",
        backtest_label="Strategy actual prior 1y",
        live_label="Live actual prior 1y",
        scatter_title="Optimiser pick vs strategy actual prior 1y",
        scatter_y_label=f"Strategy actual prior {chart.ylabel}",
    )


def plot_backtest_return_histogram(
    *,
    returns: list[float],
    stats: ReturnDistributionStats | None,
    output: Path,
) -> None:
    plot_weekly_return_histogram(
        backtest_returns=returns,
        live_returns=[],
        backtest_stats=stats,
        live_stats=None,
        output=output,
    )


def plot_backtest_drawdown_distribution(
    *,
    drawdowns: list[float],
    output: Path,
    current_drawdown: float | None = None,
) -> None:
    fig, ax = _rrd_subplots(1200, 600)
    _apply_rrd_axis(ax)
    if not drawdowns:
        ax.text(0.5, 0.5, "No drawdown data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        xs, ys = drawdown_exceedance_curve(drawdowns)
        _rrd_plot(ax, xs, ys, color=RRD_BLUE, label="Strategy")
        marker_dd = current_drawdown if current_drawdown is not None else drawdowns[-1]
        if marker_dd == marker_dd:
            marker_y = fraction_same_or_worse(marker_dd, drawdowns)
            if marker_y is not None:
                ax.scatter(
                    [marker_dd],
                    [marker_y],
                    s=36,
                    color=RRD_ORANGE,
                    edgecolors=RRD_BLACK,
                    linewidths=RRD_PIXEL,
                    zorder=5,
                    label="Current",
                )
        ax.set_xlabel("Drawdown level")
        ax.set_ylabel("% of weeks at this drawdown or worse")
        ax.set_title("Drawdown distribution")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(True, color=RRD_GRID, linewidth=RRD_PIXEL, axis="y")
        _rrd_legend(ax, loc="lower right")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.18)
    _save_rrd(fig, output)


def plot_backtest_ath_distribution(
    *,
    days_since_ath: list[int],
    output: Path,
    current_days_since_ath: int | None = None,
) -> None:
    fig, ax = _rrd_subplots(1200, 600)
    _apply_rrd_axis(ax)
    if not days_since_ath:
        ax.text(0.5, 0.5, "No ATH data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        xs, ys = ath_exceedance_curve(days_since_ath)
        _rrd_plot(ax, xs, ys, color=RRD_BLUE, label="Strategy")
        marker_days = (
            current_days_since_ath
            if current_days_since_ath is not None
            else days_since_ath[-1]
        )
        floats = [float(value) for value in days_since_ath]
        marker_y = fraction_at_least(float(marker_days), floats)
        if marker_y is not None:
            ax.scatter(
                [float(marker_days)],
                [marker_y],
                s=36,
                color=RRD_ORANGE,
                edgecolors=RRD_BLACK,
                linewidths=RRD_PIXEL,
                zorder=5,
                label="Current",
            )
        ax.set_xlabel("Days since ATH")
        ax.set_ylabel("% of weeks at this level or worse")
        ax.set_title("Days since ATH distribution")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
        ax.grid(True, color=RRD_GRID, linewidth=RRD_PIXEL, axis="y")
        _rrd_legend(ax, loc="lower right")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.18)
    _save_rrd(fig, output)


REGIME_BEARISH_SHADE = "#dd9a9a"


def plot_regime_returns(
    *,
    series: list[RegimeReturnSeries],
    tracking_start: str,
    output: Path,
) -> None:
    fig, ax = _rrd_subplots(1440, 576)
    _apply_rrd_axis(ax)

    for x0, x1 in regime_unanimous_bearish_spans(series):
        ax.axvspan(
            x0,
            x1,
            color=REGIME_BEARISH_SHADE,
            alpha=0.35,
            zorder=0,
            linewidth=0,
        )

    colors = [RRD_BLUE, RRD_ORANGE, RRD_GREEN]
    plotted = False
    for index, item in enumerate(series):
        if not item.dates:
            continue
        plotted = True
        _rrd_plot(
            ax,
            item.dates,
            item.values,
            color=colors[index % len(colors)],
            label=f"{item.months}m",
        )

    if not plotted:
        ax.text(
            0.5,
            0.5,
            "Not enough regime history",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    anchor_date = date.fromisoformat(tracking_start)
    ax.axvline(
        anchor_date,
        color=RRD_BLACK,
        linestyle="--",
        linewidth=RRD_PIXEL,
        label="Tracking start",
    )
    ax.axhline(0.0, color=RRD_BLACK, linewidth=RRD_PIXEL, alpha=0.55)
    ax.set_ylabel("Trailing return")
    ax.set_title("Regime trailing returns")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _rrd_legend(ax, loc="best")
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.22)
    _save_rrd(fig, output)


def write_sparklines(
    allocations: list,
    *,
    yahoo_dir: Path,
    spark_dir: Path,
    as_of: date,
) -> None:
    spark_dir.mkdir(parents=True, exist_ok=True)
    for row in allocations:
        if not row.spark_path or row.market_id == "__cash__":
            continue
        from site_builder.metrics import yahoo_close_prices_last_year

        prices = yahoo_close_prices_last_year(row.market_id, yahoo_dir, as_of=as_of)
        plot_sparkline(prices, spark_dir / f"{row.market_id}.png")
