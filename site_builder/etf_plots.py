"""Matplotlib plots for the ETF static site."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from matplotlib.ticker import PercentFormatter

from site_builder.metrics import (
    ReturnDistributionStats,
    RollingMetricChart,
    drawdown_exceedance_curve,
    drawdown_series,
    fraction_same_or_worse,
)
from site_builder.etf_data import WeekPointLike, rebased_equity, tracking_anchor_index
from site_builder.plots import (
    RRD_BLACK,
    RRD_BLUE,
    RRD_GRID,
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
    ax.set_ylabel("Equity (rebased to 1.0 at tracking start)")
    ax.set_title("Equity")
    _rrd_legend(ax, loc="upper left")
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
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
