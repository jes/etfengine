"""Matplotlib PNG plots for the ETF static site."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from site_builder.metrics import (
    MetricScatterSeries,
    ReturnDistributionStats,
    RollingMetricChart,
    parse_timestamp,
)

RRD_DPI = 120
RRD_PIXEL = 72 / RRD_DPI
RRD_BLUE = "#0000cc"
RRD_ORANGE = "#ff7f00"
RRD_GREEN = "#00a000"
RRD_BLACK = "#000000"
RRD_GRID = "#d4d4d4"
RRD_FACE = "#f7f7f7"
RRD_EDGE = "#808080"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 8,
        "lines.antialiased": True,
        "patch.antialiased": True,
        "text.antialiased": True,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def _rrd_subplots(width_px: int, height_px: int, *args, **kwargs):
    fig, axes = plt.subplots(
        *args,
        figsize=(width_px / RRD_DPI, height_px / RRD_DPI),
        dpi=RRD_DPI,
        **kwargs,
    )
    fig.patch.set_facecolor("white")
    return fig, axes


def _apply_rrd_axis(ax) -> None:
    ax.set_facecolor(RRD_FACE)
    ax.grid(True, color=RRD_GRID, linewidth=RRD_PIXEL)
    ax.tick_params(colors="#222222", labelsize=8, width=RRD_PIXEL, length=3)
    for spine in ax.spines.values():
        spine.set_color(RRD_EDGE)
        spine.set_linewidth(RRD_PIXEL)


def _apply_rrd_axes(*axes) -> None:
    for ax in axes:
        _apply_rrd_axis(ax)


def _rrd_plot(ax, *args, **kwargs):
    kwargs.setdefault("linewidth", RRD_PIXEL)
    kwargs.setdefault("antialiased", True)
    kwargs.setdefault("solid_capstyle", "butt")
    kwargs.setdefault("solid_joinstyle", "miter")
    return ax.plot(*args, **kwargs)


def _rrd_legend(ax, *args, **kwargs):
    kwargs.setdefault("fancybox", False)
    legend = ax.legend(*args, **kwargs)
    frame = legend.get_frame()
    frame.set_facecolor(RRD_FACE)
    frame.set_edgecolor(RRD_EDGE)
    frame.set_linewidth(RRD_PIXEL)
    frame.set_alpha(1.0)
    return legend


def _save_rrd(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=RRD_DPI,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=10 / RRD_DPI,
    )
    plt.close(fig)


def _format_stats_block(label: str, stats: ReturnDistributionStats) -> str:
    skew = f"{stats.skew:+.2f}" if stats.skew == stats.skew else "n/a"
    return (
        f"{label} (n={stats.count}): "
        f"μ={stats.mean:+.2%}  σ={stats.stdev:.2%}  skew={skew}  "
        f"ann vol={stats.ann_vol:.1%}  "
        f"min={stats.min_return:+.2%}  max={stats.max_return:+.2%}"
    )


def plot_weekly_return_histogram(
    *,
    backtest_returns: list[float],
    live_returns: list[float],
    backtest_stats: ReturnDistributionStats | None,
    live_stats: ReturnDistributionStats | None,
    output: Path,
) -> None:
    fig, ax = _rrd_subplots(1200, 600)
    _apply_rrd_axis(ax)
    combined = [*backtest_returns, *live_returns]
    if not combined:
        ax.text(0.5, 0.5, "No weekly returns", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        low = min(combined)
        high = max(combined)
        padding = max((high - low) * 0.05, 0.001)
        bins = np.linspace(low - padding, high + padding, 30)
        if backtest_returns:
            ax.hist(
                backtest_returns,
                bins=bins,
                density=True,
                alpha=0.62,
                color=RRD_BLUE,
                edgecolor=RRD_BLUE,
                linewidth=RRD_PIXEL,
                label="Backtest",
            )
        if live_returns:
            ax.hist(
                live_returns,
                bins=bins,
                density=True,
                alpha=0.62,
                color=RRD_ORANGE,
                edgecolor=RRD_ORANGE,
                linewidth=RRD_PIXEL,
                label="Live",
            )
        ax.axvline(0.0, color=RRD_BLACK, linewidth=RRD_PIXEL, alpha=0.55)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xlabel("Weekly return (%)")
        ax.set_ylabel("Density")
        ax.set_title("Weekly return distribution")
        ax.grid(True, color=RRD_GRID, linewidth=RRD_PIXEL, axis="y")
        _rrd_legend(ax, loc="upper right")

        stat_lines: list[str] = []
        if backtest_stats is not None:
            stat_lines.append(_format_stats_block("Backtest", backtest_stats))
        if live_stats is not None:
            stat_lines.append(_format_stats_block("Live", live_stats))
        if stat_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(stat_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                family="monospace",
                bbox={
                    "boxstyle": "square,pad=0.35",
                    "facecolor": RRD_FACE,
                    "edgecolor": RRD_EDGE,
                    "linewidth": RRD_PIXEL,
                    "alpha": 1.0,
                },
            )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16)
    _save_rrd(fig, output)


def _plot_metric_line(
    ax,
    series,
    *,
    label: str,
    color: str,
    linewidth: float = RRD_PIXEL,
    linestyle: str = "-",
) -> None:
    if series.dates and series.values:
        _rrd_plot(
            ax,
            series.dates,
            series.values,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )


def _scatter_fit_line(
    x_values: list[float],
    y_values: list[float],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or float(np.ptp(x)) <= 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    if total_sum <= 0:
        return None
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    y_line = slope * x_line + intercept
    return x_line, y_line, 1.0 - residual_sum / total_sum


def _plot_scatter_fit(
    ax,
    x_values: list[float],
    y_values: list[float],
    *,
    label: str,
    color: str,
) -> str | None:
    fit = _scatter_fit_line(x_values, y_values)
    if fit is None:
        return None
    x_line, y_line, r_squared = fit
    _rrd_plot(
        ax,
        x_line,
        y_line,
        color=color,
        linewidth=RRD_PIXEL,
        linestyle="--",
        alpha=0.9,
        label=f"{label} fit",
    )
    return f"{label} R^2={r_squared:.2f}"


def _scatter_date_norm(*series: MetricScatterSeries) -> Normalize | None:
    ordinals = [
        point_date.toordinal()
        for scatter in series
        for point_date in scatter.dates
    ]
    if not ordinals:
        return None
    return Normalize(vmin=min(ordinals), vmax=max(ordinals))


def _plot_date_colored_scatter(
    ax,
    scatter: MetricScatterSeries,
    *,
    norm: Normalize,
    marker: str,
    size: float,
    alpha: float,
) -> None:
    if not scatter.x:
        return
    ordinals = [point_date.toordinal() for point_date in scatter.dates]
    ax.scatter(
        scatter.x,
        scatter.y,
        s=size,
        alpha=alpha,
        c=ordinals,
        cmap="viridis",
        norm=norm,
        marker=marker,
        edgecolors=RRD_BLACK,
        linewidths=RRD_PIXEL * 0.5,
    )


def _scatter_marker_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=RRD_EDGE,
            markeredgecolor=RRD_BLACK,
            markeredgewidth=RRD_PIXEL,
            markersize=5,
            label="Backtest",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor=RRD_EDGE,
            markeredgecolor=RRD_BLACK,
            markeredgewidth=RRD_PIXEL,
            markersize=5,
            label="Live",
        ),
    ]


def plot_rolling_metric_chart(
    chart: RollingMetricChart,
    output: Path,
    *,
    first_live: str | None = None,
    bench_label: str = "VWRP prior 1y",
    optimised_label: str = "Optimiser pick prior 1y",
    backtest_label: str = "Strategy actual prior 1y",
    live_label: str = "Live actual prior 1y",
    scatter_title: str = "Optimiser pick vs strategy actual prior 1y",
    scatter_y_label: str | None = None,
) -> None:
    fig, (line_ax, scatter_ax) = _rrd_subplots(
        1680,
        600,
        1,
        2,
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    _apply_rrd_axes(line_ax, scatter_ax)

    _plot_metric_line(line_ax, chart.us500, label=bench_label, color=RRD_BLACK)
    _plot_metric_line(
        line_ax,
        chart.optimised,
        label=optimised_label,
        color=RRD_GREEN,
        linewidth=RRD_PIXEL,
    )
    _plot_metric_line(
        line_ax,
        chart.backtest,
        label=backtest_label,
        color=RRD_BLUE,
        linewidth=RRD_PIXEL,
    )
    _plot_metric_line(
        line_ax,
        chart.live,
        label=live_label,
        color=RRD_ORANGE,
        linewidth=RRD_PIXEL,
    )
    if first_live:
        line_ax.axvline(
            parse_timestamp(first_live).date(),
            color=RRD_BLACK,
            linestyle="--",
            linewidth=RRD_PIXEL,
            label="Tracking start",
        )
    line_ax.set_title(chart.title)
    line_ax.set_ylabel(chart.ylabel)
    _rrd_legend(line_ax, loc="best")
    line_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if chart.percent:
        line_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    scatter_date_norm = _scatter_date_norm(
        chart.backtest_scatter,
        chart.live_scatter,
    )
    if scatter_date_norm is not None:
        _plot_date_colored_scatter(
            scatter_ax,
            chart.backtest_scatter,
            norm=scatter_date_norm,
            marker="o",
            size=12,
            alpha=0.85,
        )
        _plot_date_colored_scatter(
            scatter_ax,
            chart.live_scatter,
            norm=scatter_date_norm,
            marker="^",
            size=18,
            alpha=0.9,
        )
        colorbar = fig.colorbar(
            ScalarMappable(cmap="viridis", norm=scatter_date_norm),
            ax=scatter_ax,
            fraction=0.08,
            pad=0.02,
        )
        colorbar.set_label("Date")
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position: date.fromordinal(int(round(value))).strftime(
                    "%Y"
                )
            )
        )
        colorbar.ax.tick_params(labelsize=8, width=RRD_PIXEL, length=3)
        colorbar.outline.set_edgecolor(RRD_EDGE)
        colorbar.outline.set_linewidth(RRD_PIXEL)
    fit_labels: list[str] = []
    backtest_fit = _plot_scatter_fit(
        scatter_ax,
        chart.backtest_scatter.x,
        chart.backtest_scatter.y,
        label="Backtest",
        color=RRD_BLUE,
    )
    if backtest_fit is not None:
        fit_labels.append(backtest_fit)
    live_fit = _plot_scatter_fit(
        scatter_ax,
        chart.live_scatter.x,
        chart.live_scatter.y,
        label="Live",
        color=RRD_ORANGE,
    )
    if live_fit is not None:
        fit_labels.append(live_fit)
    if fit_labels:
        scatter_ax.text(
            0.02,
            0.98,
            "\n".join(fit_labels),
            transform=scatter_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "square,pad=0.3",
                "facecolor": RRD_FACE,
                "edgecolor": RRD_EDGE,
                "alpha": 1.0,
            },
        )
    if not chart.backtest_scatter.x and not chart.live_scatter.x:
        scatter_ax.text(
            0.5,
            0.5,
            "No lookback comparison points",
            ha="center",
            va="center",
            transform=scatter_ax.transAxes,
        )
    scatter_ax.axhline(0, color=RRD_BLACK, linewidth=RRD_PIXEL, alpha=0.45)
    scatter_ax.axvline(0, color=RRD_BLACK, linewidth=RRD_PIXEL, alpha=0.45)
    scatter_ax.set_title(scatter_title)
    scatter_ax.set_xlabel(f"Optimised prior {chart.ylabel}")
    scatter_y = scatter_y_label or f"Strategy actual prior {chart.ylabel}"
    scatter_ax.set_ylabel(scatter_y)
    if chart.percent:
        scatter_ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        scatter_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    marker_handles = _scatter_marker_legend_handles()
    scatter_handles: list[Line2D] = []
    if chart.backtest_scatter.x:
        scatter_handles.append(marker_handles[0])
    if chart.live_scatter.x:
        scatter_handles.append(marker_handles[1])
    fit_handles, fit_labels_legend = scatter_ax.get_legend_handles_labels()
    if scatter_handles or fit_handles:
        _rrd_legend(
            scatter_ax,
            handles=scatter_handles + fit_handles,
            labels=[handle.get_label() for handle in scatter_handles] + fit_labels_legend,
            loc="best",
        )

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.18, wspace=0.24)
    _save_rrd(fig, output)


def plot_sparkline(values: list[float], output: Path) -> None:
    fig, ax = _rrd_subplots(176, 40)
    if values:
        _rrd_plot(ax, values, color=RRD_BLUE)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=RRD_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
