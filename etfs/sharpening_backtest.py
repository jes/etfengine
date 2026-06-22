#!/usr/bin/env python3
"""ETF walk-forward backtest: sparse ETF allocation for manual GIA rebalances."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ETFS_DIR = Path(__file__).resolve().parent
if str(ETFS_DIR) not in sys.path:
    sys.path.insert(0, str(ETFS_DIR))

from strategy.constants import RISK_FREE_ID
from strategy.costs import (
    apply_rebalance_drift_band,
    rebalance_spread_drag,
    return_after_spread_drag,
)
from strategy.data import Universe, allocatable_assets, load_universe
from strategy.optimizer import build_weekly_end_dates, window_start_from_end
from strategy.weights import ewma_smooth_capped_weight_rows, target_weights_for_date

from build_universe import (
    DEFAULT_INVESTENGINE_ALLOWLIST,
    clean_token,
    load_identifier_allowlist,
)
from sharpening_optimizer import (
    DEFAULT_MIN_WEIGHT,
    REBALANCE_FREQUENCIES,
    build_etf_weight_schedule,
    default_ewma_span,
    select_rebalance_end_dates,
)

DEFAULT_MARKETS = ETFS_DIR / "markets.csv"
DEFAULT_MARKETS_STATS_ALLOWLIST = ETFS_DIR / "output" / "markets_stats_allowlist.csv"
DEFAULT_YAHOO = ETFS_DIR / "yahoo"
BENCHMARK_ID = "ie00bk5bqt80"  # VWRP.L, accumulating FTSE All-World (IE ISA)
DEFAULT_OUTPUT = ETFS_DIR / "output" / "sharpening_equity.png"
DEFAULT_DIAGNOSTICS = ETFS_DIR / "output" / "sharpening_weekly_diagnostics.csv"
DEFAULT_ALLOWLIST = ETFS_DIR / "output" / "market_stats.csv"


def resolve_markets_csv(path: Path | None = None) -> Path:
    if path is not None:
        return path
    if DEFAULT_MARKETS_STATS_ALLOWLIST.is_file():
        return DEFAULT_MARKETS_STATS_ALLOWLIST
    return DEFAULT_MARKETS


def investengine_market_ids(
    markets_csv: Path,
    *,
    allowlist_path: Path = DEFAULT_INVESTENGINE_ALLOWLIST,
) -> set[str]:
    """Map InvestEngine ISIN/ticker allowlist to lowercase market ids from the manifest."""
    if not allowlist_path.is_file():
        raise SystemExit(f"InvestEngine allowlist not found: {allowlist_path}")
    allowlist = load_identifier_allowlist(allowlist_path)
    ids: set[str] = set()
    with markets_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            market_id = row.get("id", "").strip().lower()
            if not market_id:
                continue
            ticker = clean_token((row.get("yahoo_ticker") or "").removesuffix(".L"))
            if clean_token(market_id) in allowlist or ticker in allowlist:
                ids.add(market_id)
    return ids


@dataclass(frozen=True)
class EquityCurve:
    label: str
    equity: list[float]
    stats: tuple[float, float, float, float]  # mean, vol, sharpe, cagr


@dataclass(frozen=True)
class VolCapBacktestCurve:
    vol_cap: float
    equity: list[float]
    mean_ann: float
    vol_ann: float
    sharpe: float
    cagr: float
    max_drawdown: float


def format_stats_label(label: str, stats: tuple[float, float, float, float]) -> str:
    _, vol, sharpe, cagr = stats
    return f"{label}  CAGR {cagr * 100:.1f}%  vol {vol * 100:.1f}%  Sharpe {sharpe:.2f}"


def format_vol_cap_label(curve: VolCapBacktestCurve) -> str:
    cap_pct = curve.vol_cap * 100.0
    cap_label = f"{cap_pct:.0f}%" if abs(cap_pct - round(cap_pct)) < 1e-9 else f"{cap_pct:.1f}%"
    return (
        f"{cap_label} cap  "
        f"Sharpe {curve.sharpe:.2f}  "
        f"CAGR {curve.cagr * 100:.1f}%  "
        f"vol {curve.vol_ann * 100:.1f}%  "
        f"max DD {curve.max_drawdown * 100:.1f}%"
    )


def max_drawdown_from_equity(equity: list[float]) -> float:
    if not equity:
        return float("nan")
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def print_stats_table(rows: list[tuple[str, tuple[float, float, float, float]]]) -> None:
    print(f"{'':22} {'Mean ann.':>10} {'Vol ann.':>10} {'Sharpe':>8} {'CAGR':>10}")
    for name, stats in rows:
        mean_ann, vol_ann, sharpe, cagr = stats
        print(
            f"{name:22} "
            f"{mean_ann * 100:9.2f}% "
            f"{vol_ann * 100:9.2f}% "
            f"{sharpe:8.3f} "
            f"{cagr * 100:9.2f}%"
        )


def build_schedule_and_run(
    universe: Universe,
    tickers: list[str],
    *,
    backtest_start: str,
    end: str,
    lookback_months: int,
    target_vol: float,
    max_holdings: int,
    min_weight: float,
    min_coverage: float,
    listing_years: float,
    max_abs_daily_return: float,
    ewma_span: int,
    rebalance_frequency: str,
    drift_band: float,
) -> tuple[list[str], list[WeekPoint], tuple[float, float, float, float]]:
    schedule_start = window_start_from_end(backtest_start, months=lookback_months)
    end_dates, raw_rows = build_etf_weight_schedule(
        universe,
        tickers,
        lookback_months=lookback_months,
        vol_cap=target_vol,
        max_holdings=max_holdings,
        min_weight=min_weight,
        min_coverage_fraction=min_coverage,
        listing_years=listing_years,
        max_abs_daily_return=max_abs_daily_return,
        schedule_start=schedule_start,
        schedule_end=end,
        rebalance_frequency=rebalance_frequency,
    )
    smoothed_rows = ewma_smooth_capped_weight_rows(
        raw_rows,
        span=ewma_span,
        min_weight=min_weight,
    )
    smoothed_rows = scale_weight_rows_to_vol_target(
        universe,
        end_dates,
        smoothed_rows,
        lookback_months=lookback_months,
        target_vol=target_vol,
        min_weight=min_weight,
    )
    trade_dates = [
        d
        for d in universe.weekly_dates
        if d >= backtest_start and end_dates and d > end_dates[0]
    ]
    points, _ = run_weekly_backtest(
        universe,
        trade_dates,
        end_dates,
        smoothed_rows,
        drift_band=drift_band,
    )
    trade_dates = [p.iso_date for p in points]
    rf_returns = [
        universe.assets[RISK_FREE_ID].returns_by_date[d] for d in trade_dates
    ]
    stats = stats_from_weekly([p.weekly_return for p in points], rf_returns)
    return trade_dates, points, stats


@dataclass(frozen=True)
class WeekPoint:
    iso_date: str
    equity: float
    weekly_return: float
    invested_weight: float
    cash_weight: float
    spread_drag: float
    net_weekly: float
    holdings: str
    effective_weights: dict[str, float]
    target_weights: dict[str, float]


def load_allowlist_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["id"].strip().lower()
            for row in csv.DictReader(handle)
            if row.get("id")
        }


def load_dividend_policy_ids(path: Path, policy: str) -> set[str] | None:
    if policy == "any":
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "dividends" not in (reader.fieldnames or []):
            raise SystemExit(
                f"{path} does not include a dividends column; regenerate with etfs/build_universe.py"
            )
        return {
            row["id"].strip().lower()
            for row in reader
            if row.get("id") and (row.get("dividends") or "").strip().lower() == policy
        }


def combine_allowed_ids(*allowlists: set[str] | None) -> set[str] | None:
    combined: set[str] | None = None
    for allowlist in allowlists:
        if allowlist is None:
            continue
        combined = set(allowlist) if combined is None else combined & allowlist
    return combined


def portfolio_vol_ann(
    weights: dict[str, float],
    universe: Universe,
    dates: list[str],
) -> float:
    if len(dates) < 2:
        return float("nan")
    weekly = np.array(
        [
            portfolio_weekly_return(weights, universe, iso_date)
            for iso_date in dates
        ],
        dtype=float,
    )
    return float(weekly.std(ddof=1)) * math.sqrt(52)


def floor_capped_weights(
    weights: dict[str, float],
    *,
    min_weight: float,
) -> dict[str, float]:
    return {
        market_id: weight
        for market_id, weight in weights.items()
        if weight >= min_weight and weight > 1e-12
    }


def scale_row_to_vol_target(
    universe: Universe,
    end_date: str,
    weights: dict[str, float],
    *,
    lookback_months: int,
    target_vol: float,
    min_weight: float,
) -> dict[str, float]:
    scaled = floor_capped_weights(weights, min_weight=min_weight)
    if not scaled or target_vol <= 0:
        return scaled

    start_date = window_start_from_end(end_date, months=lookback_months)
    dates = [d for d in universe.weekly_dates if start_date <= d <= end_date]
    for _ in range(len(scaled) + 1):
        vol = portfolio_vol_ann(scaled, universe, dates)
        total = sum(scaled.values())
        if not vol or not math.isfinite(vol) or total <= 1e-12:
            return scaled
        scale = target_vol / vol
        scale = min(scale, 1.0 / total)
        next_scaled = {
            market_id: weight * scale for market_id, weight in scaled.items()
        }
        next_scaled = floor_capped_weights(next_scaled, min_weight=min_weight)
        if next_scaled.keys() == scaled.keys():
            return next_scaled
        scaled = next_scaled
        if not scaled:
            return {}
    return scaled


def scale_weight_rows_to_vol_target(
    universe: Universe,
    end_dates: list[str],
    weight_rows: list[dict[str, float]],
    *,
    lookback_months: int,
    target_vol: float,
    min_weight: float,
) -> list[dict[str, float]]:
    return [
        scale_row_to_vol_target(
            universe,
            end_date,
            row,
            lookback_months=lookback_months,
            target_vol=target_vol,
            min_weight=min_weight,
        )
        for end_date, row in zip(end_dates, weight_rows)
    ]


def portfolio_weekly_return(
    weights: dict[str, float],
    universe: Universe,
    iso_date: str,
) -> float:
    gross = 0.0
    for market_id, weight in weights.items():
        asset = universe.assets[market_id]
        gross += weight * asset.returns_by_date.get(iso_date, 0.0)
    return gross


def run_weekly_backtest(
    universe: Universe,
    trade_dates: list[str],
    end_dates: list[str],
    smoothed_rows: list[dict[str, float]],
    *,
    drift_band: float,
) -> tuple[list[WeekPoint], float]:
    prev_effective: dict[str, float] = {}
    equity = 1.0
    points: list[WeekPoint] = []

    for iso_date in trade_dates:
        start_equity = equity
        target = target_weights_for_date(end_dates, smoothed_rows, iso_date)
        if target is None:
            continue

        if prev_effective:
            effective = apply_rebalance_drift_band(
                prev_effective,
                target,
                drift_band=drift_band,
            )
        else:
            effective = dict(target)
        effective = {
            market_id: weight
            for market_id, weight in effective.items()
            if weight > 1e-12
        }
        invested_total = sum(effective.values())
        if invested_total > 1.0 + 1e-12:
            effective = {
                market_id: weight / invested_total
                for market_id, weight in effective.items()
            }

        spread_drag = rebalance_spread_drag(
            prev_effective,
            effective,
            universe.spread_fraction,
        )
        gross = portfolio_weekly_return(effective, universe, iso_date)
        net_weekly = return_after_spread_drag(gross, spread_drag)
        equity *= 1.0 + net_weekly
        weekly_return = 0.0 if start_equity <= 0 else (equity / start_equity) - 1.0

        invested = sum(effective.values())
        holdings = "|".join(
            sorted(effective, key=lambda mid: effective[mid], reverse=True)
        )
        points.append(
            WeekPoint(
                iso_date=iso_date,
                equity=equity,
                weekly_return=weekly_return,
                invested_weight=invested,
                cash_weight=max(0.0, 1.0 - invested),
                spread_drag=spread_drag,
                net_weekly=net_weekly,
                holdings=holdings,
                effective_weights=dict(effective),
                target_weights=dict(target),
            )
        )
        prev_effective = dict(effective)

    return points, equity


def write_diagnostics(path: Path, points: list[WeekPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "date",
                "equity",
                "weekly_return",
                "invested_weight",
                "cash_weight",
                "spread_drag",
                "net_weekly",
                "n_holdings",
                "holdings",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.iso_date,
                    point.equity,
                    point.weekly_return,
                    point.invested_weight,
                    point.cash_weight,
                    point.spread_drag,
                    point.net_weekly,
                    len(point.holdings.split("|")) if point.holdings else 0,
                    point.holdings,
                ]
            )


def equity_curve(weekly_returns: list[float]) -> list[float]:
    if not weekly_returns:
        return []
    return np.cumprod(1.0 + np.asarray(weekly_returns, dtype=float)).tolist()


def stats_from_weekly(
    weekly_returns: list[float],
    rf_weekly: list[float],
) -> tuple[float, float, float, float]:
    if len(weekly_returns) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    port = np.array(weekly_returns, dtype=float)
    rf = np.array(rf_weekly[: len(port)], dtype=float)
    mean_ann = float(port.mean()) * 52
    vol_ann = float(port.std(ddof=1)) * math.sqrt(52)
    std = float(port.std(ddof=1))
    sharpe = (
        float((port - rf).mean() / std) * math.sqrt(52) if std > 1e-12 else float("nan")
    )
    equity = float(np.cumprod(1.0 + port)[-1])
    years = len(port) / 52
    cagr = equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else float("nan")
    return mean_ann, vol_ann, sharpe, cagr


def benchmark_weekly(
    universe: Universe,
    trade_dates: list[str],
    *,
    benchmark_id: str = BENCHMARK_ID,
) -> list[float]:
    asset = universe.assets[benchmark_id]
    bench_first = asset.first_date or trade_dates[0]
    rf_asset = universe.assets[RISK_FREE_ID]
    out: list[float] = []
    for iso_date in trade_dates:
        if iso_date < bench_first or iso_date not in asset.returns_by_date:
            out.append(rf_asset.returns_by_date.get(iso_date, 0.0))
        else:
            out.append(asset.returns_by_date[iso_date])
    return out


CASH_WEIGHT_KEY = "__cash__"
CASH_COLOR = (0.82, 0.82, 0.82, 1.0)
STACK_EDGE_COLOR = "black"
STACK_EDGE_WIDTH = 1.0
WEIGHT_LEGEND_THRESHOLD = 1e-6


def weight_legend_indices(
    current_weights: list[float] | np.ndarray,
    *,
    threshold: float = WEIGHT_LEGEND_THRESHOLD,
) -> list[int | None]:
    """Legend order: current holdings top-to-bottom, separator, then inactive top-to-bottom."""
    active: list[int] = []
    inactive: list[int] = []
    for index, weight in enumerate(current_weights):
        if weight > threshold:
            active.append(index)
        else:
            inactive.append(index)
    order: list[int | None] = list(reversed(active))
    if active and inactive:
        order.append(None)
    order.extend(reversed(inactive))
    return order


def weight_legend_handles(
    labels: list[str],
    colors: list[tuple[float, float, float, float]],
    current_weights: list[float] | np.ndarray,
) -> list[Patch | Line2D]:
    patch_kw = {"edgecolor": STACK_EDGE_COLOR, "linewidth": STACK_EDGE_WIDTH}
    handles: list[Patch | Line2D] = []
    for index in weight_legend_indices(current_weights):
        if index is None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#808080",
                    linewidth=0.8,
                    marker="none",
                    label="",
                )
            )
            continue
        handles.append(
            Patch(facecolor=colors[index], label=labels[index], **patch_kw)
        )
    return handles


def stack_plot_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Distinct colours for many stacked series (tab20 + tab20b + tab20c, then Set3)."""
    if n <= 0:
        return []
    palette: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Dark2"):
        cmap = plt.get_cmap(cmap_name)
        for index in range(cmap.N):
            palette.append(cmap(index))  # type: ignore[arg-type]
            if len(palette) >= n:
                return palette[:n]
    return palette[:n]


def market_labels(universe: Universe) -> dict[str, str]:
    labels: dict[str, str] = {CASH_WEIGHT_KEY: "Cash"}
    for market_id, asset in universe.assets.items():
        if market_id == RISK_FREE_ID:
            continue
        ticker = (asset.yahoo_ticker or market_id).strip()
        name = (asset.name or ticker).strip()
        if len(name) > 36:
            name = name[:33] + "..."
        labels[market_id] = f"{ticker} ({name})" if ticker != market_id else name
    return labels


def weight_history_from_points(
    points: list[WeekPoint],
) -> tuple[list[str], list[str], list[list[float]]]:
    """Return trade dates, asset ids (cash last), and weight rows summing to 1."""
    if not points:
        return [], [], []
    asset_ids = sorted(
        {
            market_id
            for point in points
            for market_id in point.effective_weights
        },
        key=lambda market_id: -max(
            point.effective_weights.get(market_id, 0.0) for point in points
        ),
    )
    series_ids = asset_ids + [CASH_WEIGHT_KEY]
    dates = [point.iso_date for point in points]
    rows: list[list[float]] = []
    for point in points:
        cash = point.cash_weight
        rows.append(
            [point.effective_weights.get(market_id, 0.0) for market_id in asset_ids]
            + [cash]
        )
    return dates, series_ids, rows


def plot_portfolio_weights(
    points: list[WeekPoint],
    universe: Universe,
    *,
    output: Path,
    title: str = "ETF portfolio weights (post-smooth, floor, vol cap, drift band)",
) -> None:
    dates, series_ids, rows = weight_history_from_points(points)
    if not dates:
        raise ValueError("no weight history to plot")
    labels_map = market_labels(universe)
    labels = [labels_map.get(market_id, market_id) for market_id in series_ids]
    x = [date.fromisoformat(iso_date) for iso_date in dates]
    weights = np.asarray(rows, dtype=float).T

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = stack_plot_colors(len(series_ids))
    if series_ids and series_ids[-1] == CASH_WEIGHT_KEY:
        colors[-1] = CASH_COLOR
    ax.stackplot(
        x,
        weights,
        labels=labels,
        colors=colors,
        alpha=0.92,
        edgecolor=STACK_EDGE_COLOR,
        linewidth=STACK_EDGE_WIDTH,
    )
    cumulative = np.cumsum(weights, axis=0)
    for boundary in cumulative[:-1]:
        ax.plot(
            x,
            boundary,
            color=STACK_EDGE_COLOR,
            linewidth=STACK_EDGE_WIDTH,
            solid_capstyle="butt",
            zorder=5,
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(
        handles=weight_legend_handles(labels, colors, weights[:, -1]),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.5,
        frameon=False,
    )
    fig.autofmt_xdate()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_weight_history(path: Path, points: list[WeekPoint]) -> None:
    dates, series_ids, rows = weight_history_from_points(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", *series_ids])
        for iso_date, weight_row in zip(dates, rows):
            writer.writerow([iso_date, *weight_row])


def plot_equity(
    trade_dates: list[str],
    curves: list[EquityCurve],
    *,
    output: Path,
    title: str = "ETF vol-capped backtest",
) -> None:
    if not curves:
        raise ValueError("no curves to plot")
    dates = [date.fromisoformat(d) for d in trade_dates]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    linestyles = ["-", "-", "--", ":"]
    for index, curve in enumerate(curves):
        if len(curve.equity) != len(trade_dates):
            raise ValueError(
                f"plot length mismatch for {curve.label}: "
                f"{len(trade_dates)} dates, {len(curve.equity)} equity points"
            )
        ax.plot(
            dates,
            curve.equity,
            label=format_stats_label(curve.label, curve.stats),
            linewidth=1.8 if index == 0 else 1.4,
            alpha=1.0 if index == 0 else 0.9,
            linestyle=linestyles[index % len(linestyles)],
        )
    ax.set_ylabel("Equity (start = 1)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.autofmt_xdate()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class BacktestResult:
    universe: Universe
    trade_dates: list[str]
    points: list[WeekPoint]
    strat_stats: tuple[float, float, float, float]
    bench_stats: tuple[float, float, float, float]
    bench_equity: list[float]
    bench_returns: list[float]


def load_backtest_universe(
    *,
    project_root: Path,
    markets_csv: Path | None = None,
    yahoo_dir: Path | None = None,
    allowlist_csv: Path | None = DEFAULT_ALLOWLIST,
    dividends: str = "any",
) -> Universe:
    markets_csv = resolve_markets_csv(markets_csv)
    yahoo_dir = yahoo_dir or DEFAULT_YAHOO
    file_allowed_ids: set[str] | None = None
    if allowlist_csv is not None and allowlist_csv.is_file():
        file_allowed_ids = load_allowlist_ids(allowlist_csv)
    dividend_allowed_ids = load_dividend_policy_ids(markets_csv, dividends)
    investengine_ids = investengine_market_ids(DEFAULT_MARKETS)
    allowed_ids = combine_allowed_ids(
        file_allowed_ids,
        dividend_allowed_ids,
        investengine_ids,
    )
    if allowed_ids is not None:
        allowed_ids.add(BENCHMARK_ID)
        allowed_ids.add(RISK_FREE_ID)
    return load_universe(
        project_root=project_root,
        markets_csv=markets_csv,
        yahoo_dir=yahoo_dir,
        allowed_market_ids=allowed_ids,
    )


def run_etf_backtest(
    universe: Universe,
    *,
    backtest_years: float = 10.0,
    max_holdings: int = 20,
    target_vol: float = 0.25,
    lookback_months: int = 12,
    ewma_span: int | None = None,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    min_coverage: float = 0.95,
    listing_years: float = 1.0,
    max_abs_daily_return: float = 0.20,
    drift_band: float = 0.05,
    rebalance_frequency: str = "monthly",
) -> BacktestResult:
    ewma_span = (
        ewma_span
        if ewma_span is not None
        else default_ewma_span(rebalance_frequency)
    )
    end = universe.weekly_dates[-1]
    backtest_start = (
        date.fromisoformat(end) - timedelta(days=int(round(backtest_years * 365.25)))
    ).isoformat()
    all_tickers = allocatable_assets(universe)
    capped_holdings = max_holdings
    if min_weight > 0:
        capped_holdings = min(capped_holdings, max(1, int(1.0 / min_weight)))

    trade_dates, points, strat_stats = build_schedule_and_run(
        universe,
        all_tickers,
        backtest_start=backtest_start,
        end=end,
        lookback_months=lookback_months,
        target_vol=target_vol,
        max_holdings=capped_holdings,
        min_weight=min_weight,
        min_coverage=min_coverage,
        listing_years=listing_years,
        max_abs_daily_return=max_abs_daily_return,
        ewma_span=ewma_span,
        rebalance_frequency=rebalance_frequency,
        drift_band=drift_band,
    )
    rf_returns = [
        universe.assets[RISK_FREE_ID].returns_by_date[d] for d in trade_dates
    ]
    bench_returns = benchmark_weekly(universe, trade_dates)
    bench_stats = stats_from_weekly(bench_returns, rf_returns)
    bench_equity = equity_curve(bench_returns)
    return BacktestResult(
        universe=universe,
        trade_dates=trade_dates,
        points=points,
        strat_stats=strat_stats,
        bench_stats=bench_stats,
        bench_equity=bench_equity,
        bench_returns=bench_returns,
    )


def run_vol_cap_sensitivity_backtests(
    universe: Universe,
    vol_caps: list[float] | tuple[float, ...],
    *,
    backtest_years: float = 10.0,
    max_holdings: int = 20,
    lookback_months: int = 12,
    ewma_span: int | None = None,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    min_coverage: float = 0.95,
    listing_years: float = 1.0,
    max_abs_daily_return: float = 0.20,
    drift_band: float = 0.05,
    rebalance_frequency: str = "monthly",
) -> tuple[list[str], list[VolCapBacktestCurve]]:
    """Run walk-forward backtests at multiple vol caps (separate optimiser per cap)."""
    ewma_span = (
        ewma_span
        if ewma_span is not None
        else default_ewma_span(rebalance_frequency)
    )
    end = universe.weekly_dates[-1]
    backtest_start = (
        date.fromisoformat(end) - timedelta(days=int(round(backtest_years * 365.25)))
    ).isoformat()
    all_tickers = allocatable_assets(universe)
    capped_holdings = max_holdings
    if min_weight > 0:
        capped_holdings = min(capped_holdings, max(1, int(1.0 / min_weight)))

    trade_dates: list[str] = []
    curves: list[VolCapBacktestCurve] = []
    for vol_cap in vol_caps:
        print(f"  vol cap {vol_cap * 100:.0f}%…", flush=True)
        td, points, stats = build_schedule_and_run(
            universe,
            all_tickers,
            backtest_start=backtest_start,
            end=end,
            lookback_months=lookback_months,
            target_vol=vol_cap,
            max_holdings=capped_holdings,
            min_weight=min_weight,
            min_coverage=min_coverage,
            listing_years=listing_years,
            max_abs_daily_return=max_abs_daily_return,
            ewma_span=ewma_span,
            rebalance_frequency=rebalance_frequency,
            drift_band=drift_band,
        )
        if not trade_dates:
            trade_dates = td
        elif td != trade_dates:
            raise ValueError(
                f"trade date mismatch at vol cap {vol_cap}: "
                f"{len(trade_dates)} vs {len(td)} weeks"
            )
        equity = [point.equity for point in points]
        mean_ann, vol_ann, sharpe, cagr = stats
        curves.append(
            VolCapBacktestCurve(
                vol_cap=vol_cap,
                equity=equity,
                mean_ann=mean_ann,
                vol_ann=vol_ann,
                sharpe=sharpe,
                cagr=cagr,
                max_drawdown=max_drawdown_from_equity(equity),
            )
        )
    return trade_dates, curves


def estimate_runtime_steps(
    backtest_years: float,
    universe: Universe,
    *,
    lookback_months: int,
    rebalance_frequency: str,
) -> int:
    end = universe.weekly_dates[-1]
    backtest_start = (
        date.fromisoformat(end) - timedelta(days=int(round(backtest_years * 365.25)))
    ).isoformat()
    schedule_start = window_start_from_end(backtest_start, months=lookback_months)
    weeks = [
        d
        for d in build_weekly_end_dates(universe, lookback_months=lookback_months)
        if schedule_start <= d <= end
    ]
    return len(select_rebalance_end_dates(weeks, rebalance_frequency))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-years", type=float, default=10.0)
    parser.add_argument(
        "--max-holdings",
        type=int,
        default=20,
        help="Hard safety cap; with --min-weight 0.05 no more than 20 can survive",
    )
    parser.add_argument("--target-vol", type=float, default=0.25)
    parser.add_argument("--lookback-months", type=int, default=12)
    parser.add_argument(
        "--ewma-span",
        type=int,
        default=None,
        help="EWMA span in rebalance periods (default: 6 monthly, 24 weekly)",
    )
    parser.add_argument("--min-weight", type=float, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--listing-years", type=float, default=1.0)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.20)
    parser.add_argument("--drift-band", type=float, default=0.05)
    parser.add_argument(
        "--rebalance-frequency",
        choices=REBALANCE_FREQUENCIES,
        default="monthly",
        help="How often to recompute/manual-trade the ETF basket",
    )
    parser.add_argument(
        "--dividends",
        choices=["any", "accumulating", "distributing"],
        default="any",
        help="Dividend policy filter (default: any — acc and dist for IE ISA)",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--markets-csv", type=Path, default=None)
    parser.add_argument("--yahoo-dir", type=Path, default=DEFAULT_YAHOO)
    parser.add_argument(
        "--allowlist-csv",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Only load ETFs whose id appears in this CSV (default: market_stats.csv)",
    )
    args = parser.parse_args()
    ewma_span = (
        args.ewma_span
        if args.ewma_span is not None
        else default_ewma_span(args.rebalance_frequency)
    )

    markets_csv = resolve_markets_csv(args.markets_csv)

    print("Loading universe…", flush=True)
    t0 = time.perf_counter()
    universe = load_backtest_universe(
        project_root=Path.cwd(),
        markets_csv=markets_csv,
        yahoo_dir=args.yahoo_dir,
        allowlist_csv=args.allowlist_csv,
        dividends=args.dividends,
    )
    load_s = time.perf_counter() - t0
    n_assets = len([m for m in universe.assets if m != RISK_FREE_ID])
    print(f"Loaded {n_assets} allocatable ETFs (+ risk-free)", flush=True)

    n_steps = estimate_runtime_steps(
        args.backtest_years,
        universe,
        lookback_months=args.lookback_months,
        rebalance_frequency=args.rebalance_frequency,
    )
    est_lo = load_s + n_steps * 6
    est_hi = load_s + n_steps * 20
    print(
        f"Runtime estimate: ~{est_lo/60:.0f}–{est_hi/60:.0f} min "
        f"({n_steps} {args.rebalance_frequency} optimiser steps × ~6–20s, "
        f"plus ~{load_s:.0f}s load)",
        flush=True,
    )
    if args.estimate_only:
        return 0

    print(
        f"Building {args.rebalance_frequency} weight schedule (ETF GIA universe)…",
        flush=True,
    )
    t1 = time.perf_counter()
    result = run_etf_backtest(
        universe,
        backtest_years=args.backtest_years,
        max_holdings=args.max_holdings,
        target_vol=args.target_vol,
        lookback_months=args.lookback_months,
        ewma_span=ewma_span,
        min_weight=args.min_weight,
        min_coverage=args.min_coverage,
        listing_years=args.listing_years,
        max_abs_daily_return=args.max_abs_daily_return,
        drift_band=args.drift_band,
        rebalance_frequency=args.rebalance_frequency,
    )
    sched_s = time.perf_counter() - t1
    print(
        f"Schedule built in {sched_s/60:.1f} min ({len(result.trade_dates)} trade weeks)",
        flush=True,
    )
    write_diagnostics(args.diagnostics, result.points)

    strat_returns = [p.weekly_return for p in result.points]
    strat_eq = [p.equity for p in result.points]
    strat_from_returns = equity_curve(strat_returns)
    if strat_from_returns:
        max_drift = max(abs(a - b) for a, b in zip(strat_eq, strat_from_returns))
        if max_drift > 1e-6:
            print(
                f"warn: strategy equity drift vs weekly returns (max {max_drift:.2e})",
                flush=True,
            )

    curves = [
        EquityCurve(label="ETF GIA", equity=strat_eq, stats=result.strat_stats),
        EquityCurve(
            label="VWRP",
            equity=result.bench_equity,
            stats=result.bench_stats,
        ),
    ]
    plot_equity(result.trade_dates, curves, output=args.output)

    invested = [p.invested_weight for p in result.points]
    cash = [p.cash_weight for p in result.points]
    print()
    print(
        f"Hold period: {result.trade_dates[0]} .. {result.trade_dates[-1]} "
        f"({len(result.trade_dates)} weeks)"
    )
    print(
        f"Invested weight: mean {np.mean(invested)*100:.1f}%, "
        f"median {np.median(invested)*100:.1f}%, "
        f"weeks with cash > 1%: {sum(c > 0.01 for c in cash)}"
    )
    print()
    stats_rows: list[tuple[str, tuple[float, float, float, float]]] = [
        ("ETF GIA", result.strat_stats),
        ("VWRP benchmark", result.bench_stats),
    ]
    print_stats_table(stats_rows)
    print()
    print(f"Plot: {args.output}")
    print(f"Diagnostics: {args.diagnostics}")
    print(f"Total wall time: {(time.perf_counter()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
