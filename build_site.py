#!/usr/bin/env python3
"""Build ETF strategy static site in public/."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
ETFS_DIR = ROOT / "etfs"
if str(ETFS_DIR) not in sys.path:
    sys.path.insert(0, str(ETFS_DIR))

import config as etf_config  # noqa: E402
from sharpening_backtest import (  # noqa: E402
    VolCapBacktestCurve,
    load_backtest_universe,
    max_drawdown_from_equity,
    plot_portfolio_weights,
    run_etf_backtest,
    run_vol_cap_sensitivity_backtests,
    write_diagnostics,
)
from site_builder.investengine_portfolio import (  # noqa: E402
    InvestEngineSnapshot,
    append_ie_only_allocations,
    ie_icons_by_market_id,
    latest_portfolio_json,
    load_investengine_snapshot,
)
from site_builder.etf_data import (  # noqa: E402
    allocation_rows,
    ath_snapshot,
    drawdown_snapshot,
    period_returns,
    point_at_or_before,
    rebased_equity,
    rolling_metric_charts,
    summary_stats,
    tracking_anchor_index,
)
from site_builder.etf_html import build_index_html  # noqa: E402
from site_builder.etf_plots import (  # noqa: E402
    plot_backtest_ath_distribution,
    plot_backtest_drawdown_distribution,
    plot_backtest_return_histogram,
    plot_etf_drawdown,
    plot_etf_equity,
    plot_etf_rolling_metric_chart,
    plot_etf_vol_cap_equity,
    plot_invested_weight,
    write_sparklines,
)
from site_builder.metrics import (  # noqa: E402
    benchmark_regression_stats,
    days_since_ath_series,
    distribution_stats,
    drawdown_series,
)
from site_builder.publish import (  # noqa: E402
    build_timestamp,
    publish_snapshot_to_root,
    write_builds_index,
)
from strategy.constants import RISK_FREE_ID  # noqa: E402


def _load_ie_snapshot(
    *,
    project_root: Path,
    snapshot_dir: Path,
    universe,
    as_of: date,
) -> InvestEngineSnapshot | None:
    json_dir = project_root / etf_config.INVESTENGINE_JSON_DIR
    json_path = latest_portfolio_json(json_dir, prefer_date=as_of)
    if json_path is None:
        print("No InvestEngine portfolio JSON found; skipping live portfolio tables.", flush=True)
        return None
    print(f"Loading InvestEngine portfolio from {json_path}…", flush=True)
    return load_investengine_snapshot(
        json_path,
        universe=universe,
        icons_cache_dir=project_root / etf_config.INVESTENGINE_ICONS_CACHE_DIR,
        snapshot_icons_dir=snapshot_dir / "icons",
    )


def build_snapshot(
    *,
    snapshot_dir: Path,
    project_root: Path,
    generated_at: str,
    tracking_start: str,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = ETFS_DIR / "output" / "sharpening_weekly_diagnostics.csv"

    print("Loading universe…", flush=True)
    universe = load_backtest_universe(
        project_root=project_root,
        markets_csv=etf_config.MARKETS_STATS_ALLOWLIST,
        yahoo_dir=etf_config.YAHOO_DIR,
        allowlist_csv=etf_config.MARKET_STATS_ALLOWLIST,
        dividends=etf_config.DIVIDENDS,
    )

    print("Running backtest…", flush=True)
    result = run_etf_backtest(
        universe,
        backtest_years=etf_config.BACKTEST_YEARS,
        max_holdings=etf_config.MAX_HOLDINGS,
        target_vol=etf_config.TARGET_VOL,
        lookback_months=etf_config.LOOKBACK_MONTHS,
        ewma_span=etf_config.EWMA_SPAN,
        min_weight=etf_config.MIN_WEIGHT,
        rebalance_frequency=etf_config.REBALANCE_FREQUENCY,
        drift_band=etf_config.DRIFT_BAND,
    )
    write_diagnostics(diagnostics_path, result.points)

    print("Running vol-cap sensitivity backtests…", flush=True)
    vol_caps = list(etf_config.VOL_CAP_SENSITIVITY)
    vol_cap_trade_dates = result.trade_dates
    vol_cap_curves = []
    if etf_config.TARGET_VOL in vol_caps:
        strat_equity = [point.equity for point in result.points]
        mean_ann, vol_ann, sharpe, cagr = result.strat_stats
        vol_cap_curves.append(
            VolCapBacktestCurve(
                vol_cap=etf_config.TARGET_VOL,
                equity=strat_equity,
                mean_ann=mean_ann,
                vol_ann=vol_ann,
                sharpe=sharpe,
                cagr=cagr,
                max_drawdown=max_drawdown_from_equity(strat_equity),
            )
        )
        vol_caps = [cap for cap in vol_caps if cap != etf_config.TARGET_VOL]
    if vol_caps:
        extra_trade_dates, extra_curves = run_vol_cap_sensitivity_backtests(
            universe,
            vol_caps,
            backtest_years=etf_config.BACKTEST_YEARS,
            max_holdings=etf_config.MAX_HOLDINGS,
            lookback_months=etf_config.LOOKBACK_MONTHS,
            ewma_span=etf_config.EWMA_SPAN,
            min_weight=etf_config.MIN_WEIGHT,
            rebalance_frequency=etf_config.REBALANCE_FREQUENCY,
            drift_band=etf_config.DRIFT_BAND,
        )
        if extra_trade_dates != vol_cap_trade_dates:
            raise ValueError(
                "vol-cap sensitivity trade dates differ from primary backtest"
            )
        vol_cap_curves.extend(extra_curves)
    vol_cap_curves.sort(key=lambda curve: curve.vol_cap)

    latest = result.points[-1]
    as_of = date.fromisoformat(latest.iso_date)
    bench_label = etf_config.BENCHMARK_LABEL
    strat_returns = [point.weekly_return for point in result.points]
    rf_returns = [
        universe.assets[RISK_FREE_ID].returns_by_date[d] for d in result.trade_dates
    ]
    strat_summary = summary_stats(strat_returns, rf_returns)
    bench_regression = benchmark_regression_stats(strat_returns, result.bench_returns)

    plot_etf_equity(
        trade_dates=result.trade_dates,
        strat_equity=[point.equity for point in result.points],
        bench_equity=result.bench_equity,
        bench_label=bench_label,
        tracking_start=tracking_start,
        output=snapshot_dir / "equity.png",
    )
    plot_etf_vol_cap_equity(
        trade_dates=vol_cap_trade_dates,
        curves=vol_cap_curves,
        tracking_start=tracking_start,
        highlight_vol_cap=etf_config.TARGET_VOL,
        output=snapshot_dir / "equity_vol_caps.png",
    )
    plot_etf_drawdown(
        trade_dates=result.trade_dates,
        strat_equity=[point.equity for point in result.points],
        tracking_start=tracking_start,
        output=snapshot_dir / "drawdown.png",
    )
    plot_portfolio_weights(
        result.points,
        universe,
        output=snapshot_dir / "weights.png",
        title="ETF portfolio weights",
    )
    plot_invested_weight(
        points=result.points,
        tracking_start=tracking_start,
        output=snapshot_dir / "invested.png",
    )

    metric_charts = rolling_metric_charts(
        points=result.points,
        bench_returns=result.bench_returns,
        universe=universe,
    )
    sharpe_1y = None
    for chart in metric_charts:
        plot_etf_rolling_metric_chart(
            chart,
            snapshot_dir / f"{chart.slug}.png",
            tracking_start=tracking_start,
            bench_label=f"{bench_label} prior 1y",
        )
        if chart.slug == "sharpe" and chart.backtest.values:
            sharpe_1y = chart.backtest.values[-1]

    plot_backtest_return_histogram(
        returns=strat_returns,
        stats=distribution_stats(strat_returns),
        output=snapshot_dir / "weekly_returns_hist.png",
    )
    anchor = tracking_anchor_index(result.trade_dates, tracking_start)
    rebased = rebased_equity([point.equity for point in result.points], anchor)
    rebased_drawdowns = drawdown_series(rebased)
    plot_backtest_drawdown_distribution(
        drawdowns=rebased_drawdowns,
        current_drawdown=rebased_drawdowns[-1] if rebased_drawdowns else None,
        output=snapshot_dir / "drawdown_dist.png",
    )
    days_since_ath = days_since_ath_series(result.trade_dates, rebased)
    plot_backtest_ath_distribution(
        days_since_ath=days_since_ath,
        current_days_since_ath=days_since_ath[-1] if days_since_ath else None,
        output=snapshot_dir / "ath_dist.png",
    )

    ie_snapshot = _load_ie_snapshot(
        project_root=project_root,
        snapshot_dir=snapshot_dir,
        universe=universe,
        as_of=as_of,
    )
    ie_weights = ie_snapshot.etf_weights_by_market_id if ie_snapshot else None
    ie_icons = ie_icons_by_market_id(ie_snapshot) if ie_snapshot else None

    point_1y_ago = point_at_or_before(
        result.points,
        (as_of - timedelta(days=365)).isoformat(),
    )
    allocations = allocation_rows(
        universe,
        latest,
        yahoo_dir=etf_config.YAHOO_DIR,
        spark_dir=snapshot_dir / "sparklines",
        as_of=as_of,
        ie_weights_by_market_id=ie_weights,
        ie_icons_by_market_id=ie_icons,
        point_1y_ago=point_1y_ago,
    )
    allocations = append_ie_only_allocations(
        allocations,
        universe=universe,
        ie_snapshot=ie_snapshot,
        risk_free_id=RISK_FREE_ID,
    )
    write_sparklines(
        allocations,
        yahoo_dir=etf_config.YAHOO_DIR,
        spark_dir=snapshot_dir / "sparklines",
        as_of=as_of,
    )

    build_index_html(
        output=snapshot_dir / "index.html",
        universe=universe,
        generated_at=generated_at,
        tracking_start=tracking_start,
        as_of_date=latest.iso_date,
        strat_stats=strat_summary,
        bench_regression=bench_regression,
        bench_label=bench_label,
        drawdown=drawdown_snapshot(result.points),
        ath=ath_snapshot(result.points),
        period_returns=period_returns(
            result.points,
            strat_returns,
            tracking_start=tracking_start,
        ),
        allocations=allocations,
        invested_weight=latest.invested_weight,
        cash_weight=latest.cash_weight,
        sharpe_1y=sharpe_1y,
        portfolio_url=etf_config.INVESTENGINE_PORTFOLIO_URL,
        ie_snapshot=ie_snapshot,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: public/)")
    parser.add_argument(
        "--tracking-start",
        default=etf_config.TRACKING_START_DATE,
        help=f"Tracking start date YYYY-MM-DD (default: {etf_config.TRACKING_START_DATE})",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    public_dir = args.output or (project_root / "public")
    builds_dir = public_dir / "builds"
    public_dir.mkdir(parents=True, exist_ok=True)

    snapshot_id = build_timestamp()
    snapshot_dir = builds_dir / snapshot_id
    generated_at = snapshot_id.removesuffix("Z").replace("T", " ")
    build_snapshot(
        snapshot_dir=snapshot_dir,
        project_root=project_root,
        generated_at=generated_at,
        tracking_start=args.tracking_start,
    )

    publish_snapshot_to_root(snapshot_dir, public_dir)
    write_builds_index(builds_dir)

    print(f"Wrote build snapshot to {snapshot_dir}")
    print(f"Published current site to {public_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
