#!/usr/bin/env python3
"""Stacked area chart of realised ETF portfolio weights through the backtest pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ETFS_DIR = Path(__file__).resolve().parent
if str(ETFS_DIR) not in sys.path:
    sys.path.insert(0, str(ETFS_DIR))

from strategy.data import allocatable_assets, load_universe

from sharpening_backtest import (
    DEFAULT_MARKETS,
    DEFAULT_YAHOO,
    build_schedule_and_run,
    combine_allowed_ids,
    estimate_runtime_steps,
    investengine_market_ids,
    load_allowlist_ids,
    load_dividend_policy_ids,
    plot_portfolio_weights,
    resolve_markets_csv,
    write_weight_history,
)
from sharpening_optimizer import (
    DEFAULT_LISTING_YEARS,
    DEFAULT_MIN_WEIGHT,
    REBALANCE_FREQUENCIES,
    default_ewma_span,
)

DEFAULT_OUTPUT = ETFS_DIR / "output" / "sharpening_portfolio_weights.png"
DEFAULT_WEIGHTS_CSV = ETFS_DIR / "output" / "sharpening_portfolio_weights.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-years", type=float, default=10.0)
    parser.add_argument("--max-holdings", type=int, default=20)
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
    parser.add_argument("--listing-years", type=float, default=DEFAULT_LISTING_YEARS)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.20)
    parser.add_argument("--drift-band", type=float, default=0.05)
    parser.add_argument(
        "--rebalance-frequency",
        choices=REBALANCE_FREQUENCIES,
        default="monthly",
    )
    parser.add_argument(
        "--dividends",
        choices=["any", "accumulating", "distributing"],
        default="any",
        help="Dividend policy filter (default: any — acc and dist for IE ISA)",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights-csv", type=Path, default=DEFAULT_WEIGHTS_CSV)
    parser.add_argument("--markets-csv", type=Path, default=None)
    parser.add_argument("--yahoo-dir", type=Path, default=DEFAULT_YAHOO)
    parser.add_argument(
        "--allowlist-csv",
        type=Path,
        default=None,
        help="If set, only load ETFs whose id appears in this CSV",
    )
    args = parser.parse_args()
    ewma_span = (
        args.ewma_span
        if args.ewma_span is not None
        else default_ewma_span(args.rebalance_frequency)
    )

    markets_csv = resolve_markets_csv(args.markets_csv)
    file_allowed_ids: set[str] | None = None
    if args.allowlist_csv is not None:
        file_allowed_ids = load_allowlist_ids(args.allowlist_csv)
    dividend_allowed_ids = load_dividend_policy_ids(markets_csv, args.dividends)
    allowed_ids = combine_allowed_ids(
        file_allowed_ids,
        dividend_allowed_ids,
        investengine_market_ids(DEFAULT_MARKETS),
    )

    print("Loading universe…", flush=True)
    t0 = time.perf_counter()
    universe = load_universe(
        project_root=Path.cwd(),
        markets_csv=markets_csv,
        yahoo_dir=args.yahoo_dir,
        allowed_market_ids=allowed_ids,
    )
    load_s = time.perf_counter() - t0
    n_assets = len(allocatable_assets(universe))
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

    end = universe.weekly_dates[-1]
    backtest_start = (
        date.fromisoformat(end) - timedelta(days=int(round(args.backtest_years * 365.25)))
    ).isoformat()
    max_holdings = args.max_holdings
    if args.min_weight > 0:
        max_holdings = min(max_holdings, max(1, int(1.0 / args.min_weight)))

    print(
        f"Building {args.rebalance_frequency} weight schedule for weight history…",
        flush=True,
    )
    t1 = time.perf_counter()
    trade_dates, points, _ = build_schedule_and_run(
        universe,
        allocatable_assets(universe),
        backtest_start=backtest_start,
        end=end,
        lookback_months=args.lookback_months,
        target_vol=args.target_vol,
        max_holdings=max_holdings,
        min_weight=args.min_weight,
        min_coverage=args.min_coverage,
        listing_years=args.listing_years,
        max_abs_daily_return=args.max_abs_daily_return,
        ewma_span=ewma_span,
        rebalance_frequency=args.rebalance_frequency,
        drift_band=args.drift_band,
    )
    sched_s = time.perf_counter() - t1
    print(
        f"Schedule built in {sched_s/60:.1f} min ({len(trade_dates)} trade weeks)",
        flush=True,
    )

    write_weight_history(args.weights_csv, points)
    plot_portfolio_weights(points, universe, output=args.output)
    print()
    print(f"Hold period: {trade_dates[0]} .. {trade_dates[-1]} ({len(trade_dates)} weeks)")
    print(f"Plot: {args.output}")
    print(f"Weights CSV: {args.weights_csv}")
    print(f"Total wall time: {(time.perf_counter() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
