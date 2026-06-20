#!/usr/bin/env python3
"""Per-market full-history stats for the ETF universe (weekly returns)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ETFS_DIR = Path(__file__).resolve().parent

from strategy.constants import RISK_FREE_ID, WEEKS_PER_YEAR
from strategy.data import UNIT_SCALE_BOTH_DIRECTIONS, Universe, load_universe

DEFAULT_MARKETS = ETFS_DIR / "markets.csv"
DEFAULT_YAHOO = ETFS_DIR / "yahoo"
DEFAULT_OUTPUT = ETFS_DIR / "output" / "market_stats.csv"
DEFAULT_MIN_WEEKS = 500


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return float(worst)


def asset_stats(
    universe: Universe,
    market_id: str,
    *,
    bad_daily_threshold: float,
    bad_weekly_threshold: float,
    min_weeks: int,
) -> dict[str, object] | None:
    asset = universe.assets.get(market_id)
    if asset is None or not asset.returns_by_date:
        return None

    dates = sorted(asset.returns_by_date)
    if len(dates) < min_weeks:
        return None
    rf = universe.assets[RISK_FREE_ID].returns_by_date
    weekly = np.array([asset.returns_by_date[d] for d in dates], dtype=float)
    rf_weekly = np.array([rf.get(d, 0.0) for d in dates], dtype=float)
    daily = np.array(list(asset.daily_returns_by_date.values()), dtype=float)

    if weekly.size < 2:
        return None

    equity = np.cumprod(1.0 + weekly)
    years = weekly.size / WEEKS_PER_YEAR
    mean_ann = float(weekly.mean()) * WEEKS_PER_YEAR
    vol_ann = float(weekly.std(ddof=1)) * math.sqrt(WEEKS_PER_YEAR)
    std = float(weekly.std(ddof=1))
    excess = weekly - rf_weekly
    sharpe = (
        float(excess.mean() / std) * math.sqrt(WEEKS_PER_YEAR) if std > 1e-12 else float("nan")
    )
    cagr = (
        float(equity[-1] ** (1.0 / years) - 1.0)
        if years > 0 and equity[-1] > 0
        else float("nan")
    )
    max_abs_weekly = float(np.max(np.abs(weekly)))
    max_abs_daily = float(np.max(np.abs(daily))) if daily.size else float("nan")
    max_dd = _max_drawdown(equity)

    flags: list[str] = []
    for warning in asset.price_unit_warnings:
        if warning == UNIT_SCALE_BOTH_DIRECTIONS:
            flags.append("unit_scale|both_directions")
    if max_abs_daily >= bad_daily_threshold:
        flags.append(f"daily|ret|>={bad_daily_threshold:.0%}")
    if max_abs_weekly >= bad_weekly_threshold:
        flags.append(f"weekly|ret|>={bad_weekly_threshold:.0%}")
    if cagr == cagr and abs(cagr) > 1.0:
        flags.append("cagr|>|100%")
    if vol_ann == vol_ann and vol_ann > 1.0:
        flags.append("vol|>|100%")

    return {
        "id": market_id,
        "market_name": universe.market_names.get(market_id, asset.name),
        "category": universe.market_categories.get(market_id, ""),
        "yahoo_ticker": asset.yahoo_ticker,
        "first_date": dates[0],
        "last_date": dates[-1],
        "weeks": weekly.size,
        "mean_ann": mean_ann,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "max_abs_weekly_return": max_abs_weekly,
        "max_abs_daily_return": max_abs_daily,
        "flags": ";".join(flags),
    }


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    fieldnames = [
        "id",
        "market_name",
        "category",
        "yahoo_ticker",
        "first_date",
        "last_date",
        "weeks",
        "mean_ann",
        "vol_ann",
        "sharpe",
        "cagr",
        "max_drawdown",
        "max_abs_weekly_return",
        "max_abs_daily_return",
        "flags",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("mean_ann", "vol_ann", "sharpe", "cagr", "max_drawdown"):
                value = out[key]
                if isinstance(value, float) and value == value:
                    out[key] = f"{value:.6f}"
                else:
                    out[key] = ""
            for key in ("max_abs_weekly_return", "max_abs_daily_return"):
                value = out[key]
                if isinstance(value, float) and value == value:
                    out[key] = f"{value:.6f}"
                else:
                    out[key] = ""
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets-csv", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--yahoo-dir", type=Path, default=DEFAULT_YAHOO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bad-daily-threshold", type=float, default=0.20)
    parser.add_argument("--bad-weekly-threshold", type=float, default=0.50)
    parser.add_argument("--min-weeks", type=int, default=DEFAULT_MIN_WEEKS)
    parser.add_argument("--include-risk-free", action="store_true")
    args = parser.parse_args()

    print("Loading universe…", flush=True)
    universe = load_universe(
        project_root=Path.cwd(),
        markets_csv=args.markets_csv,
        yahoo_dir=args.yahoo_dir,
    )

    market_ids = sorted(universe.assets)
    if not args.include_risk_free:
        market_ids = [market_id for market_id in market_ids if market_id != RISK_FREE_ID]

    rows: list[dict[str, object]] = []
    flagged = 0
    for market_id in market_ids:
        row = asset_stats(
            universe,
            market_id,
            bad_daily_threshold=args.bad_daily_threshold,
            bad_weekly_threshold=args.bad_weekly_threshold,
            min_weeks=args.min_weeks,
        )
        if row is None:
            continue
        if row["flags"]:
            flagged += 1
        rows.append(row)

    rows.sort(key=lambda row: (row["sharpe"] if row["sharpe"] == row["sharpe"] else -999), reverse=True)
    write_csv(rows, args.output)

    print(f"Wrote {len(rows)} markets to {args.output} (min {args.min_weeks} weeks)")
    print(f"Flagged (bad tick / extreme stats): {flagged}")
    if rows:
        top = rows[0]
        print(
            f"Top Sharpe: {top['yahoo_ticker']} ({top['sharpe']})  "
            f"CAGR {float(top['cagr']):.1%}  vol {float(top['vol_ann']):.1%}"
            if top["sharpe"] == top["sharpe"]
            else "Top Sharpe: n/a"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
