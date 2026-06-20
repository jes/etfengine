#!/usr/bin/env python3
"""Intersect InvestEngine markets.csv with market_stats.csv for backtest manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.constants import RISK_FREE_ID

BENCHMARK_ID = "ie00bk5bqt80"  # VWRP.L

ETFS_DIR = Path(__file__).resolve().parent
DEFAULT_MARKETS = ETFS_DIR / "markets.csv"
DEFAULT_STATS = ETFS_DIR / "output" / "market_stats.csv"
DEFAULT_OUTPUT = ETFS_DIR / "output" / "markets_stats_allowlist.csv"
DEFAULT_FILTERED_STATS = ETFS_DIR / "output" / "market_stats.csv"


def load_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["id"].strip().lower()
            for row in csv.DictReader(handle)
            if row.get("id")
        }


def filter_csv_by_ids(
    source: Path,
    output: Path,
    allowed_ids: set[str],
    *,
    always_include: set[str] | None = None,
) -> int:
    always_include = always_include or set()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{source} is empty")
        rows = [
            row
            for row in reader
            if row.get("id", "").strip().lower() in allowed_ids
            or row.get("id", "").strip().lower() in always_include
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets-csv", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--stats-csv", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--filtered-stats-output",
        type=Path,
        default=DEFAULT_FILTERED_STATS,
        help="Rewrite market_stats.csv to the InvestEngine ∩ stats intersection",
    )
    args = parser.parse_args()

    market_ids = load_ids(args.markets_csv)
    stats_ids = load_ids(args.stats_csv)
    allowed_ids = market_ids & stats_ids
    if not allowed_ids:
        raise SystemExit("No overlap between markets.csv and market_stats.csv")

    manifest_rows = filter_csv_by_ids(
        args.markets_csv,
        args.output,
        allowed_ids,
        always_include={RISK_FREE_ID, BENCHMARK_ID},
    )
    stats_rows = filter_csv_by_ids(args.stats_csv, args.filtered_stats_output, allowed_ids)

    print(
        f"wrote {args.output}: {manifest_rows} rows "
        f"({len(market_ids)} markets ∩ {len(stats_ids)} stats)"
    )
    print(f"wrote {args.filtered_stats_output}: {stats_rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
