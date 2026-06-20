#!/usr/bin/env python3
"""Fetch daily OHLC history from Yahoo Finance for markets in a manifest CSV.

By default, existing files are updated by merging in new rows only.
Use --force for a full re-download of each series.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("etfs/output/markets_stats_allowlist.csv")
DEFAULT_OUTPUT_DIR = Path("etfs/yahoo")
HISTORY_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def load_markets(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [column for column in HISTORY_COLUMNS if column in df.columns]
    out = df[keep].copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = pd.to_datetime(out.index).normalize()
    return out.sort_index()


def read_existing_history(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return _normalize_history_df(df)


def merge_histories(
    existing: pd.DataFrame | None,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    new_df = _normalize_history_df(new_df)
    if existing is None or existing.empty:
        return new_df
    if new_df.empty:
        return existing
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def last_history_date(df: pd.DataFrame | None) -> date | None:
    if df is None or df.empty:
        return None
    return pd.Timestamp(df.index[-1]).date()


def fetch_history(ticker: str, *, start: date | None = None) -> pd.DataFrame:
    import yfinance as yf

    ticker_obj = yf.Ticker(ticker)
    if start is None:
        raw = ticker_obj.history(period="max", interval="1d", auto_adjust=False)
    else:
        raw = ticker_obj.history(start=start.isoformat(), interval="1d", auto_adjust=False)
    return _normalize_history_df(raw)


def write_history_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out.index.name = "Date"
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".csv.tmp",
    )
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        out.to_csv(tmp)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def update_market_history(
    ticker: str,
    out_path: Path,
    *,
    force: bool = False,
) -> tuple[str, int, int, date | None, date | None]:
    """Update one market file. Returns action, total rows, rows added, first date, last date."""
    existing = None if force else read_existing_history(out_path)
    rows_before = 0 if existing is None else len(existing)

    if existing is None or existing.empty:
        merged = fetch_history(ticker)
        action = "fetched"
    else:
        last = last_history_date(existing)
        start = last + timedelta(days=1) if last is not None else None
        new_df = fetch_history(ticker, start=start)
        if last is not None and not new_df.empty:
            new_df = new_df[new_df.index > pd.Timestamp(last)]
        merged = merge_histories(existing, new_df)
        added = len(merged) - rows_before
        action = "merged" if added > 0 else "unchanged"

    if merged.empty:
        return "failed", 0, 0, None, None

    write_history_atomic(out_path, merged)
    first = pd.Timestamp(merged.index[0]).date()
    last = pd.Timestamp(merged.index[-1]).date()
    added = len(merged) - rows_before
    return action, len(merged), added, first, last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between Yahoo requests (default: 0.5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download full history and replace each output file",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Only fetch these market ids (default: all rows with a yahoo_ticker)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    markets = load_markets(args.input)
    id_filter = set(args.ids) if args.ids else None

    ok = unchanged = failed = 0

    for row in markets:
        market_id = row["id"].strip()
        ticker = (row.get("yahoo_ticker") or "").strip()

        if id_filter is not None and market_id not in id_filter:
            continue

        if not ticker:
            print(f"skip {market_id}: no yahoo_ticker")
            continue

        out_path = args.output_dir / f"{market_id}.csv"
        mode = "force" if args.force else "merge"
        print(f"{mode} {market_id} ({ticker}) -> {out_path}")
        try:
            action, total_rows, added, first, last = update_market_history(
                ticker,
                out_path,
                force=args.force,
            )
            if action == "failed":
                print(f"  warn: no data returned for {ticker}", file=sys.stderr)
                failed += 1
            elif action == "unchanged":
                print(f"  unchanged: {total_rows} rows through {last}")
                unchanged += 1
            elif action == "merged":
                print(f"  merged: +{added} rows, {total_rows} total, {first} .. {last}")
                ok += 1
            else:
                print(f"  fetched: {total_rows} rows, {first} .. {last}")
                ok += 1
        except Exception as exc:
            print(f"  error: {exc}", file=sys.stderr)
            failed += 1

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\ndone: {ok} updated, {unchanged} unchanged, {failed} failed")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
