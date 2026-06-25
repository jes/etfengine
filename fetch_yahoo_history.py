#!/usr/bin/env python3
"""Fetch daily OHLC history from Yahoo Finance for markets in a manifest CSV.

By default, existing files are updated by merging in new rows only. Incremental
updates re-fetch from the last stored date (overlap) so Yahoo dividend restatements
can be detected; when Adj Close shifts on overlap, all stored Adj Close values are
rescaled before merge. Use --force for a full re-download of each series.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("etfs/markets.csv")
DEFAULT_OUTPUT_DIR = Path("etfs/yahoo")
HISTORY_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
ADJUSTED_CLOSE_COLUMN = "Adj Close"


def _valid_adj_close(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or math.isnan(parsed):
        return None
    return parsed


def infer_adj_close_restatement_factor(
    existing: pd.DataFrame,
    new_df: pd.DataFrame,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-8,
) -> float | None:
    """Return multiplicative factor to align stored Adj Close with Yahoo's overlap row.

    Yahoo backward-adjusts historical Adj Close when dividends are recorded. On an
    incremental fetch we overlap the last stored date; if Adj Close moved while
    Close is unchanged, rescale all stored Adj Close values by new_adj / old_adj.
    """
    if existing.empty or new_df.empty:
        return None
    if ADJUSTED_CLOSE_COLUMN not in existing.columns or ADJUSTED_CLOSE_COLUMN not in new_df.columns:
        return None

    overlap = existing.index.intersection(new_df.index)
    if overlap.empty:
        return None

    factors: list[float] = []
    for ts in sorted(overlap):
        old_adj = _valid_adj_close(existing.loc[ts, ADJUSTED_CLOSE_COLUMN])
        new_adj = _valid_adj_close(new_df.loc[ts, ADJUSTED_CLOSE_COLUMN])
        if old_adj is None or new_adj is None:
            continue
        old_close = _valid_adj_close(existing.loc[ts, "Close"])
        new_close = _valid_adj_close(new_df.loc[ts, "Close"])
        if old_close is None or new_close is None:
            continue
        if abs(old_close - new_close) > max(abs_tol, rel_tol * old_close):
            continue
        factors.append(new_adj / old_adj)

    if not factors:
        return None

    factor = factors[-1]
    ref = factors[-1]
    for candidate in factors[:-1]:
        if abs(candidate - ref) > max(abs_tol, rel_tol * abs(ref)):
            return None

    if abs(factor - 1.0) <= max(abs_tol, rel_tol * abs(ref)):
        return None
    return factor


def apply_adj_close_restatement(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    if ADJUSTED_CLOSE_COLUMN not in df.columns:
        return df
    out = df.copy()
    out[ADJUSTED_CLOSE_COLUMN] = out[ADJUSTED_CLOSE_COLUMN] * factor
    return out


def restate_and_merge_histories(
    existing: pd.DataFrame,
    new_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float | None]:
    factor = infer_adj_close_restatement_factor(existing, new_df)
    if factor is not None:
        existing = apply_adj_close_restatement(existing, factor)
    return merge_histories(existing, new_df), factor


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
) -> tuple[str, int, int, date | None, date | None, float | None]:
    """Update one market file.

    Returns action, total rows, rows added, first date, last date, restatement factor.
    """
    existing = None if force else read_existing_history(out_path)
    rows_before = 0 if existing is None else len(existing)
    restatement_factor: float | None = None

    if existing is None or existing.empty:
        merged = fetch_history(ticker)
        action = "fetched"
    else:
        last = last_history_date(existing)
        start = last if last is not None else None
        new_df = fetch_history(ticker, start=start)
        merged, restatement_factor = restate_and_merge_histories(existing, new_df)
        added = len(merged) - rows_before
        if restatement_factor is not None and added > 0:
            action = "merged+restatement"
        elif restatement_factor is not None:
            action = "restatement"
        elif added > 0:
            action = "merged"
        else:
            action = "unchanged"

    if merged.empty:
        return "failed", 0, 0, None, None, None

    write_history_atomic(out_path, merged)
    first = pd.Timestamp(merged.index[0]).date()
    last = pd.Timestamp(merged.index[-1]).date()
    added = len(merged) - rows_before
    return action, len(merged), added, first, last, restatement_factor


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
            action, total_rows, added, first, last, restatement_factor = update_market_history(
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
            elif action == "restatement":
                print(
                    f"  restated Adj Close x{restatement_factor:.8g}: "
                    f"{total_rows} rows, {first} .. {last}"
                )
                ok += 1
            elif action == "merged+restatement":
                print(
                    f"  merged: +{added} rows, restated Adj Close x{restatement_factor:.8g}, "
                    f"{total_rows} total, {first} .. {last}"
                )
                ok += 1
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
