from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from strategy.constants import MARKETS_CSV, RISK_FREE_ID, YAHOO_DIR

PRICE_OHLC_COLUMNS = ("Open", "High", "Low", "Close")
ADJUSTED_CLOSE_COLUMN = "Adj Close"
# Wide band: suspicious level change worth inspecting.
UNIT_DOWN_RATIO_MIN = 1.0 / 140.0
UNIT_DOWN_RATIO_MAX = 0.30  # >70% drop vs adjusted prior
UNIT_UP_RATIO_MIN = 60.0
UNIT_UP_RATIO_MAX = 140.0
# Narrow band: treat as an exact ×100 (pence↔pounds) flip, not USD↔pence.
EXACT_UNIT_RATIO = 100.0
EXACT_UNIT_RATIO_MIN = 98.0
EXACT_UNIT_RATIO_MAX = 102.0
UNIT_SCALE_BOTH_DIRECTIONS = "unit_scale_both_directions"


def _ratio_in_unit_flip_band(ratio: float) -> bool:
    return (UNIT_DOWN_RATIO_MIN <= ratio <= UNIT_DOWN_RATIO_MAX) or (
        UNIT_UP_RATIO_MIN <= ratio <= UNIT_UP_RATIO_MAX
    )


def _is_exact_hundred_x_flip(ratio: float) -> bool:
    """True when the jump is ~100× up or ~100× down (pence/pounds style)."""
    if ratio >= 1.0:
        return EXACT_UNIT_RATIO_MIN <= ratio <= EXACT_UNIT_RATIO_MAX
    if ratio <= 0:
        return False
    inverse = 1.0 / ratio
    return EXACT_UNIT_RATIO_MIN <= inverse <= EXACT_UNIT_RATIO_MAX


def _scale_entry_direction(ratio_to_adjusted: float) -> str | None:
    if ratio_to_adjusted >= UNIT_UP_RATIO_MIN:
        return "up"
    if ratio_to_adjusted <= UNIT_DOWN_RATIO_MAX:
        return "down"
    return None


def _factor_entering_wrong_scale(
    adjusted_previous: float,
    raw_current: float,
    raw_previous: float,
) -> float:
    """First day on a wrong quoting scale."""
    ratio_to_adjusted = raw_current / adjusted_previous
    if _is_exact_hundred_x_flip(ratio_to_adjusted):
        return EXACT_UNIT_RATIO if ratio_to_adjusted < 1.0 else 1.0 / EXACT_UNIT_RATIO
    ratio_to_raw = raw_current / raw_previous
    if _ratio_in_unit_flip_band(ratio_to_raw):
        return adjusted_previous / raw_current
    return adjusted_previous / raw_previous


def patch_price_unit_series(
    df: pd.DataFrame,
    *,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply walk-forward quoting-unit correction to OHLC columns.

    Tracks whether raw prices are on a wrong scale. Enter wrong scale on a
    banded jump vs the adjusted prior; exit when raw again aligns with the
    adjusted prior (without re-scaling the return to correct units).
    """
    if df.empty or "Close" not in df.columns:
        return df
    out = df.copy()
    for col in PRICE_OHLC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(float)
    missing = [col for col in PRICE_OHLC_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"price history missing columns: {', '.join(missing)}")

    adjusted_closes = out["Close"].to_numpy(dtype=float).copy()
    raw_closes = out["Close"].to_numpy(dtype=float).copy()
    in_wrong_scale = False
    entry_directions: set[str] = set()

    for index in range(1, len(adjusted_closes)):
        adjusted_previous = float(adjusted_closes[index - 1])
        raw_previous = float(raw_closes[index - 1])
        raw_current = float(raw_closes[index])
        if adjusted_previous <= 0 or raw_previous <= 0 or raw_current <= 0:
            adjusted_closes[index] = raw_current
            continue

        ratio_to_adjusted = raw_current / adjusted_previous
        if not in_wrong_scale:
            if _ratio_in_unit_flip_band(ratio_to_adjusted):
                in_wrong_scale = True
                direction = _scale_entry_direction(ratio_to_adjusted)
                if direction is not None:
                    entry_directions.add(direction)
                factor = _factor_entering_wrong_scale(
                    adjusted_previous,
                    raw_current,
                    raw_previous,
                )
            else:
                factor = 1.0
        elif not _ratio_in_unit_flip_band(ratio_to_adjusted):
            in_wrong_scale = False
            factor = 1.0
        else:
            factor = adjusted_previous / raw_previous

        if factor == 1.0:
            adjusted_closes[index] = raw_current
            continue
        scaled_columns = PRICE_OHLC_COLUMNS
        if ADJUSTED_CLOSE_COLUMN in out.columns:
            scaled_columns = (*PRICE_OHLC_COLUMNS, ADJUSTED_CLOSE_COLUMN)
        for col in scaled_columns:
            out.iat[index, out.columns.get_loc(col)] = (
                float(out.iat[index, out.columns.get_loc(col)]) * factor
            )
        adjusted_closes[index] = raw_current * factor

    if warnings is not None and len(entry_directions) > 1:
        warnings.append(UNIT_SCALE_BOTH_DIRECTIONS)
    return out


@dataclass(frozen=True)
class WeeklyBar:
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Asset:
    market_id: str
    name: str
    yahoo_ticker: str
    returns_by_date: dict[str, float]
    daily_returns_by_date: dict[str, float]
    ohlc_by_date: dict[str, WeeklyBar]
    first_date: str | None
    price_unit_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Universe:
    assets: dict[str, Asset]
    weekly_dates: list[str]
    market_names: dict[str, str]
    market_categories: dict[str, str]
    spread_fraction: dict[str, float]


def _bars_from_csv(
    path: Path,
) -> tuple[dict[str, float], dict[str, float], dict[str, WeeklyBar], tuple[str, ...]]:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    unit_warnings: list[str] = []
    df = patch_price_unit_series(df, warnings=unit_warnings)
    if ADJUSTED_CLOSE_COLUMN in df.columns:
        adj_weekly = df[ADJUSTED_CLOSE_COLUMN].resample("W-FRI").last().dropna()
        if len(adj_weekly) >= 2:
            return_column = ADJUSTED_CLOSE_COLUMN
        else:
            return_column = "Close"
    else:
        return_column = "Close"
    daily_return = df[return_column].pct_change(fill_method=None).dropna()
    daily_returns = {
        ts.strftime("%Y-%m-%d"): float(value)
        for ts, value in daily_return.items()
    }

    weekly = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    weekly_return_prices = (
        df[return_column].resample("W-FRI").last().reindex(weekly.index).dropna()
    )
    weekly_return = weekly_return_prices.pct_change(fill_method=None).dropna()

    returns = {
        ts.strftime("%Y-%m-%d"): float(value)
        for ts, value in weekly_return.items()
    }
    ohlc = {
        ts.strftime("%Y-%m-%d"): WeeklyBar(
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
        )
        for ts, row in weekly.iterrows()
    }
    return returns, daily_returns, ohlc, tuple(unit_warnings)


def load_universe(
    *,
    project_root: Path | None = None,
    markets_csv: Path | None = None,
    yahoo_dir: Path | None = None,
    allowed_market_ids: set[str] | None = None,
) -> Universe:
    root = project_root or Path.cwd()
    markets_path = markets_csv or root / MARKETS_CSV
    yahoo_path = yahoo_dir or root / YAHOO_DIR

    assets: dict[str, Asset] = {}
    market_names: dict[str, str] = {}
    market_categories: dict[str, str] = {}
    spread_fraction: dict[str, float] = {}

    with markets_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            market_id = row["id"].strip().lower()
            ticker = (row.get("yahoo_ticker") or "").strip()
            market_names[market_id] = row.get("market_name", market_id).strip()
            market_categories[market_id] = (row.get("category") or "").strip().upper()
            spread_pct = row.get("spread_pct", "").strip()
            if spread_pct:
                spread_fraction[market_id] = float(spread_pct) / 100.0
            if not ticker:
                continue
            if (
                allowed_market_ids is not None
                and market_id not in allowed_market_ids
                and market_id != RISK_FREE_ID
            ):
                continue
            history_file = yahoo_path / f"{market_id}.csv"
            if not history_file.is_file():
                continue
            returns, daily_returns, ohlc, unit_warnings = _bars_from_csv(history_file)
            if not returns:
                continue
            dates_sorted = sorted(returns)
            assets[market_id] = Asset(
                market_id=market_id,
                name=market_names[market_id],
                yahoo_ticker=ticker,
                returns_by_date=returns,
                daily_returns_by_date=daily_returns,
                ohlc_by_date=ohlc,
                first_date=dates_sorted[0],
                price_unit_warnings=unit_warnings,
            )

    if RISK_FREE_ID not in assets:
        raise RuntimeError(
            f"Risk-free asset {RISK_FREE_ID!r} is required but missing from loaded data."
        )

    weekly_dates = sorted(assets[RISK_FREE_ID].returns_by_date)
    return Universe(
        assets=assets,
        weekly_dates=weekly_dates,
        market_names=market_names,
        market_categories=market_categories,
        spread_fraction=spread_fraction,
    )


def allocatable_assets(universe: Universe) -> list[str]:
    return [
        market_id
        for market_id, asset in universe.assets.items()
        if market_id != RISK_FREE_ID
    ]


def window_dates(universe: Universe, start: str, end: str) -> list[str]:
    return [d for d in universe.weekly_dates if start <= d <= end]


def daily_dates_for_weekly_history(
    universe: Universe,
    weekly_history: list[str],
) -> list[str]:
    """All trading days from the first through last week in weekly_history."""
    if not weekly_history:
        return []
    rf_daily = sorted(universe.assets[RISK_FREE_ID].daily_returns_by_date)
    start, end = weekly_history[0], weekly_history[-1]
    return [d for d in rf_daily if start <= d <= end]


def week_daily_dates(universe: Universe) -> dict[str, list[str]]:
    """Map each Friday week-end to sorted daily dates in that calendar week."""
    rf_daily = universe.assets[RISK_FREE_ID].daily_returns_by_date
    if not rf_daily:
        return {}
    index = pd.DatetimeIndex(sorted(rf_daily))
    grouped: dict[str, list[str]] = {}
    for week_end, dates in index.to_series().groupby(pd.Grouper(freq="W-FRI")):
        if dates.empty:
            continue
        grouped[week_end.strftime("%Y-%m-%d")] = [
            d.strftime("%Y-%m-%d") for d in sorted(dates.index)
        ]
    return grouped
