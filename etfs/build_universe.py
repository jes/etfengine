#!/usr/bin/env python3
"""Build etfs/markets.csv from the justETF screener (LSE-listed, UK retail).

justETF is the best free source for a wide UCITS universe: filter by exchange=XLON
(London), local_country=GB (UK investor view), and strategy=long-only (no inverse/
leveraged products). See etfs/README.md for other sources.

Requires: pip install git+https://github.com/druzsan/justetf-scraping.git
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

ETFS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ETFS_DIR / "markets.csv"
DEFAULT_CACHE = ETFS_DIR / "_sources" / "justetf_xlon_gb_longonly.csv"
DEFAULT_INVESTENGINE_ALLOWLIST = ETFS_DIR / "_sources" / "investengine_allowlist.csv"

RISK_FREE_ROW = {
    "id": "us-30-day-fed-funds-rate",
    "market_name": "US 30-Day Fed Funds Rate",
    "category": "BONDS",
    "spread_pct": "0.01",
    "yahoo_ticker": "ZQ=F",
    "yahoo_map_source": "manual:risk_free",
}

# Placeholder one-way bid–offer tiers until we have broker quotes.
DEFAULT_SPREAD_BY_CATEGORY = {
    "ETF": 0.10,
    "BOND": 0.08,
    "ETC": 0.20,
    "ETN": 0.15,
    "OTHER": 0.15,
}

ASSET_CLASS_TO_CATEGORY = {
    "Equity": "ETF",
    "Bonds": "BOND",
    "Commodities": "ETC",
    "Precious Metals": "ETC",
    "Real Estate": "ETF",
    "Money Market": "BOND",
    "Cryptocurrencies": "ETN",
}


def slugify_isin(isin: str) -> str:
    return isin.strip().lower()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def instrument_category(row: pd.Series) -> str:
    asset = clean_text(row.get("asset_class"))
    instrument = clean_text(row.get("instrument")).upper()
    if instrument == "ETC":
        return "ETC"
    if instrument == "ETN":
        return "ETN"
    mapped = ASSET_CLASS_TO_CATEGORY.get(asset)
    if mapped:
        return mapped
    if asset == "Bonds":
        return "BOND"
    if asset in {"Commodities", "Precious Metals"}:
        return "ETC"
    if asset == "Equity":
        return "ETF"
    return "OTHER"


def fetch_justetf_xlon(*, refresh: bool, cache_path: Path) -> pd.DataFrame:
    if cache_path.is_file() and not refresh:
        df = pd.read_csv(cache_path, index_col=0)
        print(f"loaded cache: {cache_path} ({len(df)} rows)")
        return df

    try:
        import justetf_scraping as j
    except ImportError as exc:
        raise SystemExit(
            "justetf-scraping is required:\n"
            "  pip install git+https://github.com/druzsan/justetf-scraping.git"
        ) from exc

    print("fetching justETF (XLON, GB, long-only) — takes ~90s…")
    df = j.load_overview(
        strategy="epg-longOnly",
        exchange="XLON",
        local_country="GB",
        enrich=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"cached: {cache_path} ({len(df)} rows)")
    return df


def filter_universe(
    df: pd.DataFrame,
    *,
    min_size_meur: float,
    dividends: str,
    exclude_instruments: set[str],
    allowlist: set[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    if min_size_meur > 0 and "size" in out.columns:
        out = out[out["size"].fillna(0) >= min_size_meur]
    if dividends != "any" and "dividends" in out.columns:
        out = out[out["dividends"].astype(str).str.lower() == dividends]
    if exclude_instruments and "instrument" in out.columns:
        out = out[~out["instrument"].isin(exclude_instruments)]
    out = out[out["ticker"].notna() & (out["ticker"].astype(str).str.len() > 0)]
    if allowlist is not None:
        out = out[
            [
                isin_or_ticker_allowed(str(isin), row, allowlist)
                for isin, row in out.iterrows()
            ]
        ]
    return out.sort_values(["size", "name"], ascending=[False, True], na_position="last")


def clean_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def load_identifier_allowlist(path: Path) -> set[str]:
    """Load ISINs/tickers from a broker export or simple one-column CSV."""
    if not path.is_file():
        raise SystemExit(f"allowlist file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
        if not rows:
            return set()
        header = [cell.strip() for cell in rows[0]]
        known_headers = {
            "id",
            "isin",
            "ticker",
            "symbol",
            "yahoo_ticker",
        }
        has_header = any(cell.lower() in known_headers for cell in header)
        if has_header:
            handle.seek(0)
            reader = csv.DictReader(handle)
            tokens: set[str] = set()
            for row in reader:
                for key in (
                    "id",
                    "isin",
                    "ISIN",
                    "ticker",
                    "Ticker",
                    "symbol",
                    "Symbol",
                    "yahoo_ticker",
                ):
                    if key in row and row[key]:
                        tokens.add(clean_token(row[key]))
            return {token for token in tokens if token}

        return {
            clean_token(cell)
            for row in rows
            for cell in row
            if clean_token(cell)
        }


def isin_or_ticker_allowed(isin: str, row: pd.Series, allowlist: set[str]) -> bool:
    ticker = str(row.get("ticker") or "")
    return clean_token(isin) in allowlist or clean_token(ticker) in allowlist


def row_to_market(row: pd.Series, *, isin: str) -> dict[str, str]:
    ticker = clean_text(row["ticker"])
    category = instrument_category(row)
    spread = DEFAULT_SPREAD_BY_CATEGORY.get(category, 0.15)
    name = re.sub(r"\s+", " ", clean_text(row["name"]))
    return {
        "id": slugify_isin(isin),
        "market_name": name,
        "category": category,
        "spread_pct": f"{spread:.2f}",
        "yahoo_ticker": f"{ticker}.L",
        "yahoo_map_source": f"justetf:XLON:{isin}",
        "dividends": clean_text(row.get("dividends")),
        "ter_pct": clean_text(row.get("ter")),
        "size_meur": clean_text(row.get("size")),
        "currency": clean_text(row.get("currency")),
        "domicile_country": clean_text(row.get("domicile_country")),
        "hedged": clean_text(row.get("hedged")),
    }


def write_markets_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "id",
        "market_name",
        "category",
        "spread_pct",
        "yahoo_ticker",
        "yahoo_map_source",
        "dividends",
        "ter_pct",
        "size_meur",
        "currency",
        "domicile_country",
        "hedged",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output markets CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"justETF response cache (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download from justETF instead of using cache",
    )
    parser.add_argument(
        "--min-size-meur",
        type=float,
        default=None,
        help=(
            "Minimum fund size in EUR millions "
            "(default: 0 with InvestEngine allowlist, else 500)"
        ),
    )
    parser.add_argument(
        "--dividends",
        choices=["any", "accumulating", "distributing"],
        default="any",
        help="Dividend policy filter (default: any — acc and dist for IE ISA)",
    )
    parser.add_argument(
        "--exclude-instrument",
        action="append",
        default=[],
        choices=["ETN", "ETC", "ETF"],
        help="Drop instrument types (repeatable)",
    )
    parser.add_argument(
        "--investengine-allowlist",
        type=Path,
        default=DEFAULT_INVESTENGINE_ALLOWLIST
        if DEFAULT_INVESTENGINE_ALLOWLIST.is_file()
        else None,
        help=(
            "CSV of InvestEngine-available ISINs or tickers "
            f"(default: {DEFAULT_INVESTENGINE_ALLOWLIST} when present)"
        ),
    )
    args = parser.parse_args()

    df = fetch_justetf_xlon(refresh=args.refresh, cache_path=args.cache)
    allowlist = (
        load_identifier_allowlist(args.investengine_allowlist)
        if args.investengine_allowlist is not None
        else None
    )
    min_size_meur = (
        args.min_size_meur
        if args.min_size_meur is not None
        else (0.0 if allowlist is not None else 500.0)
    )
    filtered = filter_universe(
        df,
        min_size_meur=min_size_meur,
        dividends=args.dividends,
        exclude_instruments=set(args.exclude_instrument),
        allowlist=allowlist,
    )

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for isin, row in filtered.iterrows():
        market = row_to_market(row, isin=str(isin))
        if market["id"] in seen_ids:
            continue
        seen_ids.add(market["id"])
        rows.append(market)

    rows.append(dict(RISK_FREE_ROW))
    write_markets_csv(args.output, rows)

    cats = pd.Series(r["category"] for r in rows).value_counts()
    print(f"wrote {args.output}: {len(rows)} rows ({len(rows) - 1} ETFs + risk-free)")
    if allowlist is not None:
        print(f"allowlist identifiers: {len(allowlist)}")
    print(cats.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
