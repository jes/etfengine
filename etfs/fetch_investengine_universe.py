#!/usr/bin/env python3
"""Fetch InvestEngine's public ETF universe into a CSV allowlist."""

from __future__ import annotations

import argparse
import csv
import html
import http.client
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ETFS_DIR = Path(__file__).resolve().parent
DEFAULT_URL = "https://investengine.com/etfs/all/"
DEFAULT_OUTPUT = ETFS_DIR / "_sources" / "investengine_allowlist.csv"


def fetch_html(url: str) -> str:
    if shutil.which("curl"):
        try:
            proc = subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--compressed",
                    "-A",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    url,
                ],
                capture_output=True,
                check=True,
                timeout=120,
            )
            page = proc.stdout.decode("utf-8", errors="replace")
            if "__NEXT_DATA__" in page:
                return page
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            pass

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept-Encoding": "identity",
        },
    )
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read().decode("utf-8", errors="replace")
        except (TimeoutError, http.client.IncompleteRead, urllib.error.URLError) as exc:
            last_error = exc
    raise RuntimeError(f"failed to fetch {url}") from last_error


def extract_next_data(page_html: str) -> dict:
    marker = '<script id="__NEXT_DATA__" type="application/json">'
    start = page_html.find(marker)
    if start == -1:
        raise ValueError("InvestEngine page did not contain __NEXT_DATA__")
    json_start = start + len(marker)
    try:
        data, _end = json.JSONDecoder().raw_decode(page_html, json_start)
    except json.JSONDecodeError as exc:
        raise ValueError("InvestEngine __NEXT_DATA__ payload was truncated or invalid") from exc
    return data


def extract_securities(next_data: dict) -> list[dict]:
    try:
        securities = next_data["props"]["pageProps"]["defaultSecurities"]
    except KeyError as exc:
        raise ValueError("InvestEngine payload did not contain defaultSecurities") from exc
    if not isinstance(securities, list):
        raise ValueError("InvestEngine defaultSecurities payload is not a list")
    return securities


def visible_tradable(security: dict) -> bool:
    return (
        bool(security.get("is_visible_in_universe"))
        and bool(security.get("is_trading_available"))
        and not bool(security.get("is_sell_only"))
    )


def filter_securities(
    securities: list[dict],
    *,
    dividends: str,
) -> list[dict]:
    out = [security for security in securities if visible_tradable(security)]
    if dividends != "any":
        expected = dividends.upper()
        out = [
            security
            for security in out
            if str(security.get("dividends_type") or "").upper() == expected
        ]
    return out


def write_allowlist(path: Path, securities: list[dict]) -> None:
    fieldnames = [
        "isin",
        "ticker",
        "title",
        "dividends_type",
        "type",
        "ter",
        "estimated_yield",
        "provider",
        "fund_size_mm",
        "share_class_size_mm",
        "is_hedged",
        "base_currency",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for security in securities:
            properties = security.get("properties") or {}
            writer.writerow(
                {
                    "isin": security.get("isin") or "",
                    "ticker": security.get("ticker") or "",
                    "title": security.get("title") or "",
                    "dividends_type": security.get("dividends_type") or "",
                    "type": security.get("type") or "",
                    "ter": security.get("ter") or "",
                    "estimated_yield": security.get("estimated_yield") or "",
                    "provider": security.get("provider_filter_name") or "",
                    "fund_size_mm": properties.get("fund_size_mm") or "",
                    "share_class_size_mm": properties.get("share_class_size_mm") or "",
                    "is_hedged": security.get("is_hedged"),
                    "base_currency": security.get("base_currency") or "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dividends",
        choices=["any", "accumulating", "distributing"],
        default="any",
        help="Dividend policy to export (default: any — acc and dist for ISA)",
    )
    args = parser.parse_args()

    page = fetch_html(args.url)
    securities = extract_securities(extract_next_data(page))
    filtered = filter_securities(securities, dividends=args.dividends)
    write_allowlist(args.output, filtered)

    print(
        f"wrote {args.output}: {len(filtered)} {args.dividends} ETFs "
        f"from {len(securities)} InvestEngine securities"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
