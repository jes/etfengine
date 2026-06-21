#!/usr/bin/env python3
"""Fetch the public InvestEngine shared-portfolio JSON and cache logo images."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import requests

from site_builder.investengine_portfolio import logo_cache_name

DEFAULT_API_URL = (
    "https://investengine.com/api/v0.33/public/shared_portfolio/"
    "5ca383e6593408b7dca1d4140abd788324418369/"
)
DEFAULT_JSON_DIR = Path("public/json")
DEFAULT_ICONS_DIR = Path("public/icons/cache")


def _iter_logo_urls(payload: dict) -> list[tuple[str, int | str, str]]:
    logos: list[tuple[str, int | str, str]] = []
    for equity in payload.get("equities") or []:
        logo = (equity.get("logo") or "").strip()
        if logo:
            logos.append(("equity", equity.get("id", "unknown"), logo))
    for security in payload.get("securities") or []:
        logo = (security.get("logo_uri") or "").strip()
        if logo:
            logos.append(("etf", security.get("id", "unknown"), logo))
    return logos


def cache_logo(
    *,
    icons_dir: Path,
    prefix: str,
    entity_id: int | str,
    logo_url: str,
    session: requests.Session,
) -> str | None:
    logo_url = logo_url.strip()
    if not logo_url:
        return None
    filename = logo_cache_name(prefix, entity_id, logo_url)
    target = icons_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return filename
    try:
        response = session.get(logo_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"warning: failed to fetch logo {logo_url}: {exc}", file=sys.stderr)
        return None
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "png" in content_type and not filename.endswith(".png"):
        filename = re.sub(r"\.[^.]+$", ".png", filename)
        target = icons_dir / filename
    icons_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return filename


def cache_portfolio_logos(
    payload: dict,
    *,
    icons_dir: Path,
    session: requests.Session | None = None,
) -> int:
    client = session or requests.Session()
    cached = 0
    for prefix, entity_id, logo_url in _iter_logo_urls(payload):
        if cache_logo(
            icons_dir=icons_dir,
            prefix=prefix,
            entity_id=entity_id,
            logo_url=logo_url,
            session=client,
        ):
            cached += 1
    return cached


def fetch_portfolio(
    *,
    api_url: str,
    json_dir: Path,
    icons_dir: Path,
    as_of: date | None = None,
    session: requests.Session | None = None,
) -> Path:
    client = session or requests.Session()
    response = client.get(api_url, timeout=60)
    response.raise_for_status()
    payload = response.json()

    json_dir.mkdir(parents=True, exist_ok=True)
    stamp = (as_of or date.today()).strftime("%Y%m%d")
    output_path = json_dir / f"{stamp}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    logo_count = cache_portfolio_logos(payload, icons_dir=icons_dir, session=client)
    print(f"Wrote {output_path}")
    print(f"Cached {logo_count} logo(s) under {icons_dir}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--icons-dir", type=Path, default=DEFAULT_ICONS_DIR)
    parser.add_argument(
        "--date",
        default=None,
        help="Date stamp for output filename YYYYMMDD (default: today)",
    )
    args = parser.parse_args()
    as_of: date | None = None
    if args.date:
        if len(args.date) == 8 and args.date.isdigit():
            as_of = date(
                int(args.date[0:4]),
                int(args.date[4:6]),
                int(args.date[6:8]),
            )
        else:
            as_of = date.fromisoformat(args.date)
    fetch_portfolio(
        api_url=args.api_url,
        json_dir=args.json_dir,
        icons_dir=args.icons_dir,
        as_of=as_of,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
