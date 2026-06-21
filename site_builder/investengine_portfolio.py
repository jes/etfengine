"""Parse InvestEngine shared-portfolio JSON for the static site."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from strategy.data import Universe


def logo_cache_name(prefix: str, entity_id: int | str, logo_url: str) -> str:
    digest = hashlib.sha256(logo_url.encode("utf-8")).hexdigest()[:16]
    path = urlparse(logo_url).path.lower()
    if path.endswith(".png"):
        ext = ".png"
    elif path.endswith(".jpg") or path.endswith(".jpeg"):
        ext = ".jpg"
    elif path.endswith(".webp"):
        ext = ".webp"
    else:
        ext = ".img"
    return f"{prefix}-{entity_id}-{digest}{ext}"


def clean_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


@dataclass(frozen=True)
class IeEquityRow:
    name: str
    weight_pct: float
    icon_path: str


@dataclass(frozen=True)
class IeRegionRow:
    name: str
    weight_pct: float
    color: str


@dataclass(frozen=True)
class IeEtfWeight:
    market_id: str | None
    ticker: str
    title: str
    weight_pct: float
    icon_path: str


@dataclass(frozen=True)
class InvestEngineSnapshot:
    fetched_date: str
    json_path: Path
    equity_holdings: list[IeEquityRow]
    region_breakdown: list[IeRegionRow]
    etf_weights_by_market_id: dict[str, float]
    etf_weights_by_ticker: dict[str, IeEtfWeight]
    unmapped_etfs: list[IeEtfWeight]


def _parse_weight(value: object) -> float:
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return float("nan")


def latest_portfolio_json(json_dir: Path, *, prefer_date: date | None = None) -> Path | None:
    if not json_dir.is_dir():
        return None
    if prefer_date is not None:
        preferred = json_dir / f"{prefer_date:%Y%m%d}.json"
        if preferred.is_file():
            return preferred
    files = sorted(json_dir.glob("*.json"), reverse=True)
    return files[0] if files else None


def ticker_to_market_id(universe: Universe) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for market_id, asset in universe.assets.items():
        raw = (asset.yahoo_ticker or market_id).strip()
        variants = {raw, raw.removesuffix(".L"), raw.removesuffix(".l")}
        for variant in variants:
            token = clean_token(variant)
            if token:
                mapping[token] = market_id
    return mapping


def _copy_icon(
    *,
    icons_cache_dir: Path,
    snapshot_icons_dir: Path,
    prefix: str,
    entity_id: int | str,
    logo_url: str,
) -> str:
    logo_url = (logo_url or "").strip()
    if not logo_url:
        return ""
    filename = logo_cache_name(prefix, entity_id, logo_url)
    source = icons_cache_dir / filename
    if not source.is_file():
        return ""
    snapshot_icons_dir.mkdir(parents=True, exist_ok=True)
    target = snapshot_icons_dir / filename
    if not target.is_file():
        shutil.copy2(source, target)
    return f"icons/{filename}"


def load_investengine_snapshot(
    json_path: Path,
    *,
    universe: Universe,
    icons_cache_dir: Path,
    snapshot_icons_dir: Path,
    top_equities: int = 20,
) -> InvestEngineSnapshot:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    ticker_map = ticker_to_market_id(universe)

    equities = sorted(
        payload.get("equities") or [],
        key=lambda row: _parse_weight(row.get("target_weight")),
        reverse=True,
    )
    equity_rows: list[IeEquityRow] = []
    for row in equities[:top_equities]:
        weight = _parse_weight(row.get("target_weight"))
        if weight != weight or weight <= 0:
            continue
        icon_path = _copy_icon(
            icons_cache_dir=icons_cache_dir,
            snapshot_icons_dir=snapshot_icons_dir,
            prefix="equity",
            entity_id=row.get("id", "unknown"),
            logo_url=row.get("logo") or "",
        )
        equity_rows.append(
            IeEquityRow(
                name=str(row.get("name") or "").strip() or "Unknown",
                weight_pct=weight,
                icon_path=icon_path,
            )
        )

    regions = sorted(
        payload.get("regions") or [],
        key=lambda row: _parse_weight(row.get("target_weight")),
        reverse=True,
    )
    region_rows = [
        IeRegionRow(
            name=str(row.get("name") or "").strip() or "Unknown",
            weight_pct=_parse_weight(row.get("target_weight")),
            color=str(row.get("color") or "#888888"),
        )
        for row in regions
        if _parse_weight(row.get("target_weight")) == _parse_weight(row.get("target_weight"))
        and _parse_weight(row.get("target_weight")) > 0
    ]

    etf_by_market: dict[str, float] = {}
    etf_by_ticker: dict[str, IeEtfWeight] = {}
    unmapped: list[IeEtfWeight] = []
    for row in payload.get("securities") or []:
        weight = _parse_weight(row.get("target_weight"))
        if weight != weight or weight <= 0:
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        title = str(row.get("title") or ticker).strip()
        icon_path = _copy_icon(
            icons_cache_dir=icons_cache_dir,
            snapshot_icons_dir=snapshot_icons_dir,
            prefix="etf",
            entity_id=row.get("id", "unknown"),
            logo_url=row.get("logo_uri") or "",
        )
        market_id = ticker_map.get(clean_token(ticker))
        etf = IeEtfWeight(
            market_id=market_id,
            ticker=ticker,
            title=title,
            weight_pct=weight,
            icon_path=icon_path,
        )
        etf_by_ticker[ticker] = etf
        if market_id:
            etf_by_market[market_id] = weight
        else:
            unmapped.append(etf)

    fetched_date = json_path.stem
    if len(fetched_date) == 8 and fetched_date.isdigit():
        fetched_date = f"{fetched_date[0:4]}-{fetched_date[4:6]}-{fetched_date[6:8]}"

    return InvestEngineSnapshot(
        fetched_date=fetched_date,
        json_path=json_path,
        equity_holdings=equity_rows,
        region_breakdown=region_rows,
        etf_weights_by_market_id=etf_by_market,
        etf_weights_by_ticker=etf_by_ticker,
        unmapped_etfs=sorted(unmapped, key=lambda row: row.weight_pct, reverse=True),
    )


def ie_icons_by_market_id(snapshot: InvestEngineSnapshot) -> dict[str, str]:
    return {
        etf.market_id: etf.icon_path
        for etf in snapshot.etf_weights_by_ticker.values()
        if etf.market_id and etf.icon_path
    }


def append_ie_only_allocations(
    rows: list,
    *,
    universe: Universe,
    ie_snapshot: InvestEngineSnapshot | None,
    risk_free_id: str,
) -> list:
    if ie_snapshot is None:
        return rows
    from site_builder.etf_data import AllocationRow, market_label

    present = {row.market_id for row in rows}
    extra: list[AllocationRow] = []
    icons = ie_icons_by_market_id(ie_snapshot)
    for market_id, weight in ie_snapshot.etf_weights_by_market_id.items():
        if market_id in present or market_id == risk_free_id:
            continue
        extra.append(
            AllocationRow(
                market_id=market_id,
                label=market_label(universe, market_id),
                weight_pct=0.0,
                spark_path="",
                return_1y=None,
                ie_weight_pct=weight,
                icon_path=icons.get(market_id, ""),
            )
        )
    for etf in ie_snapshot.unmapped_etfs:
        extra.append(
            AllocationRow(
                market_id=f"ie:{etf.ticker}",
                label=f"{etf.ticker} — {etf.title}",
                weight_pct=0.0,
                spark_path="",
                return_1y=None,
                ie_weight_pct=etf.weight_pct,
                icon_path=etf.icon_path,
            )
        )
    merged = list(rows) + extra
    return sorted(
        merged,
        key=lambda row: max(row.ie_weight_pct or 0.0, row.weight_pct),
        reverse=True,
    )
