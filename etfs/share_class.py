"""Optional filter: drop distributing share classes when an accumulating line exists.

Dist and acc are not economic substitutes — distributing lines pay out income,
which can lower path volatility even when long-run total return is similar.
This filter is for A/B comparison only, not a recommended default.
"""

from __future__ import annotations

import re

# Share-class suffixes (not strategy words like "Equity Income").
_ACC_SUFFIXES = (
    r"\s+accumulating\s*$",
    r"\s+\(acc\)\s*$",
    r"\s+acc-usd\s*$",
    r"\s+acc\)\s*$",  # e.g. "EUR (acc)"
)

_DIST_SUFFIXES = (
    r"\s+distributing\s*$",
    r"\s+\(dist\)\s*$",
    r"\s+dist\s*$",
    r"\s+inc-usd\s*$",
    r"\s+income 1d\s*$",
    r"\s+\(dist\)\s*$",
    r"\s+\(eur\)\s+distributing\s*$",
    r"\s+\(gbp\)\s+distributing\s*$",
    r"\s+\(usd\)\s+distributing\s*$",
)

_ACC_RES = [re.compile(p, re.IGNORECASE) for p in _ACC_SUFFIXES]
_DIST_RES = [re.compile(p, re.IGNORECASE) for p in _DIST_SUFFIXES]
_ALL_RES = _ACC_RES + _DIST_RES


def classify_share_class(market_name: str) -> str | None:
    """Return 'acc', 'dist', or None when share class is not explicit."""
    name = market_name.strip()
    for pattern in _ACC_RES:
        if pattern.search(name):
            return "acc"
    for pattern in _DIST_RES:
        if pattern.search(name):
            return "dist"
    if re.search(r"\(acc\)\s*$", name, re.IGNORECASE):
        return "acc"
    if re.search(r"\(dist\)\s*$", name, re.IGNORECASE):
        return "dist"
    return None


def normalize_fund_name(market_name: str) -> str:
    """Fund identity with share-class suffixes removed."""
    name = market_name.strip()
    for pattern in _ALL_RES:
        name = pattern.sub("", name)
    name = re.sub(r"\s+\(acc\)\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+\(dist\)\s*$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def distributing_ids_when_accumulating_exists(
    market_names: dict[str, str],
    *,
    candidate_ids: set[str] | None = None,
) -> set[str]:
    """IDs to exclude: explicit dist/income lines where an acc line shares the same fund name."""
    ids = candidate_ids if candidate_ids is not None else set(market_names)
    by_base: dict[str, dict[str, list[str]]] = {}
    for market_id in ids:
        name = market_names.get(market_id, market_id)
        share_class = classify_share_class(name)
        if share_class is None:
            continue
        base = normalize_fund_name(name)
        by_base.setdefault(base, {"acc": [], "dist": []})
        by_base[base][share_class].append(market_id)

    excluded: set[str] = set()
    for base, classes in by_base.items():
        if not classes["acc"] or not classes["dist"]:
            continue
        excluded.update(classes["dist"])
    return excluded


def accumulating_only_ids(
    market_names: dict[str, str],
    *,
    candidate_ids: set[str],
) -> set[str]:
    """Candidate ids minus distributing share classes superseded by an accumulating line."""
    return candidate_ids - distributing_ids_when_accumulating_exists(
        market_names,
        candidate_ids=candidate_ids,
    )
