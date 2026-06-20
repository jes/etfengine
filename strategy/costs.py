from __future__ import annotations

from strategy.constants import RISK_FREE_ID
from strategy.data import Universe


def within_relative_drift_band(current: float, target: float, drift_band: float) -> bool:
    """True when current is within drift_band relative deviation of target (e.g. 0.10 = ±10%)."""
    if target <= 1e-12:
        return current <= 1e-12
    return abs(current - target) / target <= drift_band


def apply_rebalance_drift_band(
    current: dict[str, float],
    target: dict[str, float],
    *,
    drift_band: float,
) -> dict[str, float]:
    """Keep current weight when within drift_band of target, else move to target."""
    if not current:
        return dict(target)
    market_ids = set(current) | set(target)
    return {
        market_id: (
            current.get(market_id, 0.0)
            if within_relative_drift_band(
                current.get(market_id, 0.0),
                target.get(market_id, 0.0),
                drift_band,
            )
            else target.get(market_id, 0.0)
        )
        for market_id in market_ids
    }


def resolve_effective_weights(
    prev_effective: dict[str, float],
    target_effective: dict[str, float],
    *,
    drift_band: float | None,
    regime_changed: bool,
) -> dict[str, float]:
    if regime_changed or not prev_effective:
        actual = target_effective
    elif drift_band is None:
        actual = target_effective
    else:
        actual = apply_rebalance_drift_band(
            prev_effective,
            target_effective,
            drift_band=drift_band,
        )
    return {market_id: weight for market_id, weight in actual.items() if weight > 1e-12}


def gross_return_from_effective(
    effective: dict[str, float],
    universe: Universe,
    iso_date: str,
    *,
    use_daily: bool = False,
) -> float:
    total = 0.0
    for market_id, weight in effective.items():
        asset = universe.assets[market_id]
        returns = asset.daily_returns_by_date if use_daily else asset.returns_by_date
        if iso_date in returns:
            total += weight * returns[iso_date]
    return total


def net_return_from_effective(
    effective: dict[str, float],
    universe: Universe,
    iso_date: str,
    *,
    use_daily: bool = False,
    charge_funding: bool,
) -> float:
    if not effective:
        return 0.0
    gross = gross_return_from_effective(
        effective, universe, iso_date, use_daily=use_daily
    )
    if not charge_funding:
        return gross
    rf_returns = (
        universe.assets[RISK_FREE_ID].daily_returns_by_date
        if use_daily
        else universe.assets[RISK_FREE_ID].returns_by_date
    )
    rf = rf_returns.get(iso_date, 0.0)
    leverage = sum(effective.values())
    return gross - leverage * rf


def return_after_spread_drag(net_return: float, spread_drag: float) -> float:
    """Combine spread drag (pre-return) with period return into one net figure."""
    if spread_drag <= 0:
        return net_return
    return (1.0 - spread_drag) * (1.0 + net_return) - 1.0


def rebalance_spread_drag(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    spread_fraction: dict[str, float],
) -> float:
    """
    IG-style spread cost: half round-trip spread on buys and half on sells.

    spread_fraction values are decimals (e.g. 0.00005 for 0.005%).
    Returns drag as a fraction of equity (e.g. 0.001 = 0.1%).
    """
    if not old_weights and not new_weights:
        return 0.0
    market_ids = set(old_weights) | set(new_weights)
    cost = 0.0
    for market_id in market_ids:
        old_w = old_weights.get(market_id, 0.0)
        new_w = new_weights.get(market_id, 0.0)
        spread = spread_fraction.get(market_id, 0.0)
        if spread <= 0:
            continue
        sold = max(0.0, old_w - new_w)
        bought = max(0.0, new_w - old_w)
        cost += spread * 0.5 * sold + spread * 0.5 * bought
    return cost
