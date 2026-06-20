from __future__ import annotations

import bisect


def apply_min_unit_weight_floor(
    weights: dict[str, float],
    min_weight: float,
) -> dict[str, float]:
    """Drop sub-threshold weights and renormalize survivors to sum to 1."""
    if min_weight <= 0:
        return normalize_weight_dict(weights)
    kept = {market_id: weight for market_id, weight in weights.items() if weight >= min_weight}
    return normalize_weight_dict(kept)


def apply_min_capped_weight_floor(
    weights: dict[str, float],
    min_weight: float,
) -> dict[str, float]:
    """Drop sub-threshold weights without renormalizing cash away."""
    if min_weight <= 0:
        return {
            market_id: weight
            for market_id, weight in weights.items()
            if weight > 1e-12
        }
    return {
        market_id: weight
        for market_id, weight in weights.items()
        if weight >= min_weight
    }


def normalize_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    positive = {market_id: weight for market_id, weight in weights.items() if weight > 1e-12}
    total = sum(positive.values())
    if total <= 1e-12:
        return {}
    return {market_id: weight / total for market_id, weight in positive.items()}


def ewma_smooth_weight_rows(
    weight_rows: list[dict[str, float]],
    *,
    span: int,
) -> list[dict[str, float]]:
    """Causal EWMA of weight vectors; row i uses only rows 0..i."""
    if span < 1:
        raise ValueError("span must be >= 1")
    alpha = 2.0 / (span + 1.0)
    smoothed: list[dict[str, float]] = []
    state: dict[str, float] = {}

    for row in weight_rows:
        tickers = set(state) | set(row)
        new_state: dict[str, float] = {}
        for market_id in tickers:
            raw = row.get(market_id, 0.0)
            if not state:
                new_state[market_id] = raw
            else:
                prev = state.get(market_id, 0.0)
                new_state[market_id] = alpha * raw + (1.0 - alpha) * prev
        state = new_state
        smoothed.append(normalize_weight_dict(state))
    return smoothed


def ewma_smooth_capped_weight_rows(
    weight_rows: list[dict[str, float]],
    *,
    span: int,
    min_weight: float = 0.0,
) -> list[dict[str, float]]:
    """Causal EWMA of weight vectors; rows may sum to < 1 (remainder is cash)."""
    if span < 1:
        raise ValueError("span must be >= 1")
    alpha = 2.0 / (span + 1.0)
    smoothed: list[dict[str, float]] = []
    state: dict[str, float] = {}

    for row in weight_rows:
        tickers = set(state) | set(row)
        new_state: dict[str, float] = {}
        for market_id in tickers:
            raw = row.get(market_id, 0.0)
            if not state:
                new_state[market_id] = raw
            else:
                prev = state.get(market_id, 0.0)
                new_state[market_id] = alpha * raw + (1.0 - alpha) * prev
        total = sum(new_state.values())
        if total > 1.0 + 1e-12:
            scale = 1.0 / total
            new_state = {
                market_id: weight * scale for market_id, weight in new_state.items()
            }
        state = {
            market_id: weight
            for market_id, weight in new_state.items()
            if weight > 1e-12 and weight >= min_weight
        }
        smoothed.append(dict(state))
    return smoothed


def target_weights_for_date(
    end_dates: list[str],
    weight_rows: list[dict[str, float]],
    iso_date: str,
) -> dict[str, float] | None:
    """Latest schedule row with end_date strictly before iso_date."""
    idx = bisect.bisect_left(end_dates, iso_date) - 1
    if idx < 0:
        return None
    return weight_rows[idx]
