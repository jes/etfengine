from __future__ import annotations

import calendar
import math
import random
from datetime import date

import numpy as np

from strategy.constants import (
    OPTIMISE_GREEDY_ROUNDS,
    OPTIMISE_GREEDY_STEPS,
    OPTIMIZER_SEED,
    RISK_FREE_ID,
    WEEKS_PER_YEAR,
)
from strategy.data import Asset, Universe, window_dates


def window_start_from_end(end_iso: str, *, months: int) -> str:
    end = date.fromisoformat(end_iso)
    month = end.month - months
    year = end.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(end.day, calendar.monthrange(year, month)[1])
    return date(year, month, day).isoformat()


def asset_traded_before_window_start(asset: Asset, window_start: str) -> bool:
    return bool(asset.first_date and asset.first_date < window_start)


def eligible_assets(
    universe: Universe,
    tickers: list[str],
    window_start: str,
    dates: list[str],
    *,
    min_coverage_fraction: float,
) -> list[str]:
    if not dates:
        return []
    min_required = len(dates) * min_coverage_fraction
    eligible: list[str] = []
    for market_id in tickers:
        if market_id == RISK_FREE_ID:
            continue
        asset = universe.assets[market_id]
        if not asset_traded_before_window_start(asset, window_start):
            continue
        if min_coverage_fraction > 0:
            available = sum(1 for d in dates if d in asset.returns_by_date)
            if available < min_required:
                continue
        eligible.append(market_id)
    return eligible


def normalize_weights(weights: list[float]) -> list[float]:
    safe = [max(0.0, w) for w in weights]
    total = sum(safe)
    if total <= 0:
        equal = 1.0 / len(safe)
        return [equal for _ in safe]
    return [w / total for w in safe]


def cap_weights(weights: list[float], max_weight: float) -> list[float]:
    n = len(weights)
    if n == 0:
        return []
    if max_weight >= 1.0 - 1e-12:
        return normalize_weights(weights)
    if n * max_weight < 1.0 - 1e-12:
        return [1.0 / n for _ in range(n)]

    w = normalize_weights(weights)
    for _ in range(128):
        over_idx = [i for i, value in enumerate(w) if value > max_weight + 1e-12]
        if not over_idx:
            break
        excess = sum(w[i] - max_weight for i in over_idx)
        for i in over_idx:
            w[i] = max_weight
        recipients = [i for i in range(n) if w[i] < max_weight - 1e-12]
        if not recipients:
            return [1.0 / n for _ in range(n)]
        recipient_sum = sum(w[i] for i in recipients)
        if recipient_sum <= 1e-12:
            share = excess / len(recipients)
            for i in recipients:
                w[i] = min(max_weight, w[i] + share)
        else:
            for i in recipients:
                w[i] = min(max_weight, w[i] + excess * (w[i] / recipient_sum))
        w = normalize_weights(w)
    return w


def random_simplex(n: int, rng: random.Random) -> list[float]:
    draws = [-math.log(max(1e-12, rng.random())) for _ in range(n)]
    total = sum(draws)
    return [d / total for d in draws]


def shift_weight(
    weights: list[float],
    asset_index: int,
    delta: float,
    max_weight: float | None,
) -> list[float] | None:
    if delta == 0:
        return cap_weights(weights[:], max_weight) if max_weight is not None else weights[:]
    w = weights[asset_index]
    if max_weight is not None and delta > 0 and w >= max_weight - 1e-12:
        return None
    other_sum = 1.0 - w
    if delta > 0 and other_sum <= 1e-12:
        return None
    if delta < 0 and w + delta < -1e-12:
        return None

    next_weights = weights[:]
    next_weights[asset_index] = w + delta
    if other_sum > 1e-12:
        for j in range(len(weights)):
            if j == asset_index:
                continue
            next_weights[j] -= delta * (weights[j] / other_sum)
    if any(value < -1e-9 for value in next_weights):
        return None
    result = normalize_weights(next_weights)
    if max_weight is not None:
        result = cap_weights(result, max_weight)
    return result


def build_window_matrices(
    tickers: list[str],
    dates: list[str],
    universe: Universe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns = np.zeros((len(dates), len(tickers)), dtype=float)
    mask = np.zeros((len(dates), len(tickers)), dtype=float)
    for i, iso_date in enumerate(dates):
        for j, market_id in enumerate(tickers):
            value = universe.assets[market_id].returns_by_date.get(iso_date)
            if value is None:
                continue
            returns[i, j] = value
            mask[i, j] = 1.0
    rf = np.array(
        [
            universe.assets[RISK_FREE_ID].returns_by_date.get(iso_date, 0.0)
            for iso_date in dates
        ],
        dtype=float,
    )
    return returns, mask, rf


def evaluate_sharpe(
    weights: np.ndarray,
    window_returns: np.ndarray,
    window_mask: np.ndarray,
    rf_weekly: np.ndarray,
) -> float:
    weighted = window_mask * weights
    denom = weighted.sum(axis=1)
    safe = np.where(denom > 0, denom, 1.0)
    weekly_weights = weighted / safe[:, None]
    port = np.sum(weekly_weights * window_returns, axis=1)
    avg = float(np.mean(port))
    avg_rf = float(np.mean(rf_weekly))
    sigma = float(np.std(port, ddof=1)) if len(port) > 1 else 0.0
    if sigma == 0:
        return float("-inf")
    return ((avg - avg_rf) / sigma) * math.sqrt(WEEKS_PER_YEAR)


def greedy_refine_sharpe(
    start_weights: list[float],
    window_returns: np.ndarray,
    window_mask: np.ndarray,
    rf_weekly: np.ndarray,
    max_weight: float | None,
) -> tuple[list[float], float]:
    array = cap_weights(start_weights[:], max_weight) if max_weight is not None else start_weights[:]
    score = evaluate_sharpe(np.array(array, dtype=float), window_returns, window_mask, rf_weekly)
    n = len(array)

    for round_idx in range(OPTIMISE_GREEDY_ROUNDS):
        step = (
            OPTIMISE_GREEDY_STEPS[round_idx]
            if round_idx < len(OPTIMISE_GREEDY_STEPS)
            else OPTIMISE_GREEDY_STEPS[-1]
        )
        for asset_index in range(n):
            baseline = score
            up_trial = shift_weight(array, asset_index, step, max_weight)
            down_trial = shift_weight(array, asset_index, -step, max_weight)
            up_score = (
                evaluate_sharpe(np.array(up_trial, dtype=float), window_returns, window_mask, rf_weekly)
                if up_trial is not None
                else float("-inf")
            )
            down_score = (
                evaluate_sharpe(np.array(down_trial, dtype=float), window_returns, window_mask, rf_weekly)
                if down_trial is not None
                else float("-inf")
            )
            if up_score > baseline + 1e-12:
                array = up_trial
                score = up_score
            elif down_score > baseline + 1e-12:
                array = down_trial
                score = down_score
    return array, score


def optimize_sharpe_window(
    tickers: list[str],
    dates: list[str],
    universe: Universe,
    *,
    max_weight: float | None,
    random_trials: int,
    rng: random.Random | None = None,
) -> tuple[dict[str, float], float]:
    if not tickers:
        return {}, float("-inf")
    rng = rng or random.Random(OPTIMIZER_SEED)
    window_returns, window_mask, rf_weekly = build_window_matrices(tickers, dates, universe)
    n = len(tickers)

    candidates: list[list[float]] = [[1.0 / n for _ in range(n)]]
    for i in range(n):
        one_hot = [0.0 for _ in range(n)]
        one_hot[i] = 1.0
        candidates.append(one_hot)
    for _ in range(random_trials):
        candidates.append(random_simplex(n, rng))
    if max_weight is not None:
        candidates = [cap_weights(candidate, max_weight) for candidate in candidates]

    best_weights = candidates[0]
    best_score = float("-inf")
    for candidate in candidates:
        refined, score = greedy_refine_sharpe(
            candidate, window_returns, window_mask, rf_weekly, max_weight
        )
        if score > best_score:
            best_score = score
            best_weights = refined
    return {market_id: best_weights[i] for i, market_id in enumerate(tickers)}, best_score


def build_weekly_end_dates(universe: Universe, *, lookback_months: int) -> list[str]:
    data_start = universe.weekly_dates[0]
    end_dates: list[str] = []
    for iso_date in universe.weekly_dates:
        start = window_start_from_end(iso_date, months=lookback_months)
        if start < data_start:
            continue
        end_dates.append(iso_date)
    return end_dates


def build_optimal_weight_schedule(
    universe: Universe,
    tickers: list[str],
    *,
    lookback_months: int,
    max_weight: float,
    random_trials: int,
    min_coverage_fraction: float,
) -> tuple[list[str], list[dict[str, float]]]:
    end_dates = build_weekly_end_dates(universe, lookback_months=lookback_months)
    rng = random.Random(OPTIMIZER_SEED)
    weight_rows: list[dict[str, float]] = []

    for end_date in end_dates:
        start_date = window_start_from_end(end_date, months=lookback_months)
        dates = window_dates(universe, start_date, end_date)
        if len(dates) < 2:
            weight_rows.append({})
            continue

        eligible = eligible_assets(
            universe,
            tickers,
            start_date,
            dates,
            min_coverage_fraction=min_coverage_fraction,
        )
        if not eligible:
            weight_rows.append({})
            continue

        optimal, _score = optimize_sharpe_window(
            eligible,
            dates,
            universe,
            max_weight=max_weight,
            random_trials=random_trials,
            rng=rng,
        )
        weight_rows.append({market_id: optimal.get(market_id, 0.0) for market_id in tickers})
    return end_dates, weight_rows
