"""Vol-capped mean-return optimiser (weekly lookback) for ETF walk-forward schedule."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import numpy as np

from strategy.constants import OPTIMIZER_SEED, WEEKS_PER_YEAR
from strategy.data import Universe, window_dates
from strategy.optimizer import (
    build_window_matrices,
    build_weekly_end_dates,
    eligible_assets,
    window_start_from_end,
)

RANDOM_SET_TRIALS = 8
DEFAULT_MIN_WEIGHT = 0.05
REBALANCE_FREQUENCIES = ("weekly", "monthly", "quarterly", "annual")
DEFAULT_EWMA_SPAN_BY_FREQUENCY = {
    "monthly": 6,
    "weekly": 24,
    "quarterly": 4,
    "annual": 2,
}


def default_ewma_span(rebalance_frequency: str) -> int:
    try:
        return DEFAULT_EWMA_SPAN_BY_FREQUENCY[rebalance_frequency]
    except KeyError as exc:
        raise ValueError(f"unsupported rebalance frequency: {rebalance_frequency}") from exc


def listing_cutoff_from_end(end_date: str, *, listing_years: float) -> str:
    dt = date.fromisoformat(end_date)
    return (dt - timedelta(days=int(round(listing_years * 365.25)))).isoformat()


def filter_listing_age(
    universe: Universe,
    market_ids: list[str],
    listing_cutoff: str,
) -> list[str]:
    return [
        market_id
        for market_id in market_ids
        if universe.assets[market_id].first_date
        and universe.assets[market_id].first_date <= listing_cutoff
    ]


def filter_bad_ticks(
    universe: Universe,
    market_ids: list[str],
    dates: list[str],
    *,
    max_abs_daily_return: float,
) -> list[str]:
    if max_abs_daily_return <= 0:
        return market_ids
    kept: list[str] = []
    for market_id in market_ids:
        asset = universe.assets[market_id]
        ok = True
        for iso_date in dates:
            value = asset.daily_returns_by_date.get(iso_date)
            if value is not None and abs(value) > max_abs_daily_return:
                ok = False
                break
        if ok:
            kept.append(market_id)
    return kept


def rebalance_period_key(iso_date: str, frequency: str) -> tuple[int, int]:
    dt = date.fromisoformat(iso_date)
    if frequency == "weekly":
        return dt.isocalendar().year, dt.isocalendar().week
    if frequency == "monthly":
        return dt.year, dt.month
    if frequency == "quarterly":
        return dt.year, (dt.month - 1) // 3
    if frequency == "annual":
        return dt.year, 0
    raise ValueError(f"unsupported rebalance frequency: {frequency}")


def select_rebalance_end_dates(end_dates: list[str], frequency: str) -> list[str]:
    """Keep the last available weekly date in each rebalance period."""
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(f"unsupported rebalance frequency: {frequency}")
    selected: list[str] = []
    for iso_date in end_dates:
        if not selected:
            selected.append(iso_date)
            continue
        if rebalance_period_key(iso_date, frequency) == rebalance_period_key(
            selected[-1],
            frequency,
        ):
            selected[-1] = iso_date
        else:
            selected.append(iso_date)
    return selected


def portfolio_returns(
    weights: np.ndarray,
    returns: np.ndarray,
    rf: np.ndarray,
) -> np.ndarray:
    invested = float(weights.sum())
    if invested > 1.0 + 1e-12:
        weights = weights / invested
        invested = 1.0
    cash = 1.0 - invested
    return returns @ weights + cash * rf


def evaluate_mean_ann(
    weights: np.ndarray,
    returns: np.ndarray,
    rf: np.ndarray,
) -> tuple[float, float]:
    port = portfolio_returns(weights, returns, rf)
    if len(port) < 2:
        return float("-inf"), float("nan")
    mean_ann = float(port.mean()) * WEEKS_PER_YEAR
    vol_ann = float(port.std(ddof=1)) * math.sqrt(WEEKS_PER_YEAR)
    return mean_ann, vol_ann


def feasible(weights: np.ndarray, vol_cap: float, returns: np.ndarray, rf: np.ndarray) -> bool:
    if np.any(weights < -1e-9):
        return False
    if float(weights.sum()) > 1.0 + 1e-9:
        return False
    _, vol_ann = evaluate_mean_ann(weights, returns, rf)
    return vol_ann <= vol_cap + 1e-9


def clamp_weights(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(weights, 0.0)
    total = float(w.sum())
    if total > 1.0:
        w = w / total
    return w


def apply_min_weight_floor(weights: np.ndarray, min_weight: float) -> np.ndarray:
    """Drop tiny positions while preserving implicit cash."""
    w = clamp_weights(weights)
    if min_weight <= 0:
        return w
    w = w.copy()
    w[w < min_weight] = 0.0
    return clamp_weights(w)


def excess_returns_matrix(returns: np.ndarray, rf: np.ndarray) -> np.ndarray:
    return returns - rf.reshape(-1, 1)


def portfolio_excess_vol_ann(weights: np.ndarray, excess_returns: np.ndarray) -> float:
    if len(excess_returns) < 2:
        return float("nan")
    weekly = excess_returns @ weights
    return float(weekly.std(ddof=1)) * math.sqrt(WEEKS_PER_YEAR)


def scale_to_budget_and_vol(
    weights: np.ndarray,
    excess_returns: np.ndarray,
    *,
    vol_cap: float,
) -> np.ndarray:
    w = clamp_weights(weights)
    total = float(w.sum())
    if total > 1.0 + 1e-12:
        w = w / total
    vol = portfolio_excess_vol_ann(w, excess_returns)
    if vol_cap > 0 and vol == vol and vol > vol_cap + 1e-12:
        w = w * (vol_cap / vol)
    total = float(w.sum())
    if total > 1.0 + 1e-12:
        w = w / total
    return clamp_weights(w)


def finalize_candidate(
    weights: np.ndarray,
    excess_returns: np.ndarray,
    *,
    vol_cap: float,
    min_weight: float,
) -> np.ndarray:
    w = scale_to_budget_and_vol(weights, excess_returns, vol_cap=vol_cap)
    w = apply_min_weight_floor(w, min_weight)
    w = scale_to_budget_and_vol(w, excess_returns, vol_cap=vol_cap)
    w = apply_min_weight_floor(w, min_weight)
    return scale_to_budget_and_vol(w, excess_returns, vol_cap=vol_cap)


def covariance_direction(cov: np.ndarray, mu: np.ndarray) -> np.ndarray:
    if len(mu) == 1:
        return np.array([max(0.0, mu[0])], dtype=float)
    diag = np.diag(cov)
    ridge = max(float(np.nanmean(diag)) * 1e-6, 1e-10)
    try:
        direction = np.linalg.solve(cov + np.eye(len(mu)) * ridge, mu)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(cov + np.eye(len(mu)) * ridge) @ mu
    return np.maximum(direction, 0.0)


def fast_weight_candidates(
    returns: np.ndarray,
    rf: np.ndarray,
    *,
    vol_cap: float,
    min_weight: float,
) -> list[np.ndarray]:
    n = returns.shape[1]
    if n == 0:
        return []
    excess = excess_returns_matrix(returns, rf)
    mu = excess.mean(axis=0)
    cov = np.cov(excess, rowvar=False)
    if n == 1:
        cov = np.array([[float(cov)]], dtype=float)
    diag = np.maximum(np.diag(cov), 1e-12)

    raw: list[np.ndarray] = [np.zeros(n), np.full(n, 1.0 / n)]
    for i in range(n):
        one = np.zeros(n)
        one[i] = 1.0
        raw.append(one)

    positive_mu = np.maximum(mu, 0.0)
    if positive_mu.sum() > 0:
        raw.append(positive_mu)
        raw.append(positive_mu / diag)
    cov_dir = covariance_direction(cov, mu)
    if cov_dir.sum() > 0:
        raw.append(cov_dir)

    return [
        finalize_candidate(
            candidate,
            excess,
            vol_cap=vol_cap,
            min_weight=min_weight,
        )
        for candidate in raw
    ]


def shift_to_asset(
    weights: np.ndarray,
    asset_index: int,
    delta: float,
) -> np.ndarray | None:
    if delta == 0:
        return weights.copy()
    w = float(weights[asset_index])
    if delta > 0 and weights.sum() + delta > 1.0 + 1e-12:
        delta = 1.0 - weights.sum()
        if delta <= 1e-12:
            return None
    if delta < 0 and w + delta < -1e-12:
        return None
    next_w = weights.copy()
    next_w[asset_index] = w + delta
    return clamp_weights(next_w)


def shift_between(
    weights: np.ndarray,
    from_index: int,
    to_index: int,
    delta: float,
) -> np.ndarray | None:
    if from_index == to_index or delta <= 0:
        return None
    if weights[from_index] + 1e-12 < delta:
        return None
    next_w = weights.copy()
    next_w[from_index] -= delta
    next_w[to_index] += delta
    return clamp_weights(next_w)


def greedy_refine_weights(
    start: np.ndarray,
    returns: np.ndarray,
    rf: np.ndarray,
    *,
    vol_cap: float,
) -> tuple[np.ndarray, float]:
    w = clamp_weights(start)
    score, _ = evaluate_mean_ann(w, returns, rf)
    if not feasible(w, vol_cap, returns, rf):
        score = float("-inf")
    n = len(w)

    for _ in range(REFINE_ROUNDS):
        for step in REFINE_STEPS:
            improved = True
            while improved:
                improved = False
                baseline = score
                for i in range(n):
                    for trial in (
                        shift_to_asset(w, i, step),
                        shift_to_asset(w, i, -step),
                    ):
                        if trial is None or not feasible(trial, vol_cap, returns, rf):
                            continue
                        trial_score, _ = evaluate_mean_ann(trial, returns, rf)
                        if trial_score > baseline + 1e-12:
                            w = trial
                            score = trial_score
                            baseline = score
                            improved = True
                    for j in range(n):
                        if i == j:
                            continue
                        trial = shift_between(w, j, i, step)
                        if trial is None or not feasible(trial, vol_cap, returns, rf):
                            continue
                        trial_score, _ = evaluate_mean_ann(trial, returns, rf)
                        if trial_score > baseline + 1e-12:
                            w = trial
                            score = trial_score
                            baseline = score
                            improved = True
    return w, score


def weight_candidates(n: int, rng: random.Random) -> list[np.ndarray]:
    cands: list[np.ndarray] = []
    if n == 0:
        return cands
    cands.append(np.zeros(n))
    cands.append(np.full(n, 1.0 / n))
    for i in range(n):
        one = np.zeros(n)
        one[i] = 1.0
        cands.append(one)
        scaled = np.zeros(n)
        scaled[i] = 0.5
        cands.append(scaled)
    for _ in range(RANDOM_WEIGHT_TRIALS):
        draws = np.array([rng.random() for _ in range(n)], dtype=float)
        if draws.sum() <= 0:
            continue
        w = draws / draws.sum()
        w *= rng.random()
        cands.append(w)
    return cands


def optimize_weights_on_subset(
    returns: np.ndarray,
    rf: np.ndarray,
    *,
    vol_cap: float,
    min_weight: float,
    rng: random.Random,
) -> tuple[np.ndarray, float]:
    best_w = np.zeros(returns.shape[1])
    best_score = float("-inf")
    for candidate in fast_weight_candidates(
        returns,
        rf,
        vol_cap=vol_cap,
        min_weight=min_weight,
    ):
        if not feasible(candidate, vol_cap, returns, rf):
            continue
        score, _ = evaluate_mean_ann(candidate, returns, rf)
        if score > best_score:
            best_score = score
            best_w = candidate
    return best_w, best_score


def forward_greedy_select(
    tickers: list[str],
    returns: np.ndarray,
    rf: np.ndarray,
    *,
    max_holdings: int,
    vol_cap: float,
    min_weight: float,
    rng: random.Random,
) -> tuple[dict[str, float], float]:
    selected: list[int] = []
    best_map: dict[str, float] = {}
    best_score = float("-inf")

    for _ in range(max_holdings):
        best_add: int | None = None
        best_weights: np.ndarray | None = None
        for j in range(len(tickers)):
            if j in selected:
                continue
            trial_idx = selected + [j]
            sub = returns[:, trial_idx]
            weights, score = optimize_weights_on_subset(
                sub,
                rf,
                vol_cap=vol_cap,
                min_weight=min_weight,
                rng=rng,
            )
            if score > best_score:
                best_score = score
                best_add = j
                best_weights = weights
        if best_add is None:
            break
        selected.append(best_add)
        if best_weights is not None:
            best_map = {
                tickers[selected[i]]: float(best_weights[i])
                for i in range(len(selected))
                if best_weights[i] > 1e-12
            }

    if not selected:
        for j in range(len(tickers)):
            sub = returns[:, [j]]
            weights, score = optimize_weights_on_subset(
                sub,
                rf,
                vol_cap=vol_cap,
                min_weight=min_weight,
                rng=rng,
            )
            if score > best_score:
                best_score = score
                best_map = (
                    {tickers[j]: float(weights[0])}
                    if weights[0] > 1e-12
                    else {}
                )
    return best_map, best_score


def random_subset_search(
    returns: np.ndarray,
    rf: np.ndarray,
    tickers: list[str],
    *,
    max_holdings: int,
    vol_cap: float,
    min_weight: float,
    rng: random.Random,
    random_set_trials: int,
) -> tuple[dict[str, float], float]:
    best_map: dict[str, float] = {}
    best_score = float("-inf")
    for _ in range(random_set_trials):
        k = rng.randint(1, max_holdings)
        chosen = rng.sample(range(len(tickers)), min(k, len(tickers)))
        sub = returns[:, chosen]
        weights, score = optimize_weights_on_subset(
            sub,
            rf,
            vol_cap=vol_cap,
            min_weight=min_weight,
            rng=rng,
        )
        if score > best_score:
            best_score = score
            best_map = {
                tickers[chosen[i]]: float(weights[i])
                for i in range(len(chosen))
                if weights[i] > 1e-12
            }
    return best_map, best_score


def optimize_window(
    tickers: list[str],
    dates: list[str],
    universe: Universe,
    *,
    max_holdings: int,
    vol_cap: float,
    min_weight: float,
    rng: random.Random,
    random_set_trials: int,
) -> dict[str, float]:
    if not tickers or len(dates) < 2:
        return {}
    returns, _mask, rf = build_window_matrices(tickers, dates, universe)
    rf = np.zeros_like(rf)

    sel_g, score_g = forward_greedy_select(
        tickers,
        returns,
        rf,
        max_holdings=max_holdings,
        vol_cap=vol_cap,
        min_weight=min_weight,
        rng=rng,
    )
    sel_r, score_r = random_subset_search(
        returns,
        rf,
        tickers,
        max_holdings=max_holdings,
        vol_cap=vol_cap,
        min_weight=min_weight,
        rng=rng,
        random_set_trials=random_set_trials,
    )
    return sel_r if score_r > score_g else sel_g


def build_etf_weight_schedule(
    universe: Universe,
    tickers: list[str],
    *,
    lookback_months: int,
    vol_cap: float,
    max_holdings: int,
    min_coverage_fraction: float,
    listing_years: float,
    max_abs_daily_return: float,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    schedule_start: str | None = None,
    schedule_end: str | None = None,
    rebalance_frequency: str = "monthly",
    seed: int = OPTIMIZER_SEED,
    random_set_trials: int = RANDOM_SET_TRIALS,
) -> tuple[list[str], list[dict[str, float]]]:
    """Weekly rows maximising mean ann. return over lookback with vol <= vol_cap; cash implicit."""
    all_end_dates = build_weekly_end_dates(universe, lookback_months=lookback_months)
    if schedule_start is not None:
        all_end_dates = [d for d in all_end_dates if d >= schedule_start]
    if schedule_end is not None:
        all_end_dates = [d for d in all_end_dates if d <= schedule_end]
    all_end_dates = select_rebalance_end_dates(all_end_dates, rebalance_frequency)

    rng = random.Random(seed)
    weight_rows: list[dict[str, float]] = []

    for end_date in all_end_dates:
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
        listing_cutoff = listing_cutoff_from_end(end_date, listing_years=listing_years)
        eligible = filter_listing_age(universe, eligible, listing_cutoff)
        eligible = filter_bad_ticks(
            universe,
            eligible,
            dates,
            max_abs_daily_return=max_abs_daily_return,
        )
        if not eligible:
            weight_rows.append({})
            continue

        optimal = optimize_window(
            eligible,
            dates,
            universe,
            max_holdings=max_holdings,
            vol_cap=vol_cap,
            min_weight=min_weight,
            rng=rng,
            random_set_trials=random_set_trials,
        )
        weight_rows.append(
            {
                market_id: optimal[market_id]
                for market_id in tickers
                if market_id in optimal and optimal[market_id] > 1e-12
            }
        )

    return all_end_dates, weight_rows
