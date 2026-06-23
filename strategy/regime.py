"""Regime cash filter from always-invested shadow trailing returns."""

from __future__ import annotations

from strategy.constants import WEEKS_PER_YEAR

REGIME_MONTHS = (3, 6, 12)


def months_to_weeks(months: int) -> int:
    return round(months * WEEKS_PER_YEAR / 12)


def selection_weeks(regime_months: tuple[int, ...] = REGIME_MONTHS) -> tuple[int, ...]:
    return tuple(months_to_weeks(months) for months in regime_months)


def min_regime_history(regime_months: tuple[int, ...] = REGIME_MONTHS) -> int:
    weeks = selection_weeks(regime_months)
    return max(weeks) if weeks else 0


def trailing_compound_return(weekly_returns: list[float], weeks: int) -> float:
    if len(weekly_returns) < weeks:
        return float("nan")
    value = 1.0
    for weekly_return in weekly_returns[-weeks:]:
        value *= 1.0 + weekly_return
    return value - 1.0


def regime_vote(
    shadow_returns: list[float],
    regime_months: tuple[int, ...] = REGIME_MONTHS,
    regime_weeks: tuple[int, ...] | None = None,
) -> tuple[int, dict[int, str]]:
    """Vote cash when the shadow book lost money over the trailing window."""
    if regime_weeks is None:
        regime_weeks = selection_weeks(regime_months)
    votes: dict[int, str] = {}
    losing_votes = 0
    for months, weeks in zip(regime_months, regime_weeks, strict=True):
        trailing_return = trailing_compound_return(shadow_returns, weeks)
        if trailing_return < 0:
            losing_votes += 1
            votes[months] = "cash"
        else:
            votes[months] = "invested"
    return losing_votes, votes


def in_regime_cash(votes: dict[int, str], regime_months: tuple[int, ...] = REGIME_MONTHS) -> bool:
    return bool(votes) and all(votes.get(months) == "cash" for months in regime_months)


def vote_label(vote: str) -> str:
    if vote == "invested":
        return "bullish"
    if vote == "cash":
        return "bearish"
    return vote
