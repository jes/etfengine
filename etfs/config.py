"""ETF strategy configuration for the static site and monthly run."""

from __future__ import annotations

from pathlib import Path

# Vertical dashed line on charts; strategy and VWRP rebased to equity=1.0 here.
TRACKING_START_DATE = "2026-06-20"

BENCHMARK_ID = "ie00bk5bqt80"  # VWRP.L
BENCHMARK_LABEL = "VWRP"

# Live portfolio on InvestEngine (shared link).
INVESTENGINE_PORTFOLIO_URL = (
    "https://investengine.com/share/portfolio/"
    "5ca383e6593408b7dca1d4140abd788324418369/"
)

ETFS_DIR = Path(__file__).resolve().parent
MARKETS_MANIFEST = ETFS_DIR / "markets.csv"
MARKETS_STATS_ALLOWLIST = ETFS_DIR / "output" / "markets_stats_allowlist.csv"
MARKET_STATS_ALLOWLIST = ETFS_DIR / "output" / "market_stats.csv"
YAHOO_DIR = ETFS_DIR / "yahoo"

BACKTEST_YEARS = 10.0
REBALANCE_FREQUENCY = "monthly"
EWMA_SPAN = 6
LOOKBACK_MONTHS = 12
TARGET_VOL = 0.25
MIN_WEIGHT = 0.05
MAX_HOLDINGS = 20
DRIFT_BAND = 0.05
DIVIDENDS = "any"
