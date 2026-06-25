"""ETF strategy configuration for the static site and monthly run."""

from __future__ import annotations

from pathlib import Path

# Vertical dashed line on charts; strategy and VWRP rebased to equity=1.0 here.
TRACKING_START_DATE = "2026-06-20"

BENCHMARK_ID = "ie00bk5bqt80"  # VWRP.L
BENCHMARK_LABEL = "VWRP"

INVESTENGINE_PORTFOLIO_API_URL = (
    "https://investengine.com/api/v0.33/public/shared_portfolio/"
    "5ca383e6593408b7dca1d4140abd788324418369/"
)
# Live portfolio on InvestEngine (shared link).
INVESTENGINE_PORTFOLIO_URL = (
    "https://investengine.com/share/portfolio/"
    "5ca383e6593408b7dca1d4140abd788324418369/"
)
INVESTENGINE_JSON_DIR = Path("public/json")
INVESTENGINE_ICONS_CACHE_DIR = Path("public/icons/cache")

ETFS_DIR = Path(__file__).resolve().parent
MARKETS_MANIFEST = ETFS_DIR / "markets.csv"
YAHOO_DIR = ETFS_DIR / "yahoo"

BACKTEST_YEARS = 10.0
REBALANCE_FREQUENCY = "monthly"
EWMA_SPAN = 6
LOOKBACK_MONTHS = 12
TARGET_VOL = 0.25
VOL_CAP_SENSITIVITY = (0.05, 0.10, 0.20, 0.25, 0.35, 0.50, 0.60)
MIN_WEIGHT = 0.05
MAX_HOLDINGS = 20
DRIFT_BAND = 0.05
LISTING_YEARS = 2.0
DIVIDENDS = "any"
