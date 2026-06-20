from pathlib import Path

WEEKS_PER_YEAR = 52
DAYS_PER_YEAR = 252
RISK_FREE_ID = "us-30-day-fed-funds-rate"

_ETFS_DIR = Path("etfs")
MARKETS_CSV = _ETFS_DIR / "output" / "markets_stats_allowlist.csv"
YAHOO_DIR = _ETFS_DIR / "yahoo"

OPTIMISE_GREEDY_ROUNDS = 3
OPTIMISE_GREEDY_STEPS = (0.05, 0.02, 0.005)
DEFAULT_RANDOM_TRIALS = 10
OPTIMIZER_SEED = 1337
