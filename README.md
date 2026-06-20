# ETF Engine

Monthly InvestEngine ETF strategy: refresh Yahoo prices, run a rolling backtest, publish a static HTML dashboard under `public/`.

## Quick start

```bash
./run.sh
```

This fetches updated prices for the frozen universe (`etfs/output/markets_stats_allowlist.csv`) and rebuilds `public/index.html`. No live trading or broker API.

## Configuration

| Setting | File | Notes |
|--------|------|-------|
| Tracking start date | `etfs/config.py` → `TRACKING_START_DATE` | Vertical line on charts; strategy and VWRP rebased to 1.0 here |
| Benchmark | `etfs/config.py` → `BENCHMARK_ID` | VWRP (FTSE All-World) |
| InvestEngine link | `etfs/config.py` → `INVESTENGINE_PORTFOLIO_URL` | Shown at top of site |
| Strategy params | `etfs/config.py` | Vol target, lookback, rebalance frequency, etc. |

## Layout

- `etfs/` — universe manifest, Yahoo CSVs, backtest (`sharpening_backtest.py`), optimizer
- `strategy/` — shared data loading, optimizer, weights, costs
- `site_builder/` — metrics, plots, HTML for the static site
- `build_site.py` — orchestrates backtest + site generation
- `fetch_yahoo_history.py` — incremental Yahoo price fetch
- `public/` — generated site (served as static files)

## Docs

- `etfs/README.md` — universe build and manifest workflow
- `etfs/STRATEGY.md` — strategy rules and backtest details

## Tests

```bash
.venv/bin/python -m unittest
```
