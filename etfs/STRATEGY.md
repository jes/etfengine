# ETF strategy (InvestEngine, vol-capped sparse optimiser)

Walk-forward backtest for a **UK Stocks & Shares ISA** on **InvestEngine DIY** (0% platform,
0% dealing). Unlevered: portfolio weights sum to ≤ 1, remainder is cash.

**Implementation:** `etfs/sharpening_optimizer.py`, `etfs/sharpening_backtest.py`,
`etfs/plot_portfolio_weights.py`.

**Python:** always use the repo venv from the repository root:

```bash
.venv/bin/python ...
```

---

## Reproduction recipe (run in order)

A new agent should run these commands from the **repository root**. The
reference backtest uses **monthly** rebalance, **EWMA span 6**, and the **full InvestEngine
ISA universe** (accumulating + distributing).

### 1. InvestEngine allowlist (acc + dist)

```bash
.venv/bin/python etfs/fetch_investengine_universe.py
```

Writes `etfs/_sources/investengine_allowlist.csv`. Expect **~870** visible, tradable ETFs
(~492 accumulating, ~378 distributing). Requires `curl` on PATH (preferred) or a full HTML
download via `urllib`.

Filter rules in `etfs/fetch_investengine_universe.py`:
- `is_visible_in_universe == true`
- `is_trading_available == true`
- `is_sell_only == false`
- `--dividends any` keeps both share classes

### 2. Universe manifest (justETF ∩ InvestEngine)

```bash
.venv/bin/python etfs/build_universe.py
```

Reads cached justETF screener `etfs/_sources/justetf_xlon_gb_longonly.csv` (XLON, GB,
long-only). When `etfs/_sources/investengine_allowlist.csv` exists, `build_universe.py`
**defaults** to intersecting with it and sets **min fund size = 0**.

Writes `etfs/markets.csv`. Expect **~870 ETFs + risk-free row**.

`id` = lowercase ISIN. `yahoo_ticker` = `{LSE_ticker}.L`.

### 3. Price history (skip if `etfs/yahoo/` already populated)

```bash
.venv/bin/python fetch_yahoo_history.py \
  --input etfs/markets.csv \
  --output-dir etfs/yahoo
```

One CSV per `id` under `etfs/yahoo/`. Large universe: allow 20–40 minutes at default delay.

### 4. Per-market stats filter

```bash
.venv/bin/python etfs/market_stats.py
```

Loads all rows in `etfs/markets.csv` with Yahoo files. Keeps markets with **≥ 500 weeks**
of weekly returns (`--min-weeks 500` default). Flags bad ticks but does **not** exclude
flagged rows from the output.

Writes `etfs/output/market_stats.csv`. Expect **~389** rows (varies with price refresh).

### 5. Optional stats-pruned report

```bash
.venv/bin/python etfs/build_markets_stats_allowlist.py
```

Writes:
- `etfs/output/markets_stats_allowlist.csv` — stats-pruned manifest report
- filters `etfs/output/market_stats.csv` to the same intersection

Always includes **risk-free** (`us-30-day-fed-funds-rate`) and **benchmark** (`ie00bk5bqt80`
VWRP) even if they fail the 500-week stats cut.

Expect **~391 report rows** (389 stats + RF + VWRP). The backtest no longer uses this
file by default.

### 6. Backtest + equity plot

```bash
.venv/bin/python etfs/sharpening_backtest.py
```

(`--rebalance-frequency` defaults to **monthly**; `--ewma-span` defaults to **6**;
`--dividends` defaults to **any**.)

### 7. Stacked weights plot

```bash
.venv/bin/python etfs/plot_portfolio_weights.py
```

---

## Universe loading (what the backtest actually uses)

`sharpening_backtest.py` resolves the manifest via `resolve_markets_csv()`:

1. `etfs/markets.csv` unless `--markets-csv` is supplied

Then intersects the causal/tradability allowlists before `load_universe()`:

| Filter | Source |
|---|---|
| Price-history stats | Optional only (`--allowlist-csv`; no default) |
| Dividend policy | `--dividends any` → no extra filter |
| InvestEngine tradable | `investengine_market_ids(etfs/markets.csv)` vs `etfs/_sources/investengine_allowlist.csv` |

Always added to allowed ids: **benchmark** (`ie00bk5bqt80`), **risk-free**
(`us-30-day-fed-funds-rate`).

**Reference load (Jun 2026 prices):** all InvestEngine manifest ETFs with Yahoo history,
plus risk-free. Newer ETFs are admitted causally once they have enough history at a
rebalance date.

### Universe summary

| Layer | File | ~Count |
|---|---|---|
| IE broker allowlist | `etfs/_sources/investengine_allowlist.csv` | 870 |
| justETF ∩ IE manifest | `etfs/markets.csv` | 870 + RF |
| ≥500 weeks history report | `etfs/output/market_stats.csv` | 389 |
| Stats-pruned manifest report | `etfs/output/markets_stats_allowlist.csv` | 389 + RF + VWRP |
| Loaded for optimiser | `etfs/markets.csv` ∩ available Yahoo histories | varies |

| Field | Value |
|---|---|
| Exchange | LSE (XLON) via justETF |
| Account | InvestEngine **Stocks & Shares ISA** |
| Dividend policy | **`any`** (acc + dist) |
| Prices | `etfs/yahoo/*.csv` (daily → weekly returns) |
| Risk-free | `us-30-day-fed-funds-rate` (`ZQ=F`) |
| Benchmark | `ie00bk5bqt80` (VWRP.L, accumulating All-World) |

Non-InvestEngine ETFs cannot enter the optimiser even if present in stale CSV artifacts.

---

## Strategy pipeline

Each **rebalance date** produces a raw weight vector. That vector is smoothed, floored,
vol-scaled, then simulated weekly with drift-band logic until the next rebalance.

```
1. Optimiser (per rebalance date)
   └─ max mean annualised return over 12-month lookback
   └─ subject to portfolio vol ≤ 25% ann.

2. EWMA smooth (causal, across rebalance rows)
   └─ span = 6 rebalance periods (monthly ≈ 6 months)

3. Minimum weight floor (5%)
   └─ drop sub-threshold positions; do NOT renormalise away cash

4. Vol scaling (post-EWMA, per rebalance row)
   └─ scale weights down to hit 25% vol target; re-apply floor iteratively

5. Weekly simulation
   └─ 5% relative drift band vs prior effective weights
   └─ bid–ask spread drag on trades (half spread on buy and sell)
   └─ uninvested cash earns 0% (InvestEngine ISA)
```

### Rebalance calendar

`select_rebalance_end_dates()` keeps the **last available weekly end-date in each calendar
month** (`--rebalance-frequency monthly`).

Schedule building starts at `window_start_from_end(backtest_start, months=12)` so the first
rebalance row has a full 12-month lookback. Weekly simulation runs from `backtest_start`
(10 years before last price date) through `universe.weekly_dates[-1]`, using targets from
the latest rebalance row **strictly before** each week (`target_weights_for_date`).

---

## Optimiser (`etfs/sharpening_optimizer.py`)

**Objective:** maximise **mean annualised portfolio return** over the lookback window
(weekly returns × 52).

**Constraint:** portfolio volatility (annualised, weekly returns, ddof=1) ≤ `target_vol`
(25%). Uninvested weight `(1 − sum(w))` contributes nothing to portfolio returns.

Vol is a **cap**, not a leverage target.

### Per-window eligibility

Applied each rebalance date on the 12-month lookback window:

| Rule | Parameter |
|---|---|
| Min weekly coverage | 95% of lookback weeks (`eligible_assets`) |
| Listed before rebalance | first price date ≤ end − 2 years |
| Bad daily tick | exclude if any daily \|r\| > 20% in lookback |

### Search method (sparse)

Constants in `etfs/sharpening_optimizer.py` / `strategy/constants.py`:

| Constant | Value |
|---|---|
| `OPTIMIZER_SEED` | **1337** (`strategy/constants.py`) |
| `RANDOM_SET_TRIALS` | **8** |
| `DEFAULT_MIN_WEIGHT` | **0.05** |

Algorithm per window (`optimize_window`):

1. **Forward greedy selection** up to `max_holdings` (20): iteratively add the ETF that
   maximises mean ann. return when re-optimising weights on the selected subset.
2. **Random subset search**: 8 trials; each picks `k ~ Uniform(1..max_holdings)` random
   ETFs, optimises weights on that subset.
3. Take whichever of (1) and (2) scores higher.

Weight optimisation on a subset (`optimize_weights_on_subset`):
- Builds candidates: cash-only, equal-weight, single-name, mean-variance direction, etc.
- Each candidate passed through `finalize_candidate`: scale to vol cap, apply 5% floor,
  re-scale (see `scale_to_budget_and_vol` / `apply_min_weight_floor`).
- Weights clamped to `[0, 1]` with sum ≤ 1.

---

## Post-optimiser rules

### EWMA smoothing (`ewma_smooth_capped_weight_rows`)

Causal EWMA over successive **optimiser output rows** (one row per rebalance date):

```python
alpha = 2 / (span + 1)
smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i-1]
```

If EWMA total > 1, scale down to 1 before flooring. Then zero weights below 5%; survivors
are **not** renormalised to 1 (cash may remain).

**`span` counts rebalance observations, not calendar weeks:**

| Rebalance frequency | `span=6` | `span=24` |
|---|---|---|
| Monthly | ~6 months | ~2 years |
| Weekly | ~6 weeks | ~6 months |

**Current configuration:** monthly, **`ewma_span = 6`**.

### Minimum weight floor

- `min_weight = 0.05` (5%)
- Positions below 5% removed after EWMA and after vol scaling
- With 5% floor, at most **20** holdings (`max_holdings` capped at `floor(1/0.05)`)

### Vol scaling (post-EWMA)

On each rebalance row (`scale_row_to_vol_target`), using the same 12-month lookback ending
at that rebalance date:

1. Floor sub-5% weights.
2. Estimate portfolio vol from weekly returns (uninvested cash earns 0%).
3. Scale all weights by `target_vol / realised_vol`, capped so total ≤ 1.
4. Re-apply 5% floor; repeat if the holdings set changes.

### Drift band (weekly simulation)

Each week (`resolve_effective_weights` / `apply_rebalance_drift_band`):

- Target = latest smoothed/scaled schedule row strictly before that week.
- Effective = prior effective, except where a position has drifted **> 5% relative**
  from target (`|current − target| / target > 0.05`), then snap to target.
- On first week or empty prior: use target outright.

### Costs

| Cost | Modelled? | Detail |
|---|---|---|
| ETF OCF/TER | Yes (implicit) | In price returns |
| ETF bid–ask | Yes | `spread_pct` from manifest; half on buy, half on sell (`rebalance_spread_drag`) |
| IE platform fee | No | 0% on DIY ISA |
| IE dealing | No | £0 on DIY |
| FX | No | See `etfs/fees.yaml` for future profiles |

`spread_pct` in `markets.csv` are **placeholder tiers** (0.08–0.30% by category), not live
quotes.

---

## Parameters (canonical configuration)

| Parameter | Value | CLI flag | Notes |
|---|---|---|---|
| Account | IE ISA | — | acc + dist allowed |
| Backtest horizon | 10 years | `--backtest-years 10` | From last price date |
| Rebalance frequency | **monthly** | `--rebalance-frequency monthly` | Last week in month |
| Lookback | 12 months | `--lookback-months 12` | |
| Target vol cap | 25% ann. | `--target-vol 0.25` | Cap, not lever target |
| EWMA span | **6** (monthly default) | `--ewma-span` optional | Auto: 6 monthly, 24 weekly |
| Min weight | 5% | `--min-weight 0.05` | |
| Max holdings | 20 | `--max-holdings 20` | |
| Min data coverage | 95% | `--min-coverage 0.95` | Of lookback weeks |
| Min listing age | 2 years | `--listing-years 2.0` | Before rebalance date |
| Bad tick filter | \|daily r\| > 20% | `--max-abs-daily-return 0.20` | Excludes asset for window |
| Drift band | 5% relative | `--drift-band 0.05` | |
| Dividend filter | **any** | `--dividends` | Default: any |
| Cash return | **0%** (IE ISA) | — | Uninvested cash earns nothing |
| Optimiser seed | 1337 | — | `OPTIMIZER_SEED` |
| Random subset trials | 8 | — | `RANDOM_SET_TRIALS` |
| Stats min history | 500 weeks | — | `market_stats.py` default |

---

## Reference performance (Jun 2026 price data)

Monthly, EWMA-6, IE ISA universe (defaults), hold period **2016-06-24 .. 2026-06-19**
(522 weeks):

| | Mean ann. | Vol ann. | Sharpe | CAGR |
|---|---|---|---|---|
| **ETF ISA (monthly)** | 21.64% | 27.45% | **0.80** | **19.49%** |
| **VWRP benchmark** | 8.46% | 11.91% | 0.74 | 8.04% |

Invested weight: mean 97.3%, median 100.0%, weeks with cash > 1%: 160.

**Benchmark caveat:** VWRP listed mid-2019. `benchmark_weekly()` uses the **risk-free
weekly return** for benchmark weeks before VWRP's first price date. The printed VWRP stats
are therefore **not** a pure 10-year buy-and-hold VWRP figure. For a full-history All-World
comparison use VWRL (`ie00b3rbwm25`, distributing, longer history) as a secondary check.

Past backtest; not a forecast.

### Latest holdings snapshot (2026-06-19)

Effective weights from `etfs/output/sharpening_portfolio_weights.csv` (after drift band):

| Weight | Ticker | Fund |
|---|---|---|
| 41.5% | IKOR.L | iShares MSCI Korea UCITS ETF (Dist) |
| 14.7% | AUCP.L | L&G Gold Mining UCITS ETF |
| 10.5% | COMG.L | Amundi Bloomberg Equal-weight Commodity ex-Agriculture UCITS ETF Acc |
| 8.5% | SEMG.L | Amundi MSCI Semiconductors UCITS ETF Acc |
| 7.4% | SSLN.L | iShares Physical Silver ETC |
| 6.7% | X7PP.L | Invesco European Banks Sector UCITS ETF |
| 5.8% | KRWL.L | Amundi MSCI Korea UCITS ETF Acc |
| 4.8% | (cash) | |

---

## Outputs

| File | Produced by | Contents |
|---|---|---|
| `etfs/output/sharpening_equity.png` | `sharpening_backtest.py` | Equity vs VWRP |
| `etfs/output/sharpening_weekly_diagnostics.csv` | `sharpening_backtest.py` | Weekly equity, drag, holdings |
| `etfs/output/sharpening_portfolio_weights.png` | `plot_portfolio_weights.py` | Stacked area chart |
| `etfs/output/sharpening_portfolio_weights.csv` | `plot_portfolio_weights.py` | Weekly effective weights + `__cash__` |
| `etfs/output/market_stats.csv` | `market_stats.py` | Per-ETF history stats |
| `etfs/output/markets_stats_allowlist.csv` | `build_markets_stats_allowlist.py` | Optional stats-pruned manifest report |

---

## Verification checklist

After reproduction, expect approximately:

| Check | Expected |
|---|---|
| IE allowlist rows | ~870 |
| `markets.csv` ETF rows | ~870 |
| `market_stats.csv` rows | ~389 |
| Loaded allocatable ETFs | all manifest ETFs with Yahoo history |
| Monthly optimiser steps | ~133 |
| Trade weeks in backtest | ~522 |
| Strategy CAGR (Jun 2026 data) | ~19.5% (±1% if prices refreshed) |
| Strategy Sharpe | ~0.8 |

If results diverge, check in order:

1. `investengine_allowlist.csv` contains **both** ACCUMULATING and DISTRIBUTING (~870 rows).
2. `etfs/yahoo/` price files match `markets.csv` ids.
3. `OPTIMIZER_SEED = 1337` unchanged in `strategy/constants.py`.

---

## Related files

| File | Role |
|---|---|
| `etfs/fetch_investengine_universe.py` | IE allowlist scraper |
| `etfs/build_universe.py` | justETF → `markets.csv` |
| `etfs/market_stats.py` | History quality filter |
| `etfs/build_markets_stats_allowlist.py` | Manifest ∩ stats |
| `etfs/sharpening_optimizer.py` | Sparse vol-capped optimiser |
| `etfs/sharpening_backtest.py` | Backtest + equity plot |
| `etfs/plot_portfolio_weights.py` | Weights chart + CSV |
| `etfs/fees.yaml` | Future platform cost profiles |
| `etfs/README.md` | Broader ETF track notes |
| `strategy/data.py` | `load_universe`, weekly returns |
| `strategy/weights.py` | EWMA + `target_weights_for_date` |
| `strategy/costs.py` | Drift band + spread drag |
