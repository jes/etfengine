# ETF strategy (UK retail, unlevered)

InvestEngine DIY ETF portfolio: **UK-listed UCITS ETFs** in a normal investment account
(ISA/GIA), **no leverage above 1×**, vol-capped sparse optimiser.

## Optimiser

| | ETF strategy |
|---|---|
| Objective | Max **mean annualised excess return** over lookback |
| Vol control | **Cap** portfolio vol at `target_vol`; weights sum ≤ 1, remainder is **cash** |
| Leverage | 0–1× only (cash is the de-lever tool) |
| Costs | Platform fee + dealing fee + ETF bid–ask (see below) |

Under a vol **cap**, the optimiser chooses how much of the vol budget to use and how to
allocate it across ETFs. A low-vol mix may leave cash un invested; a high-return mix fills
the vol budget without exceeding it.

Cash earns **0%** on InvestEngine. The optimiser and weekly simulation treat uninvested
weight as dead cash.

## Ongoing fund charges (OCF/TER)

Already in the price returns — not modeled separately. Platform fee, dealing commission,
and bid–ask on rebalance are the extra costs to add in the backtest layer.

## Bid–ask (`spread_pct` in `markets.csv`)

**Current values are placeholders, not sourced data.** They were hand-assigned by rough
liquidity tier when the universe was first built:

| Tier | `spread_pct` | Examples |
|---|---|---|
| Very liquid LSE index ETFs | 0.08–0.10% | CSPX, VUKE, IBTM |
| Mid-liquidity equity | 0.12–0.15% | single-country, EM |
| Commodity ETCs | 0.20–0.30% | CRUD, NGSP, SSLN |

Before trusting cost-sensitive results, these should be replaced with **actual bid–offer
spreads** from a broker or exchange — e.g. HL live quote at trade time, InvestEngine
spread at execution, or LSE closing spread snapshots. Historical bid–ask series are hard
to get for free; a practical approach is a one-off snapshot per instrument (conservative)
or a rule based on AUM / average daily volume.

The field is **one-way spread as % of price**, applied half on buys and half on sells in
`rebalance_spread_drag`.

## UK broker costs (what to model)

HL is fine if that is where you would actually hold the portfolio, but it is **not** the
cheapest place for a multi-ETF, regularly rebalanced strategy. For a DIY ETF-only book,
**InvestEngine** (0% platform, 0% trades on DIY) or **AJ Bell** (0.25% p.a. capped at
£3.50/month, from ~£1.50/trade) are more representative of low-friction ETF investing.

### Reference fee schedules (verify on provider sites before live use)

| Provider | Platform (ETF/shares) | Dealing | Notes |
|---|---|---|---|
| **InvestEngine** | 0% | £0 | ETF-only DIY; no LISA/JISA |
| **AJ Bell** | 0.25% p.a., cap £3.50/mo | from £1.50; regular investing £0 | Broader product range |
| **HL** (from Mar 2026) | 0.35% p.a., cap £12.50/mo | £6.95 (£3.95 if 20+ trades/mo) | Regular investing £0 on selected ETFs |
| **Vanguard** | £4/mo or 0.15% | £0 on Vanguard funds; ETF trades £7.50 | Good if universe is mostly Vanguard |

`fees.yaml` holds numeric assumptions for backtests. Default scenario uses **InvestEngine**
(trading + platform = 0) plus ETF `spread_pct` from the manifest; switch profile to
`hl` or `aj_bell` to stress-test.

### Other real-world frictions (not all modeled yet)

- **FX**: USD/EUR-denominated ETFs bought in GBP incur broker FX spread (often 0.5–1%).
- **Stamp duty**: UK-domiciled ETFs on LSE — no SDRT; some US-listed products may differ
  by broker.
- **Spread on illiquid ETCs**: commodity ETCs can be wider than manifest estimates.

## Where to find a wide UK ETF universe

| Source | What you get | How we use it |
|---|---|---|
| **[justETF screener](https://www.justetf.com/en/find-etf.html)** | ~2,400+ LSE-listed UCITS ETFs/ETCs authorised for UK retail; ISIN, ticker, size, asset class | **`etfs/build_universe.py`** — primary source (`exchange=XLON`, `local_country=GB`, long-only) |
| **[HL ETF list](https://www.hl.co.uk/shares/exchange-traded-funds-etfs/list-of-etfs)** | What HL clients actually trade; sector filters | Cross-check liquidity / UK access; no bulk download |
| **[InvestEngine](https://investengine.com/etf)** | ~830 DIY ETFs | Smaller curated set; good sanity check |
| **[etfdb on GitHub](https://github.com/albertored/etfdb)** | ~4,200 European UCITS (CSV) | Broad metadata; tickers are often Xetra/gettex codes, not LSE |
| **LSE / [etftrack.com](https://etftrack.com/exchange/lse/)** | Full exchange listing (~1,400+) | Reference; manual |

Regenerate `markets.csv` from justETF (cached under `_sources/`). When
`etfs/_sources/investengine_allowlist.csv` exists, `build_universe.py` defaults to
intersecting with that InvestEngine tradable set:

```bash
.venv/bin/python etfs/fetch_investengine_universe.py
.venv/bin/python etfs/build_universe.py
.venv/bin/python etfs/market_stats.py
.venv/bin/python etfs/build_markets_stats_allowlist.py
```

Without an InvestEngine allowlist file, pass `--min-size-meur 500` explicitly for a
broader GIA proxy universe (distributing funds only):

```bash
.venv/bin/python etfs/build_universe.py --min-size-meur 500
.venv/bin/python etfs/build_universe.py --min-size-meur 50
.venv/bin/python etfs/build_universe.py --dividends any
.venv/bin/python etfs/build_universe.py --refresh  # re-scrape justETF (~90s)
```

To restrict to the actual InvestEngine tradable set explicitly:

```bash
.venv/bin/python etfs/fetch_investengine_universe.py
.venv/bin/python etfs/build_universe.py \
  --investengine-allowlist etfs/_sources/investengine_allowlist.csv
```

The InvestEngine fetcher reads the public Next.js payload from `https://investengine.com/etfs/all/`
and exports visible, tradable distributing ETFs. With an InvestEngine allowlist, the
builder defaults to no additional size floor; add `--min-size-meur 500` if you want the
liquid-only subset.

Then fetch prices (large universe: expect ~20–40 min at default 0.5s delay):

```bash
python fetch_yahoo_history.py --input etfs/markets.csv --output-dir etfs/yahoo
```

Yahoo tickers are `{justETF_ticker}.L` (verified on a random sample). `id` is the ISIN
(lower case) for uniqueness across share classes.

## Universe (`markets.csv`)

UK-retail-accessible UCITS ETFs/ETCs from justETF (LSE, long-only), filtered by the
builder options above. Columns include `id` (ISIN), `market_name`, `category`,
`spread_pct` (placeholder tier), `yahoo_ticker`, `dividends`, `ter_pct`, `size_meur`,
`currency`, `domicile_country`, and `hedged`.

Risk-free: **`ZQ=F`** (fed funds futures). Use `--min-size-meur` or
`--exclude-instrument` when the full list is too large for your backtest.

## Data layout

```
etfs/
  markets.csv       # universe manifest
  fees.yaml         # platform cost profiles
  yahoo/            # daily OHLC CSVs (one per id)
  README.md         # this file
```

Fetch / refresh prices (from repo root, venv active):

```bash
.venv/bin/python fetch_yahoo_history.py --input etfs/markets.csv --output-dir etfs/yahoo
```

## Status

- [x] InvestEngine distributing ETF allowlist from the public ETF page
      (`fetch_investengine_universe.py`)
- [x] GIA universe manifest from justETF (`build_universe.py`): distributing funds,
      intersected with the InvestEngine ISIN/ticker allowlist when supplied
- [x] Yahoo history fetch
- [x] Vol-targeted sparse optimiser scaffold (`sharpening_optimizer.py`), with
      configurable minimum ETF weight
- [x] ETF backtest CLI (`sharpening_backtest.py`) with monthly/quarterly/annual schedule
      options, EWMA smoothing, post-EWMA minimum weight floor, and post-EWMA vol scaling
- [x] ETF spread drag via manifest `spread_pct`
- [ ] Replace current heuristic optimiser with a better sparse optimiser
- [ ] Platform/account cost model beyond InvestEngine DIY zero dealing/platform fees
