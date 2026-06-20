# Agent Notes

## Python Environment

Use the repo venv for all Python commands:

```bash
.venv/bin/python ...
```

Do not assume `python` is available on `PATH`. On this machine it is not.

## Monthly run

```bash
./run.sh
```

Refreshes Yahoo prices for the frozen universe manifest (`etfs/output/markets_stats_allowlist.csv`)
and rebuilds the static site under `public/`. Does **not** refresh the investable universe.

Tracking start date (vertical line + equity rebase): `etfs/config.py` → `TRACKING_START_DATE`.

## Tests

This repo's tests are `unittest` tests. The venv currently does not have
`pytest` installed, so run tests with `unittest` unless you have explicitly
installed extra tooling into the venv.

Common commands:

```bash
.venv/bin/python -m unittest
.venv/bin/python -m unittest tests.test_etf_site_builder
```

When validating a narrow change, run the smallest relevant unittest target
first, then broaden to the containing module or full suite as time allows.
