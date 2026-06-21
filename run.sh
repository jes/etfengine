#!/bin/bash
set -euo pipefail

date=$(date +%Y%m%d)
cd /home/jes/etfengine

mkdir -p "public/logs/$date/"
mkdir -p "public/json/"

export PYTHONUNBUFFERED=1

.venv/bin/python fetch_investengine_portfolio.py \
  >> "public/logs/$date/fetch_investengine.log" 2>&1

.venv/bin/python fetch_yahoo_history.py \
  --input etfs/output/markets_stats_allowlist.csv \
  --output-dir etfs/yahoo \
  >> "public/logs/$date/fetch.log" 2>&1

.venv/bin/python build_site.py \
  >> "public/logs/$date/build_site.log" 2>&1

touch "public/logs/$date/complete.log"
