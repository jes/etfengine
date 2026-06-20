import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from etfs.build_universe import filter_universe, load_identifier_allowlist
from etfs.fetch_investengine_universe import (
    extract_next_data,
    extract_securities,
    filter_securities,
)
from etfs.sharpening_backtest import investengine_market_ids
from etfs.sharpening_optimizer import (
    build_etf_weight_schedule,
    default_ewma_span,
    optimize_window,
    select_rebalance_end_dates,
)
from strategy.weights import ewma_smooth_capped_weight_rows


class EtfUniverseFilterTests(unittest.TestCase):
    def test_filter_universe_keeps_distributing_allowlisted_large_funds(self):
        df = pd.DataFrame(
            [
                {
                    "isin": "IE00AAA",
                    "ticker": "AAA",
                    "name": "AAA Fund",
                    "size": 600,
                    "dividends": "Distributing",
                    "instrument": "ETF",
                },
                {
                    "isin": "IE00BBB",
                    "ticker": "BBB",
                    "name": "BBB Fund",
                    "size": 700,
                    "dividends": "Accumulating",
                    "instrument": "ETF",
                },
                {
                    "isin": "IE00CCC",
                    "ticker": "CCC",
                    "name": "CCC Fund",
                    "size": 100,
                    "dividends": "Distributing",
                    "instrument": "ETF",
                },
            ]
        ).set_index("isin")

        out = filter_universe(
            df,
            min_size_meur=500,
            dividends="distributing",
            exclude_instruments=set(),
            allowlist={"IE00AAA", "IE00BBB"},
        )

        self.assertEqual(list(out.index), ["IE00AAA"])

    def test_identifier_allowlist_reads_isin_and_ticker_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "allowlist.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["isin", "ticker"])
                writer.writeheader()
                writer.writerow({"isin": "IE00 AAA", "ticker": "vwrp.l"})

            self.assertEqual(load_identifier_allowlist(path), {"IE00AAA", "VWRPL"})


class EtfScheduleRuleTests(unittest.TestCase):
    def test_active_optimizer_api_has_no_screen_pool(self):
        self.assertNotIn("screen_pool", inspect.signature(optimize_window).parameters)
        self.assertNotIn(
            "screen_pool",
            inspect.signature(build_etf_weight_schedule).parameters,
        )

    def test_select_monthly_rebalance_end_dates_uses_last_week_in_month(self):
        dates = [
            "2026-01-02",
            "2026-01-09",
            "2026-01-30",
            "2026-02-06",
            "2026-02-27",
        ]

        self.assertEqual(
            select_rebalance_end_dates(dates, "monthly"),
            ["2026-01-30", "2026-02-27"],
        )

    def test_capped_ewma_drops_sub_five_percent_weights_without_renormalizing(self):
        rows = [
            {"a": 0.80, "b": 0.20},
            {"a": 0.80},
        ]

        smoothed = ewma_smooth_capped_weight_rows(rows, span=1, min_weight=0.05)

        self.assertEqual(smoothed, [{"a": 0.80, "b": 0.20}, {"a": 0.80}])


class InvestEngineUniverseTests(unittest.TestCase):
    def test_default_ewma_span_matches_rebalance_frequency(self):
        self.assertEqual(default_ewma_span("monthly"), 6)
        self.assertEqual(default_ewma_span("weekly"), 24)

    def test_investengine_market_ids_maps_isins_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            allowlist = tmp_path / "ie.csv"
            markets = tmp_path / "markets.csv"
            with allowlist.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["isin", "ticker"])
                writer.writeheader()
                writer.writerow({"isin": "IE00AAA", "ticker": "AAA"})
                writer.writerow({"isin": "IE00BBB", "ticker": "BBB"})
            with markets.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["id", "yahoo_ticker"],
                )
                writer.writeheader()
                writer.writerow({"id": "ie00aaa", "yahoo_ticker": "AAA.L"})
                writer.writerow({"id": "ie00ccc", "yahoo_ticker": "CCC.L"})

            self.assertEqual(
                investengine_market_ids(markets, allowlist_path=allowlist),
                {"ie00aaa"},
            )

    def test_extracts_visible_tradable_distributing_securities_from_next_payload(self):
        payload = {
            "props": {
                "pageProps": {
                    "defaultSecurities": [
                        {
                            "isin": "IE00DIST",
                            "ticker": "DIST",
                            "dividends_type": "DISTRIBUTING",
                            "is_visible_in_universe": True,
                            "is_trading_available": True,
                            "is_sell_only": False,
                        },
                        {
                            "isin": "IE00ACC",
                            "ticker": "ACC",
                            "dividends_type": "ACCUMULATING",
                            "is_visible_in_universe": True,
                            "is_trading_available": True,
                            "is_sell_only": False,
                        },
                    ]
                }
            }
        }
        page = (
            '<script id="__NEXT_DATA__" type="application/json">'
            f"{__import__('json').dumps(payload)}"
            "</script>"
        )

        securities = extract_securities(extract_next_data(page))
        filtered = filter_securities(securities, dividends="distributing")

        self.assertEqual([security["isin"] for security in filtered], ["IE00DIST"])


if __name__ == "__main__":
    unittest.main()
