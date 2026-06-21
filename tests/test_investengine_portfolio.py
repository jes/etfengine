from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from site_builder.investengine_portfolio import (
    load_investengine_snapshot,
    logo_cache_name,
    ticker_to_market_id,
)
from strategy.data import Asset, Universe


SAMPLE_PAYLOAD = {
    "equities": [
        {
            "id": 1,
            "name": "NVIDIA Corp",
            "logo": "https://go.investengine.com/organisations/1/logo.png",
            "target_weight": "5.000000",
        },
        {
            "id": 2,
            "name": "Apple Inc",
            "logo": "",
            "target_weight": "3.000000",
        },
    ],
    "regions": [
        {"id": 1, "name": "North America", "color": "#336699", "target_weight": "60.000000"},
        {"id": 2, "name": "Europe", "color": "#993366", "target_weight": "40.000000"},
    ],
    "securities": [
        {
            "id": 10,
            "title": "iShares MSCI Korea",
            "ticker": "IKOR",
            "logo_uri": "https://go.investengine.com/securities/10/logo.jpg",
            "target_weight": "43.000000",
        },
        {
            "id": 11,
            "title": "Unknown ETF",
            "ticker": "ZZZZ",
            "logo_uri": "",
            "target_weight": "5.000000",
        },
    ],
}


def _fake_universe() -> Universe:
    asset = Asset(
        market_id="ie00bk5bqt80",
        name="iShares MSCI Korea",
        yahoo_ticker="IKOR.L",
        returns_by_date={},
        daily_returns_by_date={},
        ohlc_by_date={},
        first_date="2026-01-01",
    )
    return Universe(
        assets={"ie00bk5bqt80": asset},
        weekly_dates=[],
        market_names={"ie00bk5bqt80": "IKOR"},
        market_categories={"ie00bk5bqt80": "ETF"},
        spread_fraction={"ie00bk5bqt80": 0.001},
    )


class InvestEnginePortfolioTests(unittest.TestCase):
    def test_logo_cache_name_is_stable(self) -> None:
        url = "https://go.investengine.com/organisations/1/logo.png"
        self.assertEqual(
            logo_cache_name("equity", 1, url),
            logo_cache_name("equity", 1, url),
        )
        self.assertTrue(logo_cache_name("equity", 1, url).endswith(".png"))

    def test_ticker_to_market_id_strips_exchange_suffix(self) -> None:
        mapping = ticker_to_market_id(_fake_universe())
        self.assertEqual(mapping["IKOR"], "ie00bk5bqt80")

    def test_load_snapshot_parses_weights_and_regions(self) -> None:
        universe = _fake_universe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "json" / "20260621.json"
            json_path.parent.mkdir(parents=True)
            json_path.write_text(json.dumps(SAMPLE_PAYLOAD), encoding="utf-8")
            snapshot = load_investengine_snapshot(
                json_path,
                universe=universe,
                icons_cache_dir=root / "icons" / "cache",
                snapshot_icons_dir=root / "snapshot" / "icons",
                top_equities=2,
            )
        self.assertEqual(snapshot.fetched_date, "2026-06-21")
        self.assertEqual(len(snapshot.equity_holdings), 2)
        self.assertAlmostEqual(snapshot.equity_holdings[0].weight_pct, 0.05)
        self.assertEqual(snapshot.equity_holdings[0].name, "NVIDIA Corp")
        self.assertAlmostEqual(snapshot.etf_weights_by_market_id["ie00bk5bqt80"], 0.43)
        self.assertEqual(len(snapshot.unmapped_etfs), 1)
        self.assertEqual(snapshot.unmapped_etfs[0].ticker, "ZZZZ")
        self.assertEqual(len(snapshot.region_breakdown), 2)
        self.assertAlmostEqual(snapshot.region_breakdown[0].weight_pct, 0.60)


if __name__ == "__main__":
    unittest.main()
