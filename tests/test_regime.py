import unittest

from etfs.sharpening_backtest import WeekPoint, apply_regime_cash_gate
from strategy.regime import (
    REGIME_MONTHS,
    in_regime_cash,
    min_regime_history,
    regime_vote,
    trailing_compound_return,
)


class RegimeTests(unittest.TestCase):
    def test_trailing_compound_return(self) -> None:
        returns = [0.01, -0.02, 0.03]
        self.assertAlmostEqual(trailing_compound_return(returns, 2), (0.98 * 1.03) - 1.0)

    def test_regime_vote_all_bearish(self) -> None:
        returns = [-0.01] * min_regime_history((1,))
        losing, votes = regime_vote(returns, regime_months=(1,), regime_weeks=(1,))
        self.assertEqual(losing, 1)
        self.assertEqual(votes[1], "cash")
        self.assertTrue(in_regime_cash(votes, (1,)))

    def test_apply_regime_cash_gate_uses_risk_free_when_unanimous(self) -> None:
        shadow = []
        for index in range(4):
            shadow.append(
                WeekPoint(
                    iso_date=f"2024-0{index + 1}-01",
                    equity=1.0 - 0.1 * (index + 1),
                    weekly_return=-0.1,
                    invested_weight=1.0,
                    cash_weight=0.0,
                    spread_drag=0.0,
                    net_weekly=-0.1,
                    holdings="a",
                    effective_weights={"a": 1.0},
                    target_weights={"a": 1.0},
                    shadow_weekly_return=-0.1,
                )
            )
        real_points = apply_regime_cash_gate(
            shadow,
            rf_returns=[0.0, 0.0, 0.0, 0.01],
            regime_months=(1,),
        )
        self.assertTrue(real_points[-1].in_regime_cash)
        self.assertAlmostEqual(real_points[-1].equity, 0.9 * 0.9 * 0.9 * 1.01)
        self.assertEqual(real_points[-1].regime_votes, ((1, "cash"),))

    def test_default_regime_months(self) -> None:
        self.assertEqual(REGIME_MONTHS, (3, 6, 12))


if __name__ == "__main__":
    unittest.main()
