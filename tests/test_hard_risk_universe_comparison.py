import unittest

import pandas as pd

from scripts.build_hard_risk_universe_comparison import _selected


class HardRiskComparisonTest(unittest.TestCase):
    def test_selected_keeps_top_k_per_signal_date(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-03"), "000001.SZ"),
                (pd.Timestamp("2025-01-03"), "000002.SZ"),
                (pd.Timestamp("2025-01-03"), "000003.SZ"),
            ],
            names=["date", "symbol"],
        )
        score = pd.Series([1.0, 3.0, 2.0], index=index)
        selected = _selected(score, "TEST", top_k=2)
        self.assertEqual(selected["symbol"].tolist(), ["000002.SZ", "000003.SZ"])
        self.assertEqual(selected["rank"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
