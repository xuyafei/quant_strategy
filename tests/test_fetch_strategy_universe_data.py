import unittest

import pandas as pd

from scripts.fetch_strategy_universe_data import weekly_decision_dates


class FetchStrategyUniverseDataTests(unittest.TestCase):
    def test_weekly_decision_dates_use_last_open_session(self) -> None:
        calendar = pd.DataFrame(
            {
                "cal_date": ["20250101", "20250102", "20250103", "20250104", "20250106"],
                "is_open": [0, 1, 1, 0, 1],
            }
        )
        dates = weekly_decision_dates(
            calendar, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-06")
        )
        self.assertEqual(dates, ["20250103", "20250106"])


if __name__ == "__main__":
    unittest.main()
