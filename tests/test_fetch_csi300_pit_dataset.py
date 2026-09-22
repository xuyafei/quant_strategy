import unittest

import pandas as pd

from scripts.fetch_csi300_pit_dataset import (
    _membership_intervals,
    _parse_tushare_dates,
)


class FetchCsi300PitDatasetTests(unittest.TestCase):
    def test_parse_tushare_dates_treats_numeric_values_as_yyyymmdd(self) -> None:
        parsed = _parse_tushare_dates(pd.Series([20240819, 20240820.0, "2024-08-21"]))

        self.assertEqual(
            parsed.tolist(),
            [
                pd.Timestamp("2024-08-19"),
                pd.Timestamp("2024-08-20"),
                pd.Timestamp("2024-08-21"),
            ],
        )

    def test_membership_snapshot_becomes_effective_after_snapshot_date(self) -> None:
        snapshots = pd.DataFrame(
            {
                "trade_date": [20240131, 20240131, 20240229, 20240229],
                "con_code": ["A", "B", "B", "C"],
            }
        )

        intervals = _membership_intervals(
            snapshots,
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-03-31"),
        ).set_index("ts_code")

        self.assertEqual(intervals.loc["A", "effective_from"], pd.Timestamp("2024-02-01"))
        self.assertEqual(intervals.loc["A", "effective_to"], pd.Timestamp("2024-02-29"))
        self.assertEqual(intervals.loc["B", "effective_from"], pd.Timestamp("2024-02-01"))
        self.assertEqual(intervals.loc["B", "effective_to"], pd.Timestamp("2024-03-31"))
        self.assertEqual(intervals.loc["C", "effective_from"], pd.Timestamp("2024-03-01"))
        self.assertEqual(intervals.loc["C", "effective_to"], pd.Timestamp("2024-03-31"))


if __name__ == "__main__":
    unittest.main()
