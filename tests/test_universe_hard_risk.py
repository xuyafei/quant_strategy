import unittest

import pandas as pd

from universe.risk import build_hard_risk_universe_report


class HardRiskUniverseTest(unittest.TestCase):
    def _base(self) -> pd.DataFrame:
        rows = []
        for date in ("2025-01-10", "2025-05-10"):
            for symbol in ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"):
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "eligible": True,
                        "exclude_reason": "",
                    }
                )
        return pd.DataFrame(rows)

    def test_hard_risks_are_point_in_time_and_missing_data_does_not_block(self) -> None:
        balance = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": 20250120,
                    "f_ann_date": 20250120,
                    "end_date": 20241231,
                    "report_type": 1,
                    "total_hldr_eqy_exc_min_int": -10.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": 20240120,
                    "f_ann_date": 20240120,
                    "end_date": 20231231,
                    "report_type": 1,
                    "total_hldr_eqy_exc_min_int": 50.0,
                },
            ]
        )
        audit = pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "ann_date": 20250301,
                    "end_date": 20241231,
                    "audit_result": "无法表示意见",
                },
                {
                    "ts_code": "000003.SZ",
                    "ann_date": 20240301,
                    "end_date": 20231231,
                    "audit_result": "保留意见",
                },
            ]
        )
        names = pd.DataFrame(
            [
                {
                    "ts_code": "000004.SZ",
                    "name": "退市测试",
                    "start_date": 20250401,
                    "end_date": None,
                    "ann_date": 20250401,
                }
            ]
        )
        report = build_hard_risk_universe_report(self._base(), balance, audit, names)
        keyed = report.set_index([report["date"].dt.strftime("%Y-%m-%d"), "symbol"])
        self.assertTrue(bool(keyed.loc[("2025-01-10", "000001.SZ"), "eligible"]))
        self.assertFalse(bool(keyed.loc[("2025-05-10", "000001.SZ"), "eligible"]))
        self.assertTrue(bool(keyed.loc[("2025-01-10", "000002.SZ"), "eligible"]))
        self.assertFalse(bool(keyed.loc[("2025-05-10", "000002.SZ"), "eligible"]))
        self.assertTrue(bool(keyed.loc[("2025-05-10", "000003.SZ"), "eligible"]))
        self.assertTrue(bool(keyed.loc[("2025-05-10", "000003.SZ"), "qualified_audit_warning"]))
        self.assertFalse(bool(keyed.loc[("2025-05-10", "000004.SZ"), "eligible"]))
        self.assertFalse(bool(keyed.loc[("2025-05-10", "000004.SZ"), "balance_available"]))

    def test_current_normal_audit_releases_prior_severe_opinion(self) -> None:
        audit = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": 20240101, "end_date": 20231231, "audit_result": "否定意见"},
                {"ts_code": "000001.SZ", "ann_date": 20250401, "end_date": 20241231, "audit_result": "标准无保留意见"},
            ]
        )
        report = build_hard_risk_universe_report(self._base(), pd.DataFrame(), audit, pd.DataFrame())
        keyed = report.set_index([report["date"].dt.strftime("%Y-%m-%d"), "symbol"])
        self.assertFalse(bool(keyed.loc[("2025-01-10", "000001.SZ"), "eligible"]))
        self.assertTrue(bool(keyed.loc[("2025-05-10", "000001.SZ"), "eligible"]))


if __name__ == "__main__":
    unittest.main()
