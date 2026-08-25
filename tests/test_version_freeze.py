"""Live version freeze manifest tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.version_freeze import build_freeze_manifest, save_freeze_outputs


class VersionFreezeTests(unittest.TestCase):
    def test_build_and_save_freeze_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data"
            data_dir.mkdir()
            pool_path = data_dir / "stock_pool.csv"
            pd.DataFrame(
                {
                    "股票代码": ["600519", "000001", "605366"],
                    "股票简称": ["贵州茅台", "平安银行", "宏柏新材"],
                    "是否启用": ["是", "否", "是"],
                }
            ).to_csv(pool_path, index=False)

            settings = replace(
                get_settings(),
                project_root=Path(__file__).resolve().parents[1],
                data_dir=data_dir,
                output_dir=root / "output",
                stock_pool_path=pool_path,
                paper_initial_cash=120000.0,
            )
            manifest = build_freeze_manifest(
                settings,
                strategy="FUSED_ROLLING_SCORE_WEIGHTED",
                as_of_date="2026-08-19",
                run_time="09:35",
                operator="unit",
                source_files=["config.py"],
            )
            self.assertEqual(manifest["as_of_date"], "2026-08-19")
            self.assertEqual(manifest["live_policy"]["capital"], 120000.0)
            self.assertFalse(manifest["live_policy"]["auto_submit_orders"])
            self.assertEqual(manifest["stock_pool"]["total_count"], 3)
            self.assertEqual(manifest["stock_pool"]["active_count"], 2)
            self.assertTrue(manifest["source_hashes"][0]["exists"])
            self.assertEqual(manifest["source_hashes"][0]["path"], "config.py")

            paths = save_freeze_outputs(settings, manifest)
            self.assertTrue(paths["json"].is_file())
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["markdown"].is_file())

            loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(loaded["operator"], "unit")
            report = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("实盘前版本冻结清单", report)
            self.assertIn("FUSED_ROLLING_SCORE_WEIGHTED", report)


if __name__ == "__main__":
    unittest.main()
