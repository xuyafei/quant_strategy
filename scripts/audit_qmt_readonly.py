"""校验 QMT 单日快照或连续运行记录。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.qmt_acceptance import (
    audit_qmt_snapshot_history,
    save_acceptance_report,
    validate_qmt_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 MiniQMT 只读同步结果")
    parser.add_argument("--snapshot-dir", type=Path, help="单日快照目录")
    parser.add_argument("--account-root", type=Path, help="包含多个日期目录的账户快照根目录")
    parser.add_argument("--ui-account", type=Path, help="从客户端抄录并标准化的账户 CSV")
    parser.add_argument("--ui-positions", type=Path, help="从客户端抄录并标准化的持仓 CSV")
    parser.add_argument("--amount-tolerance", type=float, default=1.0)
    parser.add_argument("--min-days", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.snapshot_dir) == bool(args.account_root):
        parser.error("--snapshot-dir 与 --account-root 必须且只能提供一个")

    if args.snapshot_dir:
        checks = validate_qmt_snapshot(
            args.snapshot_dir,
            ui_account_path=args.ui_account,
            ui_positions_path=args.ui_positions,
            amount_tolerance=args.amount_tolerance,
        )
        output = args.output or args.snapshot_dir / "acceptance_report.md"
        save_acceptance_report(checks, output)
        print(checks.to_string(index=False))
        print("report:", output)
        if "BLOCK" in set(checks["status"]):
            raise SystemExit(2)
    else:
        detail, summary = audit_qmt_snapshot_history(args.account_root, min_days=args.min_days)
        output = args.output or args.account_root / "continuous_acceptance.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(output, index=False)
        summary_path = output.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(detail.to_string(index=False))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["status"] != "PASS":
            raise SystemExit(3)


if __name__ == "__main__":
    main()
