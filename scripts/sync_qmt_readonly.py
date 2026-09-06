"""同步 MiniQMT 真实账户只读快照。"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.broker import RealBrokerConfig
from live.qmt_readonly import QmtReadOnlyAdapter, save_qmt_readonly_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同步 MiniQMT 真实账户只读快照（不下单、不撤单）")
    parser.add_argument("--account-id", default=os.environ.get("QUANT_BROKER_ACCOUNT_ID", ""))
    parser.add_argument("--userdata-path", default=os.environ.get("QUANT_QMT_USERDATA_PATH", ""))
    session_default = os.environ.get("QUANT_QMT_SESSION_ID", "").strip()
    parser.add_argument("--session-id", type=int, default=int(session_default) if session_default else None,
                        help="须与其他 QMT 会话不同；默认使用当前时间生成")
    parser.add_argument("--account-type", default=os.environ.get("QUANT_QMT_ACCOUNT_TYPE", "STOCK"),
                        choices=["STOCK", "CREDIT"], help="普通股票或信用账户")
    parser.add_argument("--trade-date")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "broker_snapshots" / "qmt")
    args = parser.parse_args()
    if not args.account_id or not args.userdata_path:
        parser.error("须提供 --account-id 和 --userdata-path（也可使用对应环境变量）")
    userdata = Path(args.userdata_path)
    if not userdata.is_dir():
        parser.error("userdata_mini 目录不存在: %s" % userdata)
    if args.session_id is None:
        args.session_id = int(time.time())
    return args


def main() -> None:
    args = parse_args()
    try:
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
    except ImportError as exc:
        raise SystemExit("当前环境未安装 QMT 的 xtquant SDK；请在运行 MiniQMT 的 Windows 环境执行") from exc

    trader = XtQuantTrader(str(Path(args.userdata_path).resolve()), args.session_id)
    trader.start()
    try:
        connect_result = trader.connect()
        if connect_result != 0:
            raise SystemExit("连接 MiniQMT 失败，connect_result=%s" % connect_result)
        account_ref = StockAccount(args.account_id, args.account_type)
        subscribe_result = trader.subscribe(account_ref)
        if subscribe_result != 0:
            raise SystemExit("订阅股票账户失败，subscribe_result=%s" % subscribe_result)
        try:
            import xtquant
            sdk_version = getattr(xtquant, "__version__", "unknown")
        except Exception:
            sdk_version = "unknown"
        adapter = QmtReadOnlyAdapter(
            RealBrokerConfig(provider="qmt", account_id=args.account_id),
            trader=trader,
            account_ref=account_ref,
            connection_info={
                "connect_result": connect_result,
                "subscribe_result": subscribe_result,
                "session_id": args.session_id,
                "account_type": args.account_type,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "xtquant_version": sdk_version,
            },
        )
        paths = save_qmt_readonly_snapshot(adapter, args.output_dir, trade_date=args.trade_date)
        for name, path in paths.items():
            print(f"{name}: {path}")
    finally:
        trader.stop()


if __name__ == "__main__":
    main()
