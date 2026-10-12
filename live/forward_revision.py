"""Explicit, additive V2 -> cash-dividend revision; never reseal old records."""
from pathlib import Path
import pandas as pd

from live.cash_dividends import POLICY

REVISION_DIR = "revisions/cash_dividend_v1"
EXTRA_CODE = ["live/cash_dividends.py", "live/forward_revision.py", "live/account_state.py",
              "scripts/fetch_forward_actions.py", "scripts/run_forward_daily.py",
              "scripts/fetch_forward_market.py", "scripts/upgrade_forward_dividends.py"]


def effective_config(root, base):
    from live.forward_ledger import verify, read_json, sha, utc_now
    path = root / REVISION_DIR
    if not path.exists():
        return base
    meta = verify(path)
    review = read_json(path / "review.json")
    if (meta["initial_manifest_sha256"] != sha(root / "initial/manifest.json")
            or review["old_code_hashes"] != base["code_hashes"]
            or review["policy"] != POLICY):
        raise RuntimeError("invalid session revision ancestry/policy")
    anchor = root / "days" / review["anchor_date"].replace("-", "")
    verify(anchor)
    if sha(anchor / "manifest.json") != review["anchor_manifest_sha256"]:
        raise RuntimeError("session revision account anchor changed")
    if pd.Timestamp(meta["approved_at_utc"]) > utc_now():
        raise RuntimeError("future session revision")
    return {**base, "version": "SIX_STRATEGY_FORWARD_LEDGER_V3_CASH_DIVIDEND",
            "corporate_actions": POLICY, "dividend_tax_policy": "GROSS_DIVIDEND_TAX_NOT_MODELED",
            "revision_sha256": sha(path / "manifest.json"), "revision_effective_date": review["effective_date"],
            "code_hashes": review["new_code_hashes"]}


def upgrade(root: Path, review_path: Path):
    from live.forward_ledger import (ROOT, locked, verify, read_json, sha, latest_day,
                                     calendar_dates, next_session, staging, seal, utc_now)
    import shutil
    with locked(root):
        dest = root / REVISION_DIR
        if dest.exists():
            raise RuntimeError("revision already exists; never overwrite")
        verify(root / "initial")
        base = read_json(root / "initial/session.json")
        review = read_json(review_path)
        prior, meta = latest_day(root)
        if prior == root / "initial" or review["anchor_date"] != meta["date"]:
            raise RuntimeError("revision must anchor the latest committed daily account")
        following = str(next_session(calendar_dates(root / "initial/calendar.csv"), meta["date"]).date())
        if any(not (root / "days" / p.name).exists() for p in (root / "plans").glob("????????")):
            raise RuntimeError("cannot revise while a frozen plan is unexecuted")
        if review["effective_date"] != following or review["anchor_manifest_sha256"] != sha(prior / "manifest.json"):
            raise RuntimeError("revision date/account hash mismatch")
        old = base["code_hashes"]
        new = review["new_code_hashes"]
        if review["policy"] != POLICY or review["old_code_hashes"] != old or not review.get("reason"):
            raise RuntimeError("revision review is incomplete")
        if set(new) != set(old) | set(EXTRA_CODE):
            raise RuntimeError("revision code coverage mismatch")
        for name, digest in new.items():
            if sha(ROOT / name) != digest:
                raise RuntimeError("reviewed code changed: " + name)
            if name in old and name != "live/forward_ledger.py" and digest != old[name]:
                raise RuntimeError("strategy/execution change outside cash-dividend scope: " + name)
        with staging(dest.parent) as stage:
            shutil.copy2(review_path, stage / "review.json")
            seal(stage, {"kind": "EXPLICIT_CASH_DIVIDEND_REVISION", "approved_at_utc": utc_now().isoformat(),
                         "initial_manifest_sha256": sha(root / "initial/manifest.json"),
                         "retroactive_strategy_change": False, "live_approval": False})
            stage.rename(dest)
    return dest
