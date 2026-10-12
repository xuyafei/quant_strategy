#!/usr/bin/env python3
"""Install an explicitly reviewed, hash-pinned cash-dividend ledger revision."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live.forward_revision import upgrade

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    a = p.parse_args()
    print(upgrade(a.session, a.review))
