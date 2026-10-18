#!/usr/bin/env python
"""
check_data.py
=============
Data-preparation check: load each dataset through the same loaders used for the
paper, apply the MAX_USERS=10000 subsample, and confirm the resulting user /
item / interaction counts match Table 3 of the manuscript.

Requires network access to the dataset hosts (GroupLens, SNAP) and, for Yelp,
`yelp_academic_dataset_review.json` in the working directory. Run it where the
data is reachable (e.g. Colab):

    python scripts/check_data.py
    python scripts/check_data.py --datasets ml-100k ml-1m   # subset
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reproduce import LOADERS, MAX_USERS  # noqa: E402
from multiverse import subsample_users    # noqa: E402

# Manuscript Table 3 — (users, items, interactions) after loading + subsample.
TABLE3 = {
    "ml-100k":    (943, 1682, 100000),
    "ml-1m":      (6040, 3706, 1000209),
    "yelp":       (10000, 20469, 119722),
    "gowalla":    (10000, 253409, 580420),
    "amazon-cds": (10000, 46766, 140513),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(TABLE3))
    args = ap.parse_args()

    print(f"{'dataset':<12}{'users':>10}{'items':>10}{'interactions':>14}   vs Table 3")
    fails = 0
    for name in args.datasets:
        try:
            df = LOADERS[name]()
            if MAX_USERS and df["user"].nunique() > MAX_USERS:
                df = subsample_users(df, MAX_USERS, seed=0)
        except Exception as e:
            print(f"{name:<12}  load failed: {type(e).__name__}: {e}")
            fails += 1
            continue
        u, i, n = df["user"].nunique(), df["item"].nunique(), len(df)
        eu, ei, en = TABLE3[name]
        ok = (u == eu and i == ei and n == en)
        fails += 0 if ok else 1
        print(f"{name:<12}{u:>10,}{i:>10,}{n:>14,}   "
              f"{'PASS' if ok else f'FAIL (expected {eu:,}/{ei:,}/{en:,})'}")
    print(f"\n{len(args.datasets) - fails}/{len(args.datasets)} datasets match Table 3.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
