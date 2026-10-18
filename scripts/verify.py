#!/usr/bin/env python
"""
verify.py
=========
Re-check every headline number in the manuscript against a results CSV, using
the check code that ships inside the package (`multiverse.verify`). Exits
non-zero if any check fails.

    python scripts/verify.py                      # uses results/results_all_full.csv
    python scripts/verify.py --csv path/to.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiverse import verify  # noqa: E402


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "results", "results_all_full.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    checks = verify.run_checks(df)
    fails = verify.print_report(checks)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
