#!/usr/bin/env python
"""
make_figures.py
===============
Regenerate the paper's Figures 2, 3 and 4 from a results CSV, using the figure
code that ships inside the package (`multiverse.figures`).

    python scripts/make_figures.py                       # uses results/results_all_full.csv
    python scripts/make_figures.py --csv path/to.csv --out figures/
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiverse import figures  # noqa: E402


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "results", "results_all_full.csv"),
                    help="results frame (default: results/results_all_full.csv)")
    ap.add_argument("--out", default=os.path.join(here, "figures"),
                    help="output directory (default: figures/)")
    ap.add_argument("--metric", default="ndcg")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    info = figures.make_all(df, outdir=args.out, metric=args.metric, k=args.k)
    print(f"Figures 2-4 written to {args.out}/")
    print(f"Figure 3 candidate gradient: {info['figure3_candidate_gradient']}")


if __name__ == "__main__":
    main()
