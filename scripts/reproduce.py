#!/usr/bin/env python
"""
reproduce.py
============
Standalone entry point for the multiverse analysis — no Colab, no Google Drive.

Runs the full six-model multiverse over one dataset, writes a tidy results CSV
and a standardised multiverse report, and prints the headline robustness numbers.

Examples
--------
    # synthetic smoke test (no downloads, ~1 min):
    python scripts/reproduce.py --dataset synthetic --quick

    # a real dataset at paper settings:
    python scripts/reproduce.py --dataset ml-100k --out results/

Datasets `ml-100k`, `ml-1m`, `gowalla`, `amazon-cds` download automatically.
`yelp` expects `yelp_academic_dataset_review.json` in the working directory
(Yelp Open Dataset; manual download due to its licence).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

# make the package importable whether run from repo root or scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiverse import (SpecSpace, run_multiverse, subsample_users,  # noqa: E402
                        kcore_filter, make_synthetic, analysis)
from multiverse.tuning import tune_models  # noqa: E402

MAX_USERS = 10_000
CUTOFFS = (10, 20)
METRICS = ("recall", "ndcg", "map", "mrr")


# --------------------------------------------------------------------------- #
# Dataset loaders (identical to notebooks/01_run_multiverse.ipynb)
# --------------------------------------------------------------------------- #
def _dl(url, out):
    if not os.path.exists(out):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, out)
    return out


def load_ml_100k():
    _dl("https://files.grouplens.org/datasets/movielens/ml-100k.zip", "ml-100k.zip")
    if not os.path.exists("ml-100k"):
        zipfile.ZipFile("ml-100k.zip").extractall(".")
    df = pd.read_csv("ml-100k/u.data", sep="\t", names=["user", "item", "rating", "ts"])
    return df[["user", "item", "ts"]]


def load_ml_1m():
    _dl("https://files.grouplens.org/datasets/movielens/ml-1m.zip", "ml-1m.zip")
    if not os.path.exists("ml-1m"):
        zipfile.ZipFile("ml-1m.zip").extractall(".")
    df = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",
                     names=["user", "item", "rating", "ts"], encoding="latin-1")
    return df[["user", "item", "ts"]]


def load_gowalla():
    _dl("https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz", "gowalla.txt.gz")
    df = pd.read_csv("gowalla.txt.gz", sep="\t", compression="gzip",
                     names=["user", "time", "lat", "lon", "item"])
    df["ts"] = pd.to_datetime(df["time"], errors="coerce").astype("int64") // 10 ** 9
    df = df.dropna(subset=["item"])
    return df[["user", "item", "ts"]]


def load_amazon(category="CDs_and_Vinyl", min_core=5):
    url = ("https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
           f"ratings_{category}.csv")
    out = f"amazon_{category}.csv"
    _dl(url, out)
    df = pd.read_csv(out, names=["user", "item", "rating", "ts"])[["user", "item", "ts"]]
    return kcore_filter(df, min_core)       # densify BEFORE subsampling


def load_yelp(path="yelp_academic_dataset_review.json", min_core=5, max_rows=2_000_000):
    import json
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download the Yelp Open Dataset and place "
            "yelp_academic_dataset_review.json in the working directory.")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            r = json.loads(line)
            rows.append((r["user_id"], r["business_id"], r.get("date", "")))
    df = pd.DataFrame(rows, columns=["user", "item", "ts"])
    df["user"] = df["user"].astype("category").cat.codes
    df["item"] = df["item"].astype("category").cat.codes
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce").astype("int64") // 10 ** 9
    df = df.dropna(subset=["ts"])
    return kcore_filter(df, min_core)[["user", "item", "ts"]]


LOADERS = {
    "synthetic": lambda: make_synthetic(seed=0),
    "ml-100k": load_ml_100k,
    "ml-1m": load_ml_1m,
    "gowalla": load_gowalla,
    "amazon-cds": lambda: load_amazon("CDs_and_Vinyl"),
    "yelp": load_yelp,
}


def ease_feasible(n_items, cap_gb=60.0):
    """Dense EASE Gram matrix is O(n_items^2) float64."""
    return (n_items ** 2) * 8 / (1024 ** 3) <= cap_gb


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--quick", action="store_true",
                    help="small grid (kcore 5/10, seeds 0/1, no popularity neg) for a fast check")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip validation tuning and use base budgets")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/5] loading {args.dataset} ...")
    df = LOADERS[args.dataset]()
    if MAX_USERS and df["user"].nunique() > MAX_USERS:
        df = subsample_users(df, MAX_USERS, seed=0)
    print(f"      {len(df):,} interactions | {df['user'].nunique():,} users "
          f"| {df['item'].nunique():,} items")

    # ------- specification space -------
    if args.quick:
        space = SpecSpace(kcore=[5, 10],
                          split=["random_holdout", "temporal_global", "leave_last_out"],
                          candidates=["full", "sampled"], n_negatives=[100],
                          train_neg=["uniform"], seed=[0, 1])
    else:
        space = SpecSpace(kcore=[5, 10],
                          split=["random_holdout", "temporal_global", "leave_last_out"],
                          candidates=["full", "sampled"], n_negatives=[100],
                          train_neg=["uniform", "popularity"], seed=[0, 1, 2])

    # ------- models (drop EASE on catalogues too large for a dense Gram) -------
    n_items_est = kcore_filter(df, 5)["item"].nunique()
    models = ["MostPop", "ItemKNN", "EASE", "BPRMF", "LightGCN", "SASRec"]
    if not ease_feasible(n_items_est):
        models.remove("EASE")
        print(f"      EASE skipped (n_items≈{n_items_est:,}, dense Gram too large)")

    base_kwargs = {
        "ItemKNN":  {"topk": 200},
        "BPRMF":    {"epochs": 60, "dim": 64, "lr": 1e-2, "batch": 8192},
        "LightGCN": {"epochs": 50, "dim": 64, "n_layers": 3, "batch": 8192},
        "SASRec":   {"epochs": 100, "dim": 64, "n_blocks": 2, "n_heads": 2, "batch": 128},
    }
    grids = {
        "ItemKNN":  {"topk": [50, 100, 200]},
        "BPRMF":    {"dim": [64], "lr": [5e-3, 1e-2], "reg": [1e-5, 1e-4]},
        "LightGCN": {"dim": [64], "n_layers": [2, 3], "lr": [1e-2]},
        "SASRec":   {"dim": [64], "lr": [1e-3, 5e-4], "maxlen": [50, 200], "dropout": [0.2, 0.5]},
        "EASE":     {"reg": [100.0, 250.0, 500.0]},
    }

    print(f"[2/5] {space.size()} specs × {len(models)} models")
    if args.no_tune:
        mk = base_kwargs
    else:
        print("[3/5] tuning on a validation hold-out ...")
        mk = tune_models(df, [m for m in models if m in grids],
                         grids=grids, base_kwargs=base_kwargs, kcore=5, seed=0)
        for m in models:
            mk.setdefault(m, base_kwargs.get(m, {}))

    print("[4/5] running the multiverse ...")
    t0 = time.time()
    res = run_multiverse(df, space, models=models, cutoffs=CUTOFFS,
                         metric_names=METRICS, model_kwargs=mk, n_jobs=args.n_jobs)
    res["dataset"] = args.dataset
    csv_path = os.path.join(args.out, f"results_{args.dataset}.csv")
    res.to_csv(csv_path, index=False)
    print(f"      {len(res)} rows in {time.time() - t0:.0f}s -> {csv_path}")

    print("[5/5] analysis ...")
    frag, score = analysis.robustness_report(res, "ndcg", 10)
    print(f"\n  Leaderboard robustness score (ndcg@10): {score}  (1.0 = never reverses)")
    print("  Most fragile pairs:")
    print(frag[["model_a", "model_b", "majority_winner", "fragility", "verdict"]]
          .head(5).to_string(index=False))

    report = analysis.generate_report(
        res, metric="ndcg", k=10,
        factors=space.factor_names,
        path=os.path.join(args.out, "multiverse_report.md"))
    print(f"\n  report -> {report}")


if __name__ == "__main__":
    main()
