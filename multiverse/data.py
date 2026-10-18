"""
data.py
=======
Dataset loading, density filtering, and train/test splitting.

Interactions are represented as a pandas DataFrame with columns:
    user (int), item (int), ts (int/float timestamp)

Implicit feedback only: presence of a (user, item) row = positive signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def make_synthetic(n_users: int = 800, n_items: int = 400,
                   n_interactions: int = 25_000, pop_skew: float = 1.3,
                   seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic implicit-feedback log with a skewed popularity
    distribution and a coherent global timeline. Useful for testing the
    harness end-to-end without downloading real data."""
    rng = np.random.default_rng(seed)
    # Zipf-like item popularity
    item_pop = 1.0 / np.power(np.arange(1, n_items + 1), pop_skew)
    item_pop /= item_pop.sum()
    # Each user has a latent "taste" toward a contiguous slice of items
    users = rng.integers(0, n_users, size=n_interactions)
    centers = (users / n_users * n_items).astype(int)
    noise = rng.normal(0, n_items * 0.12, size=n_interactions).astype(int)
    taste_items = np.clip(centers + noise, 0, n_items - 1)
    # Mix taste-driven and popularity-driven picks
    pop_items = rng.choice(n_items, size=n_interactions, p=item_pop)
    use_pop = rng.random(n_interactions) < 0.4
    items = np.where(use_pop, pop_items, taste_items)
    ts = np.sort(rng.integers(0, n_interactions * 3, size=n_interactions))
    df = pd.DataFrame({"user": users, "item": items, "ts": ts})
    df = df.drop_duplicates(["user", "item"], keep="first").reset_index(drop=True)
    return df


def subsample_users(df: pd.DataFrame, max_users: int, seed: int = 0) -> pd.DataFrame:
    """Randomly keep at most `max_users` users (keeps full histories of those
    users). Use to keep Amazon/Gowalla-scale logs tractable on Colab."""
    users = df["user"].unique()
    if len(users) <= max_users:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(users, size=max_users, replace=False))
    return df[df["user"].isin(keep)].reset_index(drop=True)


def load_csv(path: str, user_col="user", item_col="item", ts_col="ts",
             sep=",") -> pd.DataFrame:
    """Load a real dataset. Point this at MovieLens / Amazon / Yelp / Gowalla
    exports with (user, item, timestamp) columns."""
    df = pd.read_csv(path, sep=sep)
    df = df.rename(columns={user_col: "user", item_col: "item", ts_col: "ts"})
    df = df[["user", "item", "ts"]].dropna()
    df = df.drop_duplicates(["user", "item"], keep="first").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def kcore_filter(df: pd.DataFrame, kcore: int) -> pd.DataFrame:
    """Iteratively remove users/items with fewer than `kcore` interactions
    until the graph is stable (a standard but consequential preprocessing DoF)."""
    if kcore <= 1:
        return df.reset_index(drop=True)
    cur = df
    while True:
        uc = cur["user"].value_counts()
        ic = cur["item"].value_counts()
        keep_u = uc[uc >= kcore].index
        keep_i = ic[ic >= kcore].index
        nxt = cur[cur["user"].isin(keep_u) & cur["item"].isin(keep_i)]
        if len(nxt) == len(cur):
            break
        cur = nxt
    return cur.reset_index(drop=True)


def reindex(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Map user/item ids to contiguous 0..N integers. Returns (df, n_users, n_items)."""
    u_map = {u: i for i, u in enumerate(df["user"].unique())}
    i_map = {it: i for i, it in enumerate(df["item"].unique())}
    df = df.assign(user=df["user"].map(u_map), item=df["item"].map(i_map))
    return df.reset_index(drop=True), len(u_map), len(i_map)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    train: pd.DataFrame                 # interactions used for training
    test: Dict[int, List[int]]          # user -> held-out positive items
    n_users: int
    n_items: int


@dataclass
class TrainData:
    """Everything a model might need at fit time. Matrix-factorization / kNN
    models use `X`; sequential models (SASRec, BERT4Rec) use `train_df` (which
    carries the timestamp column so per-user sequences can be reconstructed)."""
    X: "object"                         # scipy.sparse.csr_matrix (users x items)
    train_df: pd.DataFrame              # columns: user, item, ts
    n_users: int
    n_items: int


def split_data(df: pd.DataFrame, strategy: str, seed: int,
               test_frac: float = 0.2) -> Split:
    """Produce a train/test split under the chosen strategy.

    - random_holdout : per-user random hold-out of `test_frac` of items
    - temporal_global: single global timestamp cut (last `test_frac` by time)
    - leave_last_out : per-user hold out the single most recent item
    """
    df, n_users, n_items = reindex(df)
    rng = np.random.default_rng(seed)
    test: Dict[int, List[int]] = {}

    if strategy == "temporal_global":
        cut = df["ts"].quantile(1 - test_frac)
        train = df[df["ts"] <= cut]
        test_df = df[df["ts"] > cut]
        # Only keep test users/items seen in train (avoid cold-start contamination)
        seen_u = set(train["user"]); seen_i = set(train["item"])
        for u, grp in test_df.groupby("user"):
            if u not in seen_u:
                continue
            items = [it for it in grp["item"].tolist() if it in seen_i]
            if items:
                test[u] = items
        train = train.reset_index(drop=True)

    elif strategy == "leave_last_out":
        rows_train = []
        for u, grp in df.sort_values("ts").groupby("user"):
            items = grp["item"].tolist()
            if len(items) < 2:
                rows_train.extend(grp.index.tolist())
                continue
            rows_train.extend(grp.index.tolist()[:-1])
            test[u] = [items[-1]]
        train = df.loc[rows_train].reset_index(drop=True)

    elif strategy == "random_holdout":
        rows_train = []
        for u, grp in df.groupby("user"):
            idx = grp.index.to_numpy().copy()
            if len(idx) < 2:
                rows_train.extend(idx.tolist())
                continue
            rng.shuffle(idx)
            n_test = max(1, int(round(len(idx) * test_frac)))
            test_idx = idx[:n_test]
            rows_train.extend(idx[n_test:].tolist())
            test[u] = df.loc[test_idx, "item"].tolist()
        train = df.loc[rows_train].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return Split(train=train, test=test, n_users=n_users, n_items=n_items)
