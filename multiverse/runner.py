"""
runner.py
=========
The multiverse engine. For every defensible pipeline (Spec) x every model,
trains, evaluates under the spec's candidate policy, and emits one tidy row per
(spec, model, metric, cutoff).

Evaluation is vectorized: per-user scores are computed in chunks via
`scores_batch`, and full-ranking top-K uses a batched argpartition (orders of
magnitude faster than the per-user Python loop on large catalogs).

`train_neg` is injected into models that train with negatives (uses_train_neg);
models that don't are run once (at the canonical train_neg) to avoid duplicate
work and double counting.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import Spec, SpecSpace
from .data import kcore_filter, split_data, TrainData
from .metrics import METRICS
from .models import MODEL_REGISTRY


def _build_csr(train, n_users, n_items):
    return sp.csr_matrix((np.ones(len(train)), (train["user"].to_numpy(),
                          train["item"].to_numpy())), shape=(n_users, n_items))


def _evaluate(model, split, spec, cutoffs, metric_names, rng):
    n_items = split.n_items
    maxk = max(cutoffs)
    test_users = list(split.test.keys())
    seen = {u: np.fromiter(set(split.train.loc[split.train["user"] == u, "item"]), int)
            for u in test_users}
    acc = {(m, k): [] for m in metric_names for k in cutoffs}
    # bound chunk memory to ~ a few hundred MB regardless of catalog size
    chunk = max(1, min(512, 4_000_000 // max(1, n_items)))

    for c0 in range(0, len(test_users), chunk):
        cu = test_users[c0:c0 + chunk]
        S = model.scores_batch(cu).astype(float)           # (B, n_items)

        if spec.candidates == "full":
            for b, u in enumerate(cu):
                sv = seen[u]
                if sv.size:
                    S[b, sv] = -np.inf
            negS = -S
            part = np.argpartition(negS, min(maxk, n_items - 1), axis=1)[:, :maxk]
            rows = np.arange(len(cu))[:, None]
            order = np.argsort(negS[rows, part], axis=1)
            topk = part[rows, order]                        # (B, maxk) best-first
            for b, u in enumerate(cu):
                rel = set(split.test[u]); ranked = topk[b]
                for m in metric_names:
                    fn = METRICS[m]
                    for k in cutoffs:
                        acc[(m, k)].append(fn(ranked, rel, k))
        else:  # sampled negatives per user
            for b, u in enumerate(cu):
                rel = set(split.test[u])
                forbidden = np.union1d(seen[u], np.fromiter(rel, int))
                pool = np.setdiff1d(np.arange(n_items), forbidden)
                n_neg = min(spec.n_negatives, len(pool))
                negs = rng.choice(pool, size=n_neg, replace=False) if n_neg else np.array([], int)
                cand = np.concatenate([np.fromiter(rel, int), negs])
                ranked = cand[np.argsort(-S[b, cand])]
                for m in metric_names:
                    fn = METRICS[m]
                    for k in cutoffs:
                        acc[(m, k)].append(fn(ranked, rel, k))

    return {mk: float(np.mean(v)) if v else 0.0 for mk, v in acc.items()}


def _process_spec(df, spec, models, cutoffs, metric_names, model_kwargs, canonical_tn):
    fdf = kcore_filter(df, spec.kcore)
    if len(fdf) == 0:
        return []
    split = split_data(fdf, spec.split, spec.seed)
    if not split.test:
        return []
    X = _build_csr(split.train, split.n_users, split.n_items)
    # skip specs whose filtering/splitting left no usable training signal
    if X.nnz == 0 or split.n_items < max(cutoffs) or len(split.test) < 2:
        return []
    data = TrainData(X=X, train_df=split.train, n_users=split.n_users, n_items=split.n_items)
    rows = []
    for mname in models:
        cls = MODEL_REGISTRY[mname]
        if (not cls.uses_train_neg) and spec.train_neg != canonical_tn:
            continue
        kw = dict(model_kwargs.get(mname, {}))
        if cls.uses_train_neg:
            kw["neg_sampling"] = spec.train_neg
        t0 = time.time()
        try:                                  # one bad cell must not kill the run
            model = cls(split.n_users, split.n_items, seed=spec.seed, **kw)
            model.fit(data)
            if getattr(model, "_X", None) is None:
                model._X = X
            # Fresh RNG per model, seeded by the spec, so that under sampled
            # evaluation every model is scored against the *same* negative
            # candidate set for each user (identical conditions across models).
            res = _evaluate(model, split, spec, cutoffs, metric_names,
                            np.random.default_rng(spec.seed))
        except Exception as e:
            print(f"  [warn] skipped {mname} on spec {spec.key()}: "
                  f"{type(e).__name__}: {e}")
            continue
        for (m, k), val in res.items():
            rows.append({
                **{f: getattr(spec, f) for f in
                   ("kcore", "split", "candidates", "n_negatives", "train_neg", "seed")},
                "spec": spec.key(), "model": mname, "metric": m, "k": k, "value": val,
                "n_users": split.n_users, "n_items": split.n_items,
                "n_test_users": len(split.test), "fit_secs": round(time.time() - t0, 3),
            })
    return rows


def run_multiverse(df, space, models=None, cutoffs=(5, 10, 20),
                   metric_names=("recall", "ndcg", "hit"), model_kwargs=None,
                   n_jobs=1, verbose=True):
    models = models or list(MODEL_REGISTRY.keys())
    model_kwargs = model_kwargs or {}
    specs = list(space.enumerate())
    canonical_tn = space.train_neg[0]
    if verbose:
        print(f"[multiverse] {len(specs)} specs x {len(models)} models | n_jobs={n_jobs}")

    if n_jobs == 1:
        rows = []
        for si, spec in enumerate(specs):
            rows += _process_spec(df, spec, models, cutoffs, metric_names,
                                  model_kwargs, canonical_tn)
            if verbose and (si + 1) % max(1, len(specs) // 10) == 0:
                print(f"  .. {si + 1}/{len(specs)} specs done")
    else:
        from joblib import Parallel, delayed
        batches = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_process_spec)(df, spec, models, cutoffs, metric_names,
                                   model_kwargs, canonical_tn) for spec in specs)
        rows = [r for batch in batches for r in batch]
    return pd.DataFrame(rows)
