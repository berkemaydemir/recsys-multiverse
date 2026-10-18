"""
tuning.py
=========
Validation-based hyperparameter tuning. For each model, grid-search a small
hyperparameter grid on a held-out validation split (carved from the data with a
random hold-out) and keep the configuration with the best NDCG@k. The tuned
configs are then frozen and used across the whole multiverse.

This pre-empts the standard reviewer objection that claimed differences come
from unequal tuning effort: every model is tuned under identical conditions and
the same budget search.
"""
from __future__ import annotations

import itertools
from typing import Dict

import numpy as np

from .config import Spec
from .data import kcore_filter, split_data, TrainData
from .metrics import METRICS
from .models import MODEL_REGISTRY
from .runner import _build_csr, _evaluate


# A small, sensible default grid per model. Override per dataset as needed.
DEFAULT_GRIDS: Dict[str, dict] = {
    "ItemKNN":  {"topk": [50, 100, 200]},
    "BPRMF":    {"dim": [64], "lr": [5e-3, 1e-2], "reg": [1e-5, 1e-4]},
    "LightGCN": {"dim": [64], "n_layers": [2, 3], "lr": [1e-2]},
    "SASRec":   {"dim": [64], "lr": [1e-3], "maxlen": [50, 100]},
    # MostPop has no hyperparameters
}


def tune_models(df, models, grids=None, base_kwargs=None, kcore=5, seed=0,
                cutoff=10, verbose=True) -> Dict[str, dict]:
    """Return {model: best_kwargs} chosen on a validation split.

    Tuning uses a single random hold-out validation split (full-ranking NDCG@k),
    holding all else fixed; the winning config is then frozen for the multiverse.
    """
    grids = grids or DEFAULT_GRIDS
    base_kwargs = base_kwargs or {}
    fdf = kcore_filter(df, kcore)
    split = split_data(fdf, "random_holdout", seed)
    X = _build_csr(split.train, split.n_users, split.n_items)
    # degenerate after filtering -> skip tuning, fall back to base kwargs
    if X.nnz == 0 or not split.test or split.n_items < cutoff:
        if verbose:
            print("  [warn] data too sparse to tune; using base kwargs")
        return {m: dict(base_kwargs.get(m, {})) for m in models}
    data = TrainData(X=X, train_df=split.train, n_users=split.n_users, n_items=split.n_items)
    spec = Spec(kcore=kcore, split="random_holdout", candidates="full",
                n_negatives=0, train_neg="uniform", seed=seed)

    best: Dict[str, dict] = {}
    for m in models:
        grid = grids.get(m)
        if not grid:                       # nothing to tune (e.g., MostPop)
            best[m] = dict(base_kwargs.get(m, {}))
            continue
        keys = list(grid)
        best_score, best_kw = -1.0, {}
        for combo in itertools.product(*grid.values()):
            kw = dict(base_kwargs.get(m, {}))
            kw.update(dict(zip(keys, combo)))
            try:
                model = MODEL_REGISTRY[m](split.n_users, split.n_items, seed=seed, **kw)
                model.fit(data)
                if getattr(model, "_X", None) is None:
                    model._X = X
                score = _evaluate(model, split, spec, (cutoff,), ("ndcg",),
                                  np.random.default_rng(seed))[("ndcg", cutoff)]
            except Exception as e:
                if verbose:
                    print(f"  [warn] {m} config {dict(zip(keys, combo))} failed: "
                          f"{type(e).__name__}")
                continue
            if score > best_score:
                best_score, best_kw = score, dict(zip(keys, combo))
        merged = dict(base_kwargs.get(m, {})); merged.update(best_kw)
        best[m] = merged
        if verbose:
            print(f"  tuned {m}: {best_kw}  (val ndcg@{cutoff}={best_score:.4f})")
    return best
