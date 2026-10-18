"""
metrics.py
==========
Top-N ranking metrics computed from a ranked list of candidate item ids and a
set of relevant (held-out positive) items. All metrics take the already-ranked
candidate ids so the candidate-set policy (full vs sampled) lives upstream.
"""
from __future__ import annotations

from typing import Iterable, Set

import numpy as np


def recall_at_k(ranked: np.ndarray, relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for it in topk if it in relevant)
    return hits / min(len(relevant), k)


def hit_at_k(ranked: np.ndarray, relevant: Set[int], k: int) -> float:
    return 1.0 if any(it in relevant for it in ranked[:k]) else 0.0


def ndcg_at_k(ranked: np.ndarray, relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, it in enumerate(ranked[:k]):
        if it in relevant:
            dcg += 1.0 / np.log2(rank + 2)
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def map_at_k(ranked: np.ndarray, relevant: Set[int], k: int) -> float:
    """Average Precision at k."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, it in enumerate(ranked[:k]):
        if it in relevant:
            hits += 1
            score += hits / (rank + 1)
    return score / min(len(relevant), k)


def mrr_at_k(ranked: np.ndarray, relevant: Set[int], k: int) -> float:
    """Reciprocal rank of the first hit within k."""
    for rank, it in enumerate(ranked[:k]):
        if it in relevant:
            return 1.0 / (rank + 1)
    return 0.0


METRICS = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit": hit_at_k,
    "map": map_at_k,
    "mrr": mrr_at_k,
}
