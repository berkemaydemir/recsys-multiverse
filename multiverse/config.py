"""
config.py
=========
The *multiverse* of defensible evaluation/training pipelines.

Degrees of freedom (DoFs):
    - kcore        : iterative k-core density filtering threshold
    - split        : train/test split strategy
    - candidates   : evaluation candidate set (full ranking vs sampled negatives)
    - n_negatives  : #sampled negatives (only when candidates == 'sampled')
    - train_neg    : training-time negative sampling (uniform vs popularity);
                     only affects models with uses_train_neg == True
    - seed         : random seed

Metric and cutoff K are not spec axes (one trained model is scored at all
metrics/cutoffs at once); they are handled as additional *analysis* axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Iterator


@dataclass(frozen=True)
class Spec:
    kcore: int
    split: str
    candidates: str
    n_negatives: int
    train_neg: str
    seed: int

    def key(self) -> str:
        neg = self.n_negatives if self.candidates == "sampled" else "-"
        return (f"k{self.kcore}|{self.split}|{self.candidates}|neg{neg}"
                f"|tn-{self.train_neg}|s{self.seed}")


@dataclass
class SpecSpace:
    kcore: List[int] = field(default_factory=lambda: [5, 10])
    split: List[str] = field(default_factory=lambda: [
        "random_holdout", "temporal_global", "leave_last_out"])
    candidates: List[str] = field(default_factory=lambda: ["full", "sampled"])
    n_negatives: List[int] = field(default_factory=lambda: [100])
    train_neg: List[str] = field(default_factory=lambda: ["uniform"])
    seed: List[int] = field(default_factory=lambda: [0, 1, 2])

    def enumerate(self) -> Iterator[Spec]:
        seen = set()
        for kcore, split, cand, nneg, tneg, seed in product(
            self.kcore, self.split, self.candidates, self.n_negatives,
            self.train_neg, self.seed
        ):
            nneg_eff = nneg if cand == "sampled" else 0
            spec = Spec(kcore=kcore, split=split, candidates=cand,
                        n_negatives=nneg_eff, train_neg=tneg, seed=seed)
            if spec.key() in seen:
                continue
            seen.add(spec.key())
            yield spec

    def size(self) -> int:
        return sum(1 for _ in self.enumerate())

    @property
    def factor_names(self) -> List[str]:
        names = []
        if len(self.kcore) > 1: names.append("kcore")
        if len(self.split) > 1: names.append("split")
        if len(self.candidates) > 1: names.append("candidates")
        if len(self.train_neg) > 1: names.append("train_neg")
        if len(self.seed) > 1: names.append("seed")
        return names
