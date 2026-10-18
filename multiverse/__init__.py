"""recsys-multiverse: a multiverse / specification-curve analysis harness for
offline recommender-system evaluation."""
from .config import Spec, SpecSpace
from .data import (make_synthetic, load_csv, kcore_filter, split_data,
                   subsample_users, TrainData, Split)
from .models import (MODEL_REGISTRY, BaseModel, MostPop, ItemKNN, EASE, BPRMF,
                     BPRMFnp, LightGCN, SASRec)
from .runner import run_multiverse
from . import analysis
from . import figures
from . import verify
from .tuning import tune_models, DEFAULT_GRIDS

__all__ = [
    "Spec", "SpecSpace", "make_synthetic", "load_csv", "kcore_filter",
    "split_data", "TrainData", "Split", "MODEL_REGISTRY", "BaseModel",
    "MostPop", "ItemKNN", "EASE", "BPRMF", "LightGCN", "SASRec",
    "run_multiverse", "analysis", "figures", "verify", "tune_models", "DEFAULT_GRIDS",
]
