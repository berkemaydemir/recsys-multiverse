"""
figures.py
==========
Regenerate the paper's data figures directly from a results frame
(as produced by `run_multiverse` / `results_all_full.csv`):

    Figure 2  figure_variance   -- per-model variance decomposition (stacked eta^2)
    Figure 3  figure_gradient   -- candidate vs split vs other, personalised-averaged
    Figure 4  figure_speccurves -- spec curves for the most fragile top-3 pair

All three use the same within-model one-way eta^2 as `analysis`; Figure 3 averages
over the *personalised* models only (MostPop excluded), matching the manuscript.
Matplotlib is imported lazily so importing the package never requires a display.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import analysis

# ----- presentation constants (Okabe-Ito colourblind-safe palette) ---------- #
ORDER = ["ml-100k", "ml-1m", "yelp", "amazon-cds", "gowalla"]
NICE = {"ml-100k": "MovieLens-100K", "ml-1m": "MovieLens-1M", "yelp": "Yelp",
        "amazon-cds": "Amazon CDs & Vinyl", "gowalla": "Gowalla"}
XTICK = ["ML-100K\n(1.7K)", "ML-1M\n(3.7K)", "Yelp\n(20K)",
         "Amazon\n(47K)", "Gowalla\n(253K)"]
COL = {"candidates": "#0072B2", "split": "#D55E00", "kcore": "#009E73",
       "train_neg": "#CC79A7", "seed": "#999999"}
FACTORS = ["candidates", "split", "kcore", "train_neg", "seed"]
FNICE = {"candidates": "candidate", "split": "split", "kcore": "k-core",
         "train_neg": "train-neg", "seed": "seed"}
MODELS = ["MostPop", "ItemKNN", "EASE", "BPRMF", "LightGCN", "SASRec"]
PERS = ["ItemKNN", "EASE", "BPRMF", "LightGCN", "SASRec"]   # personalised (no MostPop)
SPLIT_COL = {"random_holdout": "#0072B2", "temporal_global": "#D55E00",
             "leave_last_out": "#009E73"}


def _eta_model(d, factor, model, metric="ndcg", k=10):
    """Within-model one-way eta^2 of `factor` for a single model."""
    sub = d[(d.metric == metric) & (d.k == k) & (d.model == model)]
    if len(sub) < 2:
        return np.nan
    g = sub["value"].mean()
    sst = ((sub["value"] - g) ** 2).sum()
    if sst == 0:
        return 0.0
    ssf = sum(len(gr) * (gr["value"].mean() - g) ** 2 for _, gr in sub.groupby(factor))
    return ssf / sst


def _save(fig, out_png, out_pdf):
    import matplotlib.pyplot as plt
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    if out_pdf:
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 -- candidate-sampling gradient (personalised-averaged)
# --------------------------------------------------------------------------- #
def figure_gradient(df, metric="ndcg", k=10, out_png=None, out_pdf=None):
    import matplotlib.pyplot as plt
    order = [ds for ds in ORDER if ds in df["dataset"].unique()]
    cand, split, other = [], [], []
    for ds in order:
        d = df[df.dataset == ds]
        ph = [m for m in PERS if m in d["model"].unique()]
        cand.append(np.nanmean([_eta_model(d, "candidates", m, metric, k) for m in ph]))
        split.append(np.nanmean([_eta_model(d, "split", m, metric, k) for m in ph]))
        other.append(np.nanmean([sum(_eta_model(d, f, m, metric, k)
                                     for f in ["kcore", "train_neg", "seed"])
                                 for m in ph]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(order))
    ax.plot(x, cand, "o-", color=COL["candidates"], lw=2.2, ms=8, label="candidate-sampling")
    ax.plot(x, split, "s--", color=COL["split"], lw=1.8, ms=7, label="split strategy")
    ax.plot(x, other, "^:", color=COL["kcore"], lw=1.6, ms=7, label="all other decisions")
    ax.set_xticks(list(x))
    ax.set_xticklabels([XTICK[ORDER.index(ds)] for ds in order], fontsize=9)
    ax.set_ylabel("Mean variance share (η²)", fontsize=11)
    ax.set_xlabel("Dataset (ordered by catalogue size)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="center right")
    for xi, c in zip(x, cand):
        ax.annotate(f"{c:.2f}", (xi, c), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=COL["candidates"], fontweight="bold")
    fig.tight_layout()
    _save(fig, out_png, out_pdf)
    return [f"{c:.2f}" for c in cand]


# --------------------------------------------------------------------------- #
# Figure 2 -- per-model variance decomposition (stacked eta^2)
# --------------------------------------------------------------------------- #
def figure_variance(df, metric="ndcg", k=10, out_png=None, out_pdf=None):
    import matplotlib.pyplot as plt
    order = [ds for ds in ORDER if ds in df["dataset"].unique()]
    panel = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    fig, axes = plt.subplots(len(order), 1, figsize=(7.5, 2.6 * len(order)))
    if len(order) == 1:
        axes = [axes]
    for idx, ds in enumerate(order):
        ax = axes[idx]
        d = df[df.dataset == ds]
        mh = [m for m in MODELS if m in d["model"].unique()]
        bottoms = np.zeros(len(mh))
        for f in FACTORS:
            vals = [_eta_model(d, f, m, metric, k) for m in mh]
            vals = [0 if np.isnan(v) else v for v in vals]
            ax.barh(range(len(mh)), vals, left=bottoms, color=COL[f],
                    label=FNICE[f] if idx == 0 else "", height=0.65)
            bottoms += np.array(vals)
        ax.set_yticks(range(len(mh)))
        ax.set_yticklabels(mh, fontsize=9)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title(f"{panel[idx]} {NICE[ds]}", fontsize=10, loc="left")
        ax.set_xlabel("Variance share (η²)" if idx == len(order) - 1 else "", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].legend(ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.tight_layout()
    _save(fig, out_png, out_pdf)


# --------------------------------------------------------------------------- #
# Figure 4 -- spec curves for the most fragile pair among the top-3 contenders
# --------------------------------------------------------------------------- #
def _fragile_top_pair(d, metric, k):
    """(a, b) = most fragile pair among the three highest-mean models."""
    sub = d[(d.metric == metric) & (d.k == k)]
    means = sub.groupby("model")["value"].mean().sort_values(ascending=False)
    top3 = list(means.index[:3])
    frag = analysis.fragility_index(d, metric, k)
    frag = frag[frag.model_a.isin(top3) & frag.model_b.isin(top3)]
    if frag.empty:
        return None
    r = frag.iloc[0]          # fragility_index is sorted desc by fragility
    return r["model_a"], r["model_b"]


def figure_speccurves(df, metric="ndcg", k=10, out_png=None, out_pdf=None):
    import matplotlib.pyplot as plt
    order = [ds for ds in ORDER if ds in df["dataset"].unique()]
    panel = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    fig, axes = plt.subplots(len(order), 1, figsize=(7.5, 2.4 * len(order)))
    if len(order) == 1:
        axes = [axes]
    for idx, ds in enumerate(order):
        ax = axes[idx]
        d = df[df.dataset == ds]
        pair = _fragile_top_pair(d, metric, k)
        if pair is None:
            ax.set_visible(False)
            continue
        a, b = pair
        sc = analysis.spec_curve(d, a, b, metric, k)       # sorted gap + split
        colors = [SPLIT_COL.get(s, "#999999") for s in sc["split"]]
        ax.bar(range(len(sc)), sc["gap"], color=colors, width=0.9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{panel[idx]} {NICE[ds]}: {a} vs {b}", fontsize=10, loc="left")
        ax.set_ylabel(f"{metric}@{k} gap", fontsize=9)
        if idx == len(order) - 1:
            ax.set_xlabel("specification (sorted by gap)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    # shared split-strategy legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=s) for s, c in SPLIT_COL.items()]
    fig.legend(handles=handles, title="split strategy", ncol=3,
               loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out_png, out_pdf)


# --------------------------------------------------------------------------- #
def make_all(df, outdir="figures", metric="ndcg", k=10):
    """Regenerate Figures 2, 3 and 4 (PNG + PDF) into `outdir`."""
    known = [ds for ds in ORDER if ds in df["dataset"].unique()]
    if not known:
        raise ValueError(
            "No recognised datasets in the results frame. figures expects the "
            f"paper datasets {ORDER}; got {sorted(df['dataset'].unique())}.")
    os.makedirs(outdir, exist_ok=True)
    figure_variance(df, metric, k,
                    os.path.join(outdir, "figure2_variance.png"),
                    os.path.join(outdir, "figure2_variance.pdf"))
    grad = figure_gradient(df, metric, k,
                           os.path.join(outdir, "figure3_gradient.png"),
                           os.path.join(outdir, "figure3_gradient.pdf"))
    figure_speccurves(df, metric, k,
                      os.path.join(outdir, "figure4_speccurves.png"),
                      os.path.join(outdir, "figure4_speccurves.pdf"))
    return {"figure3_candidate_gradient": grad}
