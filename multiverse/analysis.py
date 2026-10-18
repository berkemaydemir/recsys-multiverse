"""
analysis.py
===========
Turns the raw multiverse results into the paper's headline artifacts:

    1. pairwise_dominance  : P(model A > model B) across all specs
    2. fragility_index     : how often a claimed ordering reverses under
                             defensible alternative pipelines
    3. spec_curve          : the specification curve for one model pair
    4. variance_decomposition : which DoF drives the observed variation (eta^2)

These map directly onto the research questions:
    RQ1  How stable are method rankings across defensible evaluation pipelines?
    RQ2  Which methodological decisions dominate the outcome?
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _pivot(df: pd.DataFrame, metric: str, k: int) -> pd.DataFrame:
    """spec x model table of metric values for a fixed (metric, k)."""
    sub = df[(df["metric"] == metric) & (df["k"] == k)]
    return sub.pivot_table(index="spec", columns="model", values="value")


def pairwise_dominance(df: pd.DataFrame, metric: str, k: int) -> pd.DataFrame:
    """For each ordered pair (A, B): fraction of specs in which A scored
    strictly higher than B. 0.5 = a coin-flip ordering (maximally fragile)."""
    piv = _pivot(df, metric, k).dropna()
    models = list(piv.columns)
    out = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            if a == b:
                out.loc[a, b] = np.nan
            else:
                out.loc[a, b] = float((piv[a] > piv[b]).mean())
    return out


def fragility_index(df: pd.DataFrame, metric: str, k: int) -> pd.DataFrame:
    """Per unordered pair: a fragility score in [0, 0.5].

    fragility = min(p, 1-p) where p = P(A>B). 0 means the ordering is perfectly
    consistent across the multiverse; 0.5 means it is a coin flip. A 'claim'
    A>B is FRAGILE when fragility is high (the conclusion depends on arbitrary,
    defensible evaluation choices)."""
    piv = _pivot(df, metric, k).dropna()
    models = list(piv.columns)
    if piv.shape[0] == 0 or len(models) < 2:
        return pd.DataFrame(columns=["model_a", "model_b", "P(A>B)", "P(B>A)",
                                     "P(tie)", "majority_winner", "fragility", "n_specs"])
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            p = float((piv[a] > piv[b]).mean())          # P(A>B)
            q = float((piv[b] > piv[a]).mean())          # P(B>A)
            ties = float((piv[a] == piv[b]).mean())
            rows.append({
                "model_a": a, "model_b": b,
                "P(A>B)": round(p, 3),
                "P(B>A)": round(q, 3),
                "P(tie)": round(ties, 3),
                "majority_winner": a if p >= q else b,
                # paper §3.5: phi = min(p, q), p=P(A>B), q=P(B>A) (ties excluded)
                "fragility": round(min(p, q), 3),
                "n_specs": len(piv),
            })
    return pd.DataFrame(rows).sort_values("fragility", ascending=False).reset_index(drop=True)


def ranking_stability(df: pd.DataFrame, metric: str, k: int) -> dict:
    """How often does the per-spec top-ranked model equal the globally most
    common winner? 1.0 = the leaderboard winner never changes."""
    piv = _pivot(df, metric, k).dropna()
    if piv.shape[0] == 0 or piv.shape[1] == 0:
        return {"metric": metric, "k": k, "modal_winner": None,
                "stability": float("nan"), "distinct_winners": 0,
                "winner_distribution": {}}
    winners = piv.idxmax(axis=1)
    modal = winners.mode().iloc[0]
    return {
        "metric": metric, "k": k,
        "modal_winner": modal,
        "stability": round(float((winners == modal).mean()), 3),
        "distinct_winners": int(winners.nunique()),
        "winner_distribution": winners.value_counts(normalize=True).round(3).to_dict(),
    }


def spec_curve(df: pd.DataFrame, model_a: str, model_b: str,
               metric: str, k: int) -> pd.DataFrame:
    """Specification-curve data: the per-spec performance gap (A - B), sorted,
    with the DoF settings attached so the plot can be annotated by factor."""
    sub = df[(df["metric"] == metric) & (df["k"] == k) &
             (df["model"].isin([model_a, model_b]))]
    piv = sub.pivot_table(index=["spec", "kcore", "split", "candidates", "seed"],
                          columns="model", values="value").dropna().reset_index()
    piv["gap"] = piv[model_a] - piv[model_b]
    piv = piv.sort_values("gap").reset_index(drop=True)
    piv["rank"] = np.arange(len(piv))
    return piv


def summary_with_ci(df: pd.DataFrame, metric: str, k: int,
                    group=("model",), ci: float = 0.95) -> pd.DataFrame:
    """Mean +/- normal-approx confidence interval of a metric, aggregated over
    the multiverse (all specs/seeds). Use group=('model','split') etc. to slice.
    Reporting CIs over seeds/specs is a basic rigor requirement for this kind
    of paper."""
    from scipy import stats
    sub = df[(df["metric"] == metric) & (df["k"] == k)]
    if sub.empty:
        return pd.DataFrame(columns=list(group) + ["mean", "ci_low", "ci_high", "std", "n"])
    z = stats.norm.ppf(0.5 + ci / 2)
    rows = []
    for keys, g in sub.groupby(list(group)):
        vals = g["value"].to_numpy()
        m, sd, n = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, len(vals)
        half = z * sd / np.sqrt(n) if n > 1 else 0.0
        rec = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        rec.update({"mean": round(m, 4), "ci_low": round(m - half, 4),
                    "ci_high": round(m + half, 4), "std": round(sd, 4), "n": n})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def metric_robustness(df, k=10, metrics=("ndcg", "recall", "map", "mrr")):
    """Does the conclusion depend on the metric family? Reports, per metric, the
    modal leaderboard winner and the most-fragile pair, and flags disagreement.
    Treating metric choice as a defensible DoF strengthens the 'joint' claim."""
    metrics = [m for m in metrics if m in df["metric"].unique()]
    rows = []
    for m in metrics:
        stab = ranking_stability(df, m, k)
        frag = fragility_index(df, m, k).iloc[0]
        rows.append({"metric": f"{m}@{k}", "modal_winner": stab["modal_winner"],
                     "top1_stability": stab["stability"],
                     "most_fragile_pair": f"{frag['model_a']}/{frag['model_b']}",
                     "fragility": frag["fragility"]})
    out = pd.DataFrame(rows)
    out.attrs["winner_agrees"] = out["modal_winner"].nunique() == 1
    return out


def variance_decomposition(df: pd.DataFrame, target_model: str,
                           metric: str, k: int,
                           factors: List[str]) -> pd.DataFrame:
    """Marginal (one-way) eta^2 for each DoF: the share of variance in a model's
    score explained by that factor alone. Interpretable first-pass attribution;
    note it ignores interactions (a full ANOVA is a natural extension)."""
    sub = df[(df["metric"] == metric) & (df["k"] == k) &
             (df["model"] == target_model)].copy()
    y = sub["value"].to_numpy()
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    rows = []
    for f in factors:
        ss_between = 0.0
        for _, g in sub.groupby(f):
            gv = g["value"].to_numpy()
            ss_between += len(gv) * (gv.mean() - grand) ** 2
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        rows.append({"factor": f, "eta_squared": round(eta2, 3)})
    return (pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# (4) Full ANOVA with interactions + permutation significance
# --------------------------------------------------------------------------- #
def _ss(sub, factors, grand):
    """Sum of squares explained by the joint grouping over `factors`
    (balanced design => this is the orthogonal main/joint effect SS)."""
    if not factors:
        return 0.0
    ss = 0.0
    for _, g in sub.groupby(list(factors)):
        v = g["value"].to_numpy()
        ss += len(v) * (v.mean() - grand) ** 2
    return float(ss)


def anova_decomposition(df, target_model, metric, k, factors,
                        interactions=True, n_perm=2000, seed=0):
    """N-way variance decomposition for one model's score.

    Returns main-effect eta^2 (with a permutation p-value) and, optionally,
    all pairwise interaction eta^2. Design is the balanced Cartesian product,
    so main effects are orthogonal; interactions are computed by subtraction.
    """
    sub = df[(df["metric"] == metric) & (df["k"] == k) &
             (df["model"] == target_model)].copy()
    y = sub["value"].to_numpy()
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total == 0:
        return pd.DataFrame(columns=["term", "type", "eta_squared", "p_perm"])
    rng = np.random.default_rng(seed)
    rows = []
    # main effects + permutation test
    for f in factors:
        obs = _ss(sub, [f], grand) / ss_total
        if n_perm and sub[f].nunique() > 1:
            perm = sub.copy(); cnt = 0
            vals = perm[f].to_numpy().copy()
            for _ in range(n_perm):
                rng.shuffle(vals); perm[f] = vals
                if _ss(perm, [f], grand) / ss_total >= obs:
                    cnt += 1
            p = (cnt + 1) / (n_perm + 1)
        else:
            p = np.nan
        rows.append({"term": f, "type": "main",
                     "eta_squared": round(obs, 4), "p_perm": round(p, 4)})
    # pairwise interactions
    if interactions:
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                a, b = factors[i], factors[j]
                inter = (_ss(sub, [a, b], grand) - _ss(sub, [a], grand)
                         - _ss(sub, [b], grand)) / ss_total
                rows.append({"term": f"{a}×{b}", "type": "interaction",
                             "eta_squared": round(max(inter, 0.0), 4), "p_perm": np.nan})
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)


def spec_curve_test(df, model_a, model_b, metric, k, n_perm=5000, seed=0):
    """Specification-curve inference (Simonsohn-style permutation test).

    H0: models A and B are exchangeable, so the sign of each per-spec gap is
    random. Tests whether the observed curve (median gap, share favoring A) is
    more extreme than under H0.
    """
    piv = _pivot(df, metric, k)[[model_a, model_b]].dropna()
    gap = (piv[model_a] - piv[model_b]).to_numpy()
    obs_median = float(np.median(gap))
    obs_share = float((gap > 0).mean())
    rng = np.random.default_rng(seed)
    n = len(gap)
    med_ge = sh_ge = 0
    for _ in range(n_perm):
        flip = rng.choice([1, -1], size=n)
        g = gap * flip
        if abs(np.median(g)) >= abs(obs_median):
            med_ge += 1
        if abs((g > 0).mean() - 0.5) >= abs(obs_share - 0.5):
            sh_ge += 1
    return {"model_a": model_a, "model_b": model_b, "n_specs": n,
            "median_gap": round(obs_median, 4),
            "share_favoring_a": round(obs_share, 3),
            "p_median": round((med_ge + 1) / (n_perm + 1), 4),
            "p_share": round((sh_ge + 1) / (n_perm + 1), 4)}


# --------------------------------------------------------------------------- #
# (1) Reporting protocol: robustness verdicts + a generated report
# --------------------------------------------------------------------------- #
def _verdict(frag):
    if frag < 0.1:
        return "robust"
    if frag < 0.35:
        return "fragile"
    return "coin-flip"


def robustness_report(df, metric, k):
    """Per-pair fragility + a plain-language verdict, plus a single leaderboard
    robustness score = 1 - mean(fragility) (1.0 = every ordering is stable)."""
    frag = fragility_index(df, metric, k)
    frag["verdict"] = frag["fragility"].apply(_verdict)
    score = round(1.0 - frag["fragility"].mean(), 3)
    return frag, score


def _md(table):
    try:
        return table.to_markdown(index=False)
    except Exception:
        return "```\n" + table.to_string(index=False) + "\n```"


def generate_report(df, metric="ndcg", k=10, factors=None, path="multiverse_report.md"):
    """Write a standardized multiverse-evaluation report: the proposed reporting
    protocol output (distribution + fragility verdicts + variance attribution +
    a robustness score) that should accompany any single-number leaderboard claim."""
    datasets = df["dataset"].unique() if "dataset" in df.columns else [None]
    lines = ["# Multiverse evaluation report", "",
             f"Metric: **{metric}@{k}** · specifications per dataset: "
             f"**{df['spec'].nunique()}**", ""]
    for ds in datasets:
        d = df if ds is None else df[df["dataset"] == ds]
        if d.empty or d[(d["metric"] == metric) & (d["k"] == k)].empty:
            lines.append(f"## {ds or 'dataset'}\n\n_No usable specifications (data too sparse after filtering)._\n")
            continue
        lines.append(f"## {ds or 'dataset'}")
        ci = summary_with_ci(d, metric, k)
        lines.append("\n**Performance across the multiverse (mean ± 95% CI):**\n")
        lines.append(_md(ci))
        stab = ranking_stability(d, metric, k)
        frag, score = robustness_report(d, metric, k)
        lines.append(f"\n**Leaderboard robustness score: {score}** "
                     f"(1.0 = orderings never reverse). "
                     f"Top-1 stability {stab['stability']} with "
                     f"{stab['distinct_winners']} distinct winners across pipelines.\n")
        lines.append("**Pairwise robustness verdicts:**\n")
        lines.append(_md(frag[["model_a", "model_b", "majority_winner",
                               "fragility", "verdict"]]))
        if factors:
            lines.append("\n**What drives the score (variance decomposition):**\n")
            for m in d["model"].unique():
                vd = variance_decomposition(d, m, metric, k, factors)
                top = ", ".join(f"{r.factor}={r.eta_squared}" for r in vd.itertuples())
                lines.append(f"- {m}: {top}")
            # strongest pairwise interaction per trainable model
            lines.append("\n**Top variance interactions (ANOVA):**\n")
            for m in d["model"].unique():
                av = anova_decomposition(d, m, metric, k, factors,
                                         interactions=True, n_perm=0)
                inter = av[av["type"] == "interaction"].head(1)
                if len(inter):
                    r = inter.iloc[0]
                    lines.append(f"- {m}: {r['term']} (η²={r['eta_squared']})")
        # metric robustness
        mr = metric_robustness(d, k)
        agree = "agree" if mr.attrs.get("winner_agrees") else "DISAGREE"
        lines.append(f"\n**Metric-family robustness:** modal winners {agree} "
                     f"across {', '.join(mr['metric'])}.\n")
        lines.append(_md(mr))
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
