# recsys-multiverse

Reproducibility package for:

> **A Multiverse Analysis of Offline Recommender Systems Evaluation: Quantifying the Fragility of Model Comparisons**
> Resul Berkem Aydemir and Burcu Demirelli Okkalioglu, 2026.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23453297.svg)](https://doi.org/10.5281/zenodo.23453297)

Archived release (v1.0): https://doi.org/10.5281/zenodo.23453297

The paper asks a simple question: when we rank recommender models on an offline
benchmark, **how much of the leaderboard is an artifact of defensible-but-arbitrary
evaluation choices?** We answer it by running the *same* six models through a
*multiverse* of pipelines — every combination of six evaluation/training degrees
of freedom — and measuring how often the conclusions reverse.

This repository contains the full engine, the driver notebooks that produced the
paper's results, and a standalone script to reproduce them from scratch.

---

## What's inside

```
rs-multiverse/
├── multiverse/              # the engine (pure Python, importable package)
│   ├── config.py            #   SpecSpace: the 6 degrees of freedom
│   ├── data.py              #   loaders, k-core filtering, 3 split strategies
│   ├── models.py            #   MostPop, ItemKNN, EASE, BPRMF, LightGCN, SASRec
│   ├── metrics.py           #   recall / ndcg / map / mrr / hit @ k
│   ├── runner.py            #   run_multiverse(): trains × evaluates every spec
│   ├── analysis.py          #   eta^2 / ANOVA, fragility index, robustness score
│   ├── tuning.py            #   validation-based per-model grid search
│   ├── figures.py           #   regenerate Figures 2, 3 & 4 from a results frame
│   └── verify.py            #   46 selected consistency checks against saved results
├── notebooks/
│   ├── run_multiverse.ipynb       # main pipeline: package, data, full multiverse, analysis, Figs 2-4
│   └── appendix_robustness.ipynb  # appendix: subsample-size + alternative-tuning checks
├── scripts/
│   ├── reproduce.py          # CLI runner — no Colab / Google Drive needed
│   ├── make_figures.py       # rebuild Figures 2-4 from results/results_all_full.csv
│   ├── verify.py             # 46 selected manuscript-consistency checks (exits non-zero on mismatch)
│   └── check_data.py         # check loaded dataset counts against Supplementary Table S4 (needs network)
├── results/                  # results CSV, tuned configs, appendix/ (robustness-check results)
├── figures/                  # Figure 1 (PNG) and Figures 2, 3 & 4 (PNG + PDF), as in the paper
├── requirements.txt
├── CITATION.cff
└── LICENSE                   # MIT
```

## The six degrees of freedom

Each *specification* is one defensible way to run an offline evaluation. The
Cartesian product of these axes is the multiverse (`multiverse/config.py`):

| DoF            | Values                                                   |
|----------------|----------------------------------------------------------|
| `kcore`        | 5, 10 (iterative k-core density filtering threshold)     |
| `split`        | `random_holdout`, `temporal_global`, `leave_last_out`    |
| `candidates`   | `full` (full ranking), `sampled` (sampled negatives)     |
| `n_negatives`  | 100 (only when `candidates == sampled`)                  |
| `train_neg`    | `uniform`, `popularity` (training-time negative sampling) |
| `seed`         | 0, 1, 2                                                  |

Metric family (`recall`, `ndcg`, `map`, `mrr`) and cutoff *k* (10, 20) are scored
on every trained model at once, so they are treated as *analysis* axes rather
than training specifications. This yields **72 specifications per dataset** (36
for models that ignore `train_neg`).

## The six models

`MostPop`, `ItemKNN`, `EASE` (Steck, WWW 2019), `BPRMF` (Rendle et al., UAI 2009),
`LightGCN` (He et al., SIGIR 2020), and `SASRec` (Kang & McAuley, ICDM 2018). All
trainable models are tuned under an identical budget (`multiverse/tuning.py`) so
that observed differences cannot be attributed to unequal tuning effort.

> **Note on SASRec.** The `SASRec` class ships in its strengthened, paper-grade
> configuration (2 attention heads, batch 128, up to 100 epochs with early
> stopping). The small batch is deliberate — see "SASRec correction" below.

> **Note on EASE.** EASE builds a dense *n_items × n_items* Gram matrix, which is
> infeasible on very large catalogues (e.g. Gowalla); the driver skips EASE there,
> which is why EASE is absent from the Gowalla rows of the results table.

## Datasets

The loaders (`scripts/reproduce.py`, and `notebooks/run_multiverse.ipynb`) pull
public data directly from the original sources:

| Key          | Dataset                 | Source                                                            |
|--------------|-------------------------|-------------------------------------------------------------------|
| `ml-100k`    | MovieLens 100K          | https://files.grouplens.org/datasets/movielens/ml-100k.zip        |
| `ml-1m`      | MovieLens 1M            | https://files.grouplens.org/datasets/movielens/ml-1m.zip          |
| `gowalla`    | Gowalla check-ins       | https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz   |
| `amazon-cds` | Amazon CDs & Vinyl      | SNAP Amazon product ratings (CDs_and_Vinyl)                       |
| `yelp`       | Yelp reviews            | Yelp Open Dataset (`yelp_academic_dataset_review.json`)           |

MovieLens, Gowalla and Amazon download automatically. The **Yelp** dump must be
obtained from the Yelp Open Dataset (license requires manual acceptance) and
placed next to the notebook as `yelp_academic_dataset_review.json`. Large logs
(Amazon, Gowalla, Yelp) are capped at `MAX_USERS = 10000` sampled users, held
fixed by seed, matching the counts in Supplementary Table S4 of the paper.

## Reproduce

### Option A — standalone script (no Colab, no Drive)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# a fast end-to-end check on a synthetic log (~1 min, needs no downloads):
python scripts/reproduce.py --dataset synthetic --quick

# one real dataset, paper settings:
python scripts/reproduce.py --dataset ml-100k --out results/
```

`reproduce.py` writes `results/results_<dataset>.csv` and a
`results/multiverse_report.md`, and prints the leaderboard-robustness score and
the most fragile model pairs.

### Option B — notebooks (as used for the paper)

Two notebooks, each run top to bottom on its own (Colab, GPU):

Both notebooks are shipped **executed**, with the outputs of the full-scale
runs (Colab, NVIDIA A100) that produced the files in `results/`.

1. **`notebooks/run_multiverse.ipynb` — main pipeline.** With `QUICK = False`
   it builds the package, loads the five datasets, tunes every model, runs the
   full paper grid (all six models), writes `results_all_full.csv` and
   `tuned_configs_full.json`, and produces the headline analysis, Figures 2–4 of
   the paper, permutation tests and the report. About 8 hours on an A100 (see
   `ENVIRONMENT.md`).
2. **`notebooks/appendix_robustness.ipynb` — appendix checks.** Run it after the
   main notebook, in the same folder. It re-runs the multiverse (A) on Gowalla
   at 5k / 10k / 20k users and (B) on MovieLens-1M with an alternative tuning
   protocol, and compares against the main run. About 9.6 hours on an A100; its
   per-run results are in `results/appendix/`.

The embedded `%%writefile` cells in both notebooks write exactly the
`multiverse/` engine this repo ships, so the notebooks and the scripts run the
same code. The results are in `results/` and are checked by `scripts/verify.py`.

### Verify the paper's numbers

```bash
python scripts/verify.py          # 46 selected consistency checks on the saved results
```

This runs **46 selected manuscript-consistency checks**: it recomputes a chosen
set of headline numbers — grid sizes, the candidate-sampling η² gradient, SASRec
means, the largest interaction η² (Supplementary Table S2), R(top-3)/R(all)
(Table 2), and the Yelp R(top-n) profile (Supplementary Table S3) — from `results/results_all_full.csv`, prints PASS/FAIL per check,
and exits non-zero on any mismatch. The check code is `multiverse/verify.py`;
each expected value is the number printed in the manuscript.

**Scope (what this does and does not do).** These checks confirm that the
*saved* results reproduce the paper's reported numbers; they do **not** retrain
the models, and they do not independently re-derive every table cell, confidence
interval, permutation p-value, or figure value. To check the data-preparation
stage against Supplementary Table S4, run `python scripts/check_data.py` (needs network). To
re-derive the results from scratch, use Option A/B above. See `TESTING.md` for
the exact coverage.

> **Numbering in code comments and printed output.** Code comments in `multiverse/*.py` and the
> labels printed by the notebooks and `scripts/verify.py` use the table/section numbers of an earlier
> draft of the manuscript; the code itself is frozen so that it stays identical to the executed
> notebooks. The correspondence with the published numbering is:
> "Table 3" → Supplementary Table S4 (dataset statistics); "Table 4" → Supplementary Table S5
> (selected hyperparameters); "Table 5" (largest interaction) → Supplementary Table S2;
> "Table 6" (Yelp R(top-n) row) → Supplementary Table S3; "Table 7" → Table 2;
> "§3.x" → Methods. Figure numbers are unchanged.

### Figures

```bash
python scripts/make_figures.py          # rebuilds Figures 2-4 into figures/
```

The figure code lives in the package (`multiverse/figures.py`), so the committed
`figures/figure{2,3,4}_*.{png,pdf}` are reproduced exactly by the line above from
`results/results_all_full.csv`. Figure 4 derives the most-fragile top-3 pair per
dataset from the data itself (no hard-coded pairs).

> **Single source of truth.** The engine is `multiverse/*.py`. SASRec is a normal
> first-class model there (not a patch applied elsewhere); `scripts/reproduce.py`,
> `scripts/make_figures.py` and both notebooks use exactly this code.

### SASRec correction (provenance)

An earlier run used a weak SASRec configuration (1 attention head, 20 epochs),
which understated its accuracy. The model was strengthened to **2 heads, batch
128, up to 100 epochs with early stopping**, tuned over lr∈{1e-3,5e-4},
maxlen∈{50,200}, dropout∈{0.2,0.5}. This is the default `SASRec` in
`multiverse/models.py` and the config used in `scripts/reproduce.py` and
`run_multiverse.ipynb`; it raises SASRec NDCG@10 on MovieLens-100K from 0.118 to
0.269. The small batch is essential: on small catalogues a large batch gives too
few updates per epoch and under-trains SASRec back to the weak ~0.12 level.

## How outputs map to the paper

| Paper artifact                              | Produced by                                            |
|---------------------------------------------|--------------------------------------------------------|
| Variance decomposition (η²), per model      | `analysis.variance_decomposition` / `anova_decomposition` |
| Fragility index φ = min(p, q)               | `analysis.fragility_index`                             |
| Leaderboard robustness score R = 1 − mean φ | `analysis.robustness_report`                           |
| Permutation significance                    | `analysis.anova_decomposition`, `analysis.spec_curve_test` |
| Metric-family agreement                     | `analysis.metric_robustness`                           |
| Figure 2 (per-model variance shares)        | `figures.figure_variance` · `scripts/make_figures.py` |
| Figure 3 (candidate-sampling gradient)      | `figures.figure_gradient` · `scripts/make_figures.py` |
| Figure 4 (spec curves, most fragile top-3 pair) | `figures.figure_speccurves` · `scripts/make_figures.py` |
| Standardised multiverse report              | `analysis.generate_report`                             |
| Consistency checks (46 selected)            | `multiverse.verify` · `scripts/verify.py`              |
| Data-prep check vs Supplementary Table S4    | `scripts/check_data.py`                                |

## Data & results availability

`results/results_all_full.csv` (10,080 rows: 5 datasets × 6 models × specs ×
metrics × cutoffs) is the authoritative output the paper's reported numbers are
checked against, and it is **included in this package**. It carries the
strengthened-SASRec scores (SASRec NDCG@10 on MovieLens-100K ≈ 0.269, matching
the paper); run `python scripts/verify.py` against it for the 46 consistency
checks without re-running the grid. Options A/B above regenerate it from scratch.
`SHA256SUMS.txt` records the checksums of the shipped data files.

`results_all_full.csv` was produced by a full from-scratch run of the current
code (`run_multiverse.ipynb`, Colab A100), and `scripts/verify.py` passes all 46
checks on it.

`results/tuned_configs_full.json` holds the per-dataset hyperparameters selected
by validation tuning in that same run (the source of Supplementary Table S5), so the exact
configuration can be reused instead of re-tuning.

## Citation

Please cite both the paper and this software. Software:

> Aydemir, R. B. & Demirelli Okkalioglu, B. recsys-multiverse: a multiverse / specification-curve analysis
> harness for offline recommender-system evaluation (v1.0). Zenodo https://doi.org/10.5281/zenodo.23453297 (2026).

Machine-readable metadata are in `CITATION.cff`.

## License

MIT — see `LICENSE`.
