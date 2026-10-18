# Testing & verification coverage

This document states precisely what the shipped checks do and do not cover, so
the scope is not overstated.

## `scripts/verify.py` — 46 selected manuscript-consistency checks

Runs `multiverse.verify.run_checks` against `results/results_all_full.csv` and
prints PASS/FAIL per check (exit non-zero on any mismatch). Last run: **46/46
PASS**. The 46 checks cover:

| Group | What is checked | Against |
|-------|-----------------|---------|
| 1 | Grid sizes per dataset (72 / 36 specs), Amazon k-core = {5} | Methods; Results |
| 2 | Fragility-definition invariance: R(all) identical under `min(p,q)` and `min(p,1-p)` at 2 dp | Methods (fragility index) |
| 3 | Candidate-sampling η² gradient (personalised-avg); split η² for ML-100K/1M | Fig. 3 |
| 4 | SASRec mean NDCG@10 per dataset | Table 1 |
| 5 | Largest pairwise interaction η² (personalised-avg); Yelp term = candidate×k-core | Supplementary Table S2 |
| 6 | R(top-3) and R(all) per dataset | Table 2 |
| 7 | Yelp R(top-n) subsample row; last column == R(all) | Supplementary Table S3 |

Each expected value is the number printed in the manuscript.
Values are compared at the precision printed in the manuscript (absolute tolerance 0.0015 for
η²); for example, the Amazon interaction η² is computed as 0.025, as printed, against a stored
expectation of 0.026 from an earlier run, and passes within that tolerance.

### What `verify.py` does NOT do

- It does **not** retrain any model. It checks the *saved* results, not that a
  fresh training run reproduces them.
- It does **not** independently re-derive every table cell, every confidence
  interval, every permutation p-value, or every value inside the figures. It
  checks the selected headline quantities listed above.
- It does **not** check data preparation (that is `check_data.py`).

## `scripts/check_data.py` — data preparation vs Supplementary Table S4

Loads each dataset through the paper's loaders, applies the `MAX_USERS = 10000`
subsample, and checks the resulting user / item / interaction counts against
Supplementary Table S4. Requires network access (GroupLens, SNAP) and, for Yelp, the Yelp Open
Dataset file. Not run in the offline package build.

## `scripts/make_figures.py` — figures

Regenerates Figures 2, 3 and 4 from `results/results_all_full.csv` via
`multiverse.figures`. The committed PNG/PDF files were produced by this code;
Figure 3's candidate gradient reproduces as 0.48 / 0.64 / 0.87 / 0.90 / 0.99.

## End-to-end reproduction (full scale)

The main pipeline (`run_multiverse.ipynb`) was run from scratch at full scale
with the current code **twice**, independently (Colab, NVIDIA A100,
`QUICK = False`; ≈ 7.9 h each, one uninterrupted pass, execution counts 1–22).
The two runs agree to the fourth decimal: across all 29 dataset × model mean
NDCG@10 values the largest difference is 0.0001, and the validation tuning
selected identical hyperparameters in both. The shipped notebook, `results_all_full.csv`
and `tuned_configs_full.json` are from the second run; `scripts/verify.py`
passes 46/46 on it.

The appendix notebook (`appendix_robustness.ipynb`) was then run at full scale
on the second run's results (≈ 9.6 h, execution counts 1–20). It is shipped
executed; its per-run results are in `results/appendix/`.

## Notebook execution test

Both notebooks were also executed cell by cell, in order, in an automated
harness that runs the notebooks' own code cells (the `%%writefile` cells write
the package; only the Google Drive mount and shell `!` lines are stubbed), with
synthetic data and tiny training budgets injected after the configuration cell:

- `run_multiverse.ipynb`: every code cell executes — package build, loaders,
  configuration, the full multiverse over five datasets × six models, the
  headline analysis, the per-dataset diagnostic plots, permutation tests and the
  report.
- `appendix_robustness.ipynb`: every code cell executes — setup, loading the
  main run's results, (A) the subsample-size experiment (including the EASE
  memory-exclusion branch) and (B) the alternative-tuning experiment, writing
  all of its result files.

This checks that the notebooks run end to end without errors; it does not
replace the full-scale runs, which need a GPU.

The CPU synthetic smoke test of the CLI (`scripts/reproduce.py --dataset
synthetic --quick`) also runs and passes.
