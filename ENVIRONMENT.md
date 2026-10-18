# Environment

Two environments matter for this package: the **training environment** that
produced `results/results_all_full.csv`, and the **analysis environment** used
to run the lightweight analysis/verification/figure code.

## Training environment (produced the results)

The full multiverse (all deep models) was run on **Google Colab** with an
**NVIDIA A100-SXM4-80GB** GPU and **PyTorch 2.11.0 (CUDA 13.0 build, cu130)**.
Deep-model results (BPRMF, LightGCN, SASRec) depend on the exact PyTorch/CUDA
build and on GPU floating-point nondeterminism. To record the full stack on the
machine that runs the notebooks, save the output of:

```bash
python -c "import sys,torch,numpy,pandas,scipy; \
print('python', sys.version.split()[0]); \
print('torch', torch.__version__, '| cuda', torch.version.cuda, \
'| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); \
print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__)"
pip freeze > pip-freeze-training.txt
```

| Component | Value |
|-----------|-------|
| Platform  | Google Colab |
| GPU       | NVIDIA A100-SXM4-80GB |
| PyTorch   | 2.11.0+cu130 |
| Python    | Colab default 3.x (record with the command above) |

Wall-clock time of the main run (`run_multiverse.ipynb`, tuning included):

| Dataset | Time |
|---------|------|
| MovieLens-100K | 1,681 s (≈ 0.5 h) |
| MovieLens-1M | 20,116 s (≈ 5.6 h) |
| Gowalla | 3,405 s (≈ 0.9 h) |
| Amazon CDs & Vinyl | 1,045 s (≈ 0.3 h) |
| Yelp | 2,354 s (≈ 0.7 h) |
| **Total** | **≈ 7.9 h** |

Wall-clock time of the appendix run (`appendix_robustness.ipynb`):

| Experiment | Time |
|------------|------|
| Gowalla, 5,000 users | 1,565 s (≈ 0.4 h) |
| Gowalla, 10,000 users | 3,387 s (≈ 0.9 h) |
| Gowalla, 20,000 users | 10,491 s (≈ 2.9 h) |
| MovieLens-1M, alternative tuning | 19,143 s (≈ 5.3 h) |
| **Total** | **≈ 9.6 h** |

> The classical models (MostPop, ItemKNN, EASE) are deterministic and
> reproduce exactly on any platform; only the three torch models carry
> GPU/version-dependent variation, which stays below the paper's reported
> precision.

## Analysis environment (verification & figures — reference)

The analysis, verification and figure code (`multiverse/analysis.py`,
`verify.py`, `figures.py`; `scripts/verify.py`, `make_figures.py`) is pure
NumPy/pandas/Matplotlib and needs no GPU. It was last run successfully with:

| Package     | Version (reference) |
|-------------|---------------------|
| Python      | 3.13 |
| numpy       | 2.5.3 |
| pandas      | 3.0.5 |
| scipy       | 1.18.1 |
| matplotlib  | 3.11.2 |
| torch       | 2.14.1 (cu130) |

The minimum versions in `requirements.txt` are intentionally permissive; the
table above is a known-good reference set. Pin to it if you need to match the
analysis environment closely.

## Hardware notes / runtime

- Large catalogues are capped at `MAX_USERS = 10000` sampled users (held fixed
  by seed) to keep Amazon/Gowalla/Yelp tractable.
- A full paper-grade run (`QUICK = False`, all five datasets, all models, with
  per-model validation tuning) takes on the order of hours on an A100; Gowalla
  and the SASRec runs dominate the cost.
- The CPU-only synthetic smoke test (`python scripts/reproduce.py --dataset
  synthetic --quick`) completes in about a minute and needs no downloads.

## Data integrity

`SHA256SUMS.txt` lists checksums for the shipped results and figures. Verify with:

```bash
sha256sum -c SHA256SUMS.txt
```
