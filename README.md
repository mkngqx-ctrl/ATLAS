# ATLAS: Autoregressive Trajectories for Longitudinal Analysis of Survival

This repository reproduces the November 18, 2026 Statistics in Medicine manuscript. The two SFSA experiments use synthetic patient trajectories. **Start with the CPU workflow** to recover test errors directly from the saved survival and cumulative incidence predictions. All paths used by the entry points are relative to this repository.

## Abstract

Dynamic survival prediction requires estimates that reflect a patient's observed clinical history. We introduce ATLAS (Autoregressive Trajectories for Longitudinal Analysis of Survival), a framework that trains a sequence model on longitudinal data and autoregressively samples possible futures conditional on an individual's covariates and observed history. These trajectories form a patient-specific synthetic cohort from which Kaplan–Meier survival curves and Aalen–Johansen cumulative incidence curves are estimated. We evaluate ATLAS using two stochastic finite-state automaton benchmarks with exact conditional reference curves: a single-covariate process with duration dependence and a late rule change, and a three-covariate Markov process with recovery and competing causes of death. Comparisons include eight direct survival models, including validation-optimized DRSA and Dynamic-DeepHit and history-aware SurvTRACE adaptations, with each neural model trained nine times in the first experiment and five times in the second. After baseline, every ATLAS configuration had lower mean survival-curve error than every fitted direct comparator in both experiments. With a common Sobol sampler and 10,000 trajectories per patient, the best ATLAS model did so in every individual training run, and its mean error was 62% to 84% lower than that of the strongest direct comparator. Baseline advantages were smaller and less consistent in the duration-dependent experiment. ATLAS provides a generative framework for history-conditioned survival estimation, while the accompanying benchmarks enable direct assessment of individual survival and competing-risk predictions against known reference curves.

## Reproduce the results on CPU

Clone or extract the whole repository, open a terminal in the folder containing `reproduce.py`, and create a Python 3.10 environment. These commands use the interpreter selected in your terminal. On Windows activate with `.venv\Scripts\activate`. On macOS or Linux use `source .venv/bin/activate`.

```sh
python -m venv .venv
# Activate the environment as described above.
python -m pip install -r requirements.txt
python reproduce.py all
```

The tested scientific stack is Python 3.10.20, NumPy 2.2.6, SciPy 1.15.2 and Matplotlib 3.10.8. CPU scoring, bootstrap and figure regeneration do not require a GPU. Outputs are written under `outputs/`. Individual commands are restartable.

| Command | Inputs and expected outputs |
|---|---|
| `python reproduce.py scores` | Reads the original prediction arrays for all 14 training runs, substitutes the final optimized DDH/DRSA predictions, confirms test-split membership, and reconstructs per-run and mean survival/CIF MAEs. Checks 2,079 per-run rows against archived MAE, RMSE and, where reported, post-24 MAE at absolute tolerance 1e-7. The summary also includes between-run standard deviations and ranges for MAE. Outputs `per_run_MAE.csv` and `mean_MAE.csv`. |
| `python reproduce.py bootstrap` | Uses saved patient-level errors and the 5,000 paired patient draws per experiment to recalculate the 240 contrasts and their pointwise intervals. Outputs `bootstrap_verification.json`. |
| `python reproduce.py figures` | Recreates the five main figures and supplementary comparison, resource and bootstrap plots from the archived arrays or plot values. Also recomputes the 81 original DDH MAEs for S10. See `outputs/figures/` and `outputs/original_DDH_MAE.csv`. Exact typeset figures and table sources are in `paper/`. |
| `python reproduce.py prepare` | Extracts the two seed-42 archives into `outputs/runs/{Hard24,CR3}/42/`, including cohorts, patient splits, reference curves, checkpoints, training settings and landmark-specific test patient IDs. |
| `python reproduce.py s9` | After installing training requirements, verifies the 140 S9 IID/Sobol sampled prediction files against saved risk/event counts and recalculates patient MAEs. |

Each stochastic run contributes its mean error across five sampling seeds. Training runs receive equal weight. In `mean_MAE.csv`, select `direct` for fitted direct comparators, `sobol` for the main ATLAS comparisons and `iid` for the sampled oracle. Hard24 uses observed-process Kaplan–Meier reference curves. CR3 uses survival and Death A/B cumulative incidence references. Both baseline evaluations use 300 test patients. Later landmarks use the eligible subsets. The generator, censoring, landmark and evaluation definitions are in the manuscript and reference implementations.

## Retraining and checkpoint studies

For training, create and activate a separate Python 3.11 environment, then install `requirements-training.txt`. The archived lifelines version requires Python 3.11 or later. For GPU training install a PyTorch build compatible with the host's GPU driver. The DRSA optimizer requires CUDA. Preserve the source bytes in the training bundles since the original training scripts verify their hashes. The archived DDH and DRSA training environments are recorded beside their results. Training times and fitted weights can differ across hardware and PyTorch builds.

```sh
python -m pip install -r requirements-training.txt
python reproduce.py train-base --experiment Hard24 --seed 43 --device cuda
python reproduce.py train-base --experiment CR3 --seed 43 --device cuda
python reproduce.py train-ddh --device cuda
python reproduce.py train-drsa
python reproduce.py resource --device cuda
```

For the original base comparisons, repeat `train-base` for Hard24 seeds **42, 43, 44, 45, 50, 51, 52, 62, 63** and CR3 seeds **42, 43, 45, 46, 47**. Seed 42 has its own workflow. The Hard24 runner for seeds 50–52 includes the original Windows DDH backend adjustment. Training and evaluation run in separate experiment-specific processes. The base workflows include the original comparators. Use the separate `train-ddh` and `train-drsa` commands for the optimized comparisons in the main paper. They run the validation searches and final fits for both experiments. Each writes to its own output directory. The CPU scoring command reads the frozen published predictions. To assess a new fit, use the evaluation outputs written by its training workflow.

## Where to find the inputs and evidence

- `runs/{Hard24,CR3}/seed_*.zip` contains one original run per archive. In each, `checkpoints/cohort.npz` includes the cohort and train/validation/test indices. Under `evaluation/L*/reference.npz` are the patient IDs, times and reference curves. Prediction arrays come from the same archive and landmark. Historical labels beginning with Glazier correspond to ATLAS.
- `optimized_ddh/` contains the returned FAST study, selection locks, final checkpoints and predictions. `training/ddh/` contains the matching model, engine, train/validation input arrays and sealed evaluation payload.
- `archive/supplementary/optimized_DRSA_integrated_2026-11-16/` contains the final all-model comparison CSVs, DRSA checkpoints and predictions, patient-bootstrap arrays and plotting data. Its summary CSV is the frozen comparison target for the current paper. `training/drsa/` contains the portable optimizer and `protocols/drsa/` the recorded decisions.
- `archive/supplementary/original_DDH_results_2026-11-02/` provides the original DDH data for S10. Earlier DDH-only integration files record an intermediate comparison.
- `archive/classical_comparators/` contains the S9 fitted transition models, direct and sampled predictions, fit scripts and tests. `supplement_S9/` contains scoring and comparison tables. Run `prepare` before using the fit scripts. These research scripts write beside their inputs, so use a working copy for refitting. For example run `python archive/classical_comparators/fit_semimarkov.py` for the Hard24 history-aware fit.
- `resource_study/` holds the separate original Weibull/ReLU study, checkpoints, notebook and recorded full results ZIP. Its inputs differ from the Hard24 main experiment.
- `archive/reference_code/` has the benchmark reference calculations. `paper/` has the manuscript and supplementary sources, figures and tables. See `paper/README.md` for LaTeX build instructions. Historical scripts and run records under `archive/` and `provenance/` may record original machine paths. Use the top-level commands for the portable workflow.
- `MANIFEST.sha256` lists the repository file checksums. The prepublication manuscript contains a pending data-availability citation.

## Verification scope

The published-result checks are deterministic reanalyses of archived model outputs. Bootstrap reproduction begins with the archived patient-error arrays. Training code and exact cohorts are included for fresh fits. Full GPU refitting and cross-platform execution were not repeated when assembling this repository.
