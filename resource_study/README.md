# Semi-Markov sampling-budget notebook

Open `Glazier_SemiMarkov_Sampling_Budget.ipynb` and run its cells in order. In Colab, select a GPU and upload this ZIP when requested. Locally, extract the ZIP, open the notebook inside the extracted directory, and select your GPU environment. The notebook includes setup instructions and an optional Google Drive output location.

The bundle includes both previously trained local full-run semi-Markov cohorts and all eight checkpoints per scenario. No retraining is performed. The diagnostic compares the three Glazier variants, a matched-budget true-generator sampler, DynamicDeepHit and SurvTransformer. You can replace the `sm_Weibull` and `sm_ReLU` folders with compatible folders saved by the extension workflow if you want to evaluate your own notebook checkpoints instead. Use the default preserved generator; custom generator changes also require changing the deterministic reference.

Default full diagnostic: 500 / 2,000 / 10,000 trajectories, five sampling seeds, 300 fixed test patients, Weibull and ReLU, baseline and landmarks 10 / 15 / 20. `MODE="smoke"` runs a small execution check. Changing settings or code requires a new output folder. Matching saved replicates resume automatically. Full mode has been coded but has not been run as part of this notebook-delivery task.

References use the deterministic finite-state population KM calculation, avoiding reference Monte Carlo noise. They preserve the original observed-generator censoring convention. The diagnostic saves per-replicate curves, reference curves, patient IDs, metrics, plots, source/checkpoint hashes, and a downloadable results archive.

Error bars show sampling-seed standard deviations, not confidence intervals. The MSE decomposition is a finite-replicate identity; ensemble error is not pure model bias. The direct landmark models do not receive the observed prefix, so landmark architecture comparisons remain unmatched in input information.

Validation: all notebook code cells executed in local GPU smoke mode across both scenarios and all included models. All three Glazier models also passed 10,000-trajectory prediction checks at baseline and landmark 10. Deterministic references were compared with 50,000-trajectory simulations for two eligible patients per scenario at baseline and landmark 10 (maximum absolute differences approximately 0.0051). The finite-replicate MSE identity passed an analytic check. Validation records are included; smoke runs establish execution, not scientific conclusions.

For the original local gpu_env, the notebook can use support packages in the workspace's `work/gpu_support` folder. Elsewhere, install NumPy, pandas, matplotlib, PyTorch and lifelines 0.30.3 in the selected environment. Colab setup installs the additional analysis dependencies without replacing its PyTorch installation. Use a fresh kernel so the MKL threading setting takes effect before numerical imports.
