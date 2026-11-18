CURRENT MANUSCRIPT RESULTS — updated 2026-11-16

Main comparisons now use both DRSA (optimized) and Dynamic-DeepHit (optimized).
Use supplementary/optimized_DRSA_integrated_2026-11-16/ for current integrated
all-model summary/per-training-run metrics, updated bootstrap results and figure
data. In those machine-readable comparison files the internal model key DRSA
means DRSA (optimized), and DynamicDeepHit means Dynamic-DeepHit (optimized).
Earlier result folders/manifests are archived source records and may contain
superseded DRSA rows; they do not define the current manuscript comparisons.

CURRENT DDH RESULTS (updated 2026-11-02)

The main comparisons use Dynamic-DeepHit (optimized). The current integrated
results are in supplementary/optimized_DDH_integrated_2026-11-02/:
- updated_all_models_summary.csv: aggregate comparisons with optimized DDH.
- updated_all_models_per_training_run.csv: per-training-run comparisons.
- optimized_DDH_per_seed_metrics.csv: optimized DDH target/seed metrics.
- strongest_direct_by_setting.csv and strongest_direct_by_seed.csv: updated
  strongest-direct-comparator identities.
- paired_training_run_sign_tests.csv: updated paired training-run statistics.
- figure_assets/: updated comparison figures, plot data and generation materials.
- bootstrap/: updated 5,000-resample paired patient bootstrap analysis and figures
  used in Supplementary Section S8.

Selected-fit epochs are recorded in supplementary/optimized_DDH_selected_epochs.csv.
Original DDH results are retained in Supplementary Section S10 and archived in
supplementary/original_DDH_results_2026-11-02/, including original aggregate and
per-seed metrics, predictions/checkpoints, bootstrap comparisons and figure assets.
Earlier multi-seed tables, predictions and manifests may describe the original
DDH adaptation. For current DDH comparisons, use the integrated results above;
the earlier MULTISEED_MANIFEST.json does not index this optimized DDH update.
Original timing summaries describe the original DDH adaptation, not the optimized fits.

ATLAS MULTI-SEED SUPPLEMENT

Primary training seeds:
Hard24: 42,43,44,45,50,51,52,62,63 (nine neural fits per model).
CR3: 42,43,45,46,47 (five neural fits per model).
Rollout seeds: 41001,52001,63001,74001,85001; M=10000.

For stochastic methods, average MAE within each trained checkpoint first, then average checkpoints equally. Between SD is the sample SD of checkpoint means. Pooled rollout SD is sqrt(mean(within-checkpoint sample variances)). Neither SD is a confidence interval. Curve averaging and error averaging answer different questions.

The original results_20260919 and resource_results folders, FILE_MANIFEST.json and original figure assets remain historical records. MULTISEED_MANIFEST.json records the earlier multi-seed table/figure mappings and hashes; use the current DDH results listed above for the optimized update. The original FILE_MANIFEST.json is not a manifest of this revised project.

Hard24 uses a fixed cohort across nine runs. CR3 uses a fixed cohort across five runs. Neural checkpoints differ; CPH predictions agree to numerical precision and true-generator oracle predictions are reused controls. Within-model sampler variants share a checkpoint in each training run.

CR3 supplied scalar_metrics_aggregated reports arithmetic means of per-run sampling SDs as a separate metric; the verified_summary.csv pooled_rollout_SD and manuscript use the root-mean-square definition above. CR3 pooled timings exclude seed 42 because of hardware differences; all five runs contribute to accuracy. See SOURCE_COMPARABILITY.txt for the resolved source-version question.

The original paired_patient_comparisons.csv remains conditional on seed-42 fits. It is not a confidence interval for multi-training-run results. All five CR3 runs remain primary; four-run sensitivity files are supplementary descriptive checks.
