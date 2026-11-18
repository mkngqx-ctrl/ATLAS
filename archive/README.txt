CURRENT MANUSCRIPT RESULTS — updated 2026-11-16

Main comparisons now use both DRSA (optimized) and Dynamic-DeepHit (optimized).
Use supplementary/optimized_DRSA_integrated_2026-11-16/ for current integrated
all-model summary/per-training-run metrics, updated bootstrap results and figure
data. In those machine-readable comparison files the internal model key DRSA
means DRSA (optimized), and DynamicDeepHit means Dynamic-DeepHit (optimized).
Earlier result folders/manifests are archived source records and may contain
superseded DRSA rows; they do not define the current manuscript comparisons.

ATLAS DATA AND CODE PACKAGE

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

Package directories:
- supplementary/: multi-seed results and supplementary analyses; see its README.txt.
- reference_code/: benchmark/reference implementations; see its README.md.
- classical_comparators/: exploratory classical comparator materials.
- results_20260919/ and resource_results/: historical result/resource records.

The manuscript supplement contains Sections S1-S10. Original DDH results are
historical transparency materials and should not replace optimized DDH values
in the current main comparisons.
