PATHS second independent training seed
=====================================

Open PATHS_Second_Training_Seed.ipynb in Colab with a GPU runtime. The notebook
uses this code ZIP plus the completed hard24 v4 or cr3cov v2 results archive.
It does not contain checkpoints or new experimental results.

1. Upload PATHS_second_seed_bundle.zip when the notebook asks for the code.
2. Choose EXPERIMENT='hard24' and TRAINING_SEED=43. Set PREVIOUS_RUN to the
   completed full_hard24_add_history_v4 result root, or upload its results ZIP.
3. Run preflight, then the full-run cell, then aggregation.
4. Repeat from configuration with EXPERIMENT='cr3', using full_cr3cov_v2.

Keep cohorts, splits, archived hyperparameters, validation-prefix seed 145,
M=10,000, and rollout seeds 41001/52001/63001/74001/85001 unchanged. Use fresh
output directories, such as hard24_training_seed_43 and cr3_training_seed_43.
The archived hard24 v4 run trained only its three newer history comparators;
this runner trains every neural model anew. Do not supply the older Run_1
archive as the completed hard24 v4 run.

The runner copies only the cohort, references and oracle samples. Existing
learned checkpoints and predictions are not copied. Completed new models and
prediction files can resume after interruption; partially trained models
restart because mid-fit optimizer states are not saved.

The original model IDs and filenames are preserved. Figures and exported
tables use ATLAS-architecture-sampler display names. After evaluation, see
ATLAS_figures in each new output directory. atlas_figures.py can also export
ATLAS figures from the old runs without training.

aggregate_seeds.py separates the two checkpoint means from five rollout
repetitions within each checkpoint. Ten rollout estimates are not ten trained
models. Direct models have no within-checkpoint rollout variation to estimate.

Validation completed: preflight on both full archives; small one-epoch CPU
tests of all 19 neural-model instances; same-seed repeatability and different-
seed initialization/order/weights; notebook syntax/schema; synthetic nested-
aggregation tests. Full second-seed training and evaluation have not been run.

Use the accompanying Technical_Assessment.md for the detailed source audit,
exact replacement snippets, local execution instructions and statistical
interpretation. No manuscript project is changed by these tools.
