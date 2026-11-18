# Weibull semi-Markov hard24 benchmark — history comparators and saved-run reuse

## Add new models to your saved Run 1 (recommended for existing results)

Open **Glazier_Hard24_Add_New_Models.ipynb**, select your **gpu_env** kernel, restart it, and run all cells in order. It is preset to `Run_1_results` beside the notebook. It validates and copies the existing 50,000-person cohort, split and eight checkpoints; trains only DynamicDeepHit, SurvTransformer and SurvTransformer-Discrete; and imports all **460 saved prediction files** instead of regenerating old rollouts. The original models are displayed with the requested Lite names. New comparisons retain M=10,000, the same 300 starting test patients, five seeds and landmarks 0,20,23,24.

Combined tables and figures are saved under `results/full_hard24_add_history_v3/evaluation`, with figures in its `plots` subfolder. The final cell makes a combined results ZIP. `Run_1_results` is not modified. Rerunning with identical options restores completed new models and cached predictions; interrupted model training restarts for that model. Changing source code/options requires a fresh output folder.

The read-only preflight checks cohort and checkpoint hashes, actual model weights, source compatibility, evaluation patients, time grids and deterministic reference curves. `reused_predictions.json` records copied files and their hashes; metric tables include `reused_prediction`. Original runtime measurements are preserved and are **not** necessarily comparable to new timings on different hardware.

For Colab use **Glazier_Weibull_HardRule24_Reuse_v3.zip** and supply the extracted old results separately (the source ZIP excludes results/checkpoints). Set `PREVIOUS_RUN` and `OUT` to the appropriate paths, preferably on Drive. Run `python -m unittest test_reuse_results -v` for the eight cache tests; these do not train models.

## Train a new comparison from scratch

Open `Glazier_Weibull_HardRule24.ipynb` with your GPU kernel or Colab and run cells in order. Start with `MODE='smoke'`; use `MODE='full'` for the performance comparison. The current source bundle is `Glazier_Weibull_HardRule24_Reuse_v3.zip`. Optional Drive output preserves progress across resets.

## New models and names

- **DynamicDeepHit:** longitudinal GRU, latest-conditioned attention, conditional PMF, dynamic ranking, next-measurement supervision and L1 regularization.
- **SurvTransformer:** SurvTRACE adapted to observed state/time feature tokens, retaining its original hazard objective, embedding shortcut and two auxiliary tasks.
- **SurvTransformer-Discrete:** the same history architecture with an exact discrete-step likelihood, separately labeled.
- The previous implementations are **DynamicDeepHIt-Lite** and **SurvTransformer-Lite**.

Both new architectures consume every observed state through the landmark. Future states are excluded; baseline has no observed prefix. CPH, Weibull, DRSA, three Glazier architectures and all seven samplers remain available. Read **HISTORY_COMPARATORS.md** for equations, sources, fidelity decisions and limitations.

The t=20 duration-risk change and hard rule from t=24 are unchanged. From t=24, a proposed Sicker immediately following Sicker becomes Dead. The generator and reference enforce this; models learn it from data.

## Full runs and checkpoint reuse

Full mode uses 50,000 patients, an 80/10/10 split, at most 200 joint-training epochs, patience 15, M=10,000, five rollout seeds, 300 test patients and landmarks 0,20,23,24. DynamicDeepHit also has five auxiliary burn-in epochs. Prefix expansion makes training more expensive than Lite training.

Set `REUSE_FROM` to compatible completed hard24 results to copy their cohort and unchanged checkpoints into a fresh folder. Old files remain untouched and old names become Lite. New models are trained on the copied split. Source/settings/checkpoint hashes protect resumability. Changing code or options requires a new output folder.

New full performance runs remain pending. Smoke scores are execution checks. Existing validation records concern the original generator/samplers; new validation records use history-prefixed names.

## Tests, dependencies and outputs

The notebook runs new model tests and the existing rule/sampler checks. Run the new tests alone with `python -m unittest test_history_models -v` from this directory. No pytest dependency is needed. Colab installs lifelines 0.30.3, pandas and matplotlib while retaining PyTorch. The notebook can find the workspace support libraries for the local gpu_env.

Outputs include the cohort, model checkpoints, per-epoch training CSVs, configuration/source hashes, individual predictions, references, metrics, plots and a results ZIP. The reference is still an observed-generator KM functional with unresolved biological interpretation. Rollout-seed error bars are not training uncertainty; direct models have no rollout sampling error.
