# History-conditioned comparison models

This v2 bundle adds three direct predictors to the existing Weibull semi-Markov hard24 experiment:

| Results label | Implementation |
|---|---|
| DynamicDeepHit | Longitudinal Dynamic-DeepHit adaptation: GRU, latest-observation-conditioned attention, conditional PMF, elapsed-time ranking, next-measurement prediction and L1 regularization. |
| SurvTransformer | SurvTRACE **history-feature adaptation**, with the article's feature attention, original-embedding shortcut, SELU alignment, hazard head and two auxiliary tasks. Uses the original continuous-time PCH likelihood. |
| SurvTransformer-Discrete | Identical SurvTRACE history architecture, using an exact discrete-step survival likelihood. Explicit sensitivity variant, not the published objective. |
| DynamicDeepHIt-Lite | Earlier baseline-covariate GRU comparator, renamed as requested. |
| SurvTransformer-Lite | Earlier baseline-covariate transformer PMF comparator, renamed as requested. |

CPH, Weibull, DRSA, all three Glazier architectures and all seven rollout methods remain available. The primary new comparators are single-event models for this hard24 experiment. No full competing-risk replication or new full-training performance result is claimed.

## Information available to the models

Every new comparator trains on the SFSA state sequences. At prediction time it receives **all observed states through the prediction landmark**, never the future trajectory. At t=0 there is no observed state: the input is x and a baseline clock, with missing state indicators. At L=23 the entire s1,...,s23 prefix is available. Dead/censored observations are outcomes, not predictor inputs. Changing a supplied array's suffix after L cannot change the prediction.

Patients are split before any history processing. The training dataset expands each patient's eligible prefixes L=0,...,min(d-1,29), where d is the observed event/censoring step. This includes baseline training examples and uses the full range of observed histories. Prefixes are indexed without duplicating the underlying sequence arrays. The same patient never crosses training/validation/test partitions. Different prefixes of one patient are excluded from ranking pairs.

At nonzero landmarks the new direct models and Glazier now have the same observed covariate and prefix information. The Lite models, CPH, Weibull and DRSA still use their baseline survival-ratio updates. Training objectives and architectures remain different, so this does not make the experiment a controlled ablation of generation alone.

## Dynamic-DeepHit: implemented and adapted components

The implementation follows the article's architecture and equations (3)-(9), rather than reproducing every detail of its public TensorFlow demonstration:

- Measurements contain standardized x, three binary live-state indicators and the known absolute clock in units of 30 steps. Missingness indicators accompany the measurements; the baseline health state is missing. SFSA has regular, fully observed visits, so the inter-visit gap is 1/30; padding is separate from missingness.
- The GRU processes observations **before** the latest visit. A two-layer attention network scores each past hidden state together with the latest measurement and missingness mask. The latest measurement and the attention context feed a cause-specific multilayer head. With no past visit the context is zero.
- One joint softmax represents event-time mass, with a shared beyond-step-30 category. Conditioning removes past event mass and renormalizes over future times plus the tail. This is recomputed from the current history, not from the original baseline prediction. Log-space normalization is used to avoid cancellation.
- Event and censoring likelihoods condition on survival to the last observation. The ranking loss compares patients at equal elapsed horizons since their respective last observations, using the earlier event's cause. It is not the earlier Lite model's common-baseline ranking.
- The auxiliary task predicts the next observed live-state indicators using masked binary cross-entropy, following the article's binary-covariate loss. Only targets already inside the training prefix contribute. Terminal death/censor tokens and their absorbing repetitions are excluded. This is next-measurement supervision, not an auxiliary death label disguised as an input.
- L1 regularization is included. The full notebook enables five auxiliary-only burn-in epochs, following the public implementation's pretraining idea. Subsequent early stopping uses dynamic validation concordance.

SFSA-specific adaptations are the categorical live-state encoding, known absolute clock, covariate-only baseline observation, and explicit beyond-horizon mass. The original finite-lifetime assumption is inappropriate for administrative survivors at step 30. The default single-layer GRU has 64 units; attention and cause-head widths are 64, with two cause-head hidden layers and dropout 0.1. These are configurable implementation defaults, **not recovered original clinical study hyperparameters**.

The article writes sums. Here the NLL, auxiliary sum and pairwise ranking sum are divided by minibatch size; auxiliary BCE still sums observed next-measurement entries. Ranking defaults to `rank_reduction='per_patient'`; `pairs` is an explicit alternative, requiring retuning its weight. Defaults are ranking weight 0.1, ranking scale 0.1, auxiliary weight 0.1 and L1 weight 1e-5. They are recorded and configurable, not claimed to be optimal. Empty comparable-pair sets return zero.

Validation concordance uses the article's equal-elapsed-time comparison idea on a fixed, reproducible subset of validation prefixes, with ties worth one half. It is not an IPCW estimate or the original paper's complete clinical evaluation protocol. If no comparable validation pairs exist, selection falls back to validation loss. No analytical SFSA reference curve or held-out test outcome is used for model selection.

Sources: [Lee, Yoon and van der Schaar, Dynamic-DeepHit](https://doi.org/10.1109/TBME.2019.2909027), [authors' implementation and prefix expansion](https://github.com/chl8856/Dynamic-DeepHit). The PyTorch code here is an independent implementation; it does not import the TensorFlow project.

## SurvTRACE: fidelity and required history adaptation

Published SurvTRACE is a baseline tabular model. Following the user's selected adaptation, each observed SFSA state/time position is a categorical feature field with its own learned embeddings. Numeric x and landmark time are additional feature tokens. Unobserved future fields are masked from attention and zeroed before the readout. This retains feature-token attention while giving the network the full observed sequence; no external run-length rule or handcrafted sojourn summary is supplied.

The encoder implements the **article's** SELU residual attention and feed-forward construction. The readout concatenates contextual feature embeddings with their original embeddings, applies the SELU alignment layer, and uses distinct multilayer heads for hazards, event status and observed duration. This deliberately includes published components missing from the authors' inspected public training path. It is not a byte-for-byte port of that public BERT implementation, which uses a different readout and GELU/layer normalization.

Default dimensions are embedding width 16, three encoder layers, two attention heads, feed-forward/alignment width 64, two hidden layers per task head and dropout 0.1. All are configurable. The mortality task predicts the observed event indicator; it does not treat censoring as proof of eventual survival. The duration task predicts observed event/censoring time, measured in units of the 30-step horizon. Both auxiliary weights default to 1. There is no event-propensity/IPS term in this single-cause experiment: adding a competing-cause propensity model here would not reproduce its intended use.

Both variants output positive integrated hazard increments a_j=softplus(z_j). Conditional survival is

\[
S_L(t\mid H_L)=\exp\{-\sum_{j=L+1}^{t}a_j(H_L)\}.
\]

The PCH objective for an observation at integer step d is

\[
\ell_{\rm PCH}=-\delta\log a_d+\sum_{j=L+1}^{d}a_j.
\]

This is the original continuous-time hazard likelihood, conditional on the landmark, with a full final unit interval. It models event density, not point mass at integer d. The paper's displayed PCH equation and its released code differ; this implementation follows the operational accumulated-hazard likelihood in the authors' released loss.

The separately labeled discrete variant uses

\[
\ell_{\rm discrete}
=-\delta\log(1-e^{-a_d})
 +\sum_{j=L+1}^{d-1}a_j+(1-\delta)a_d.
\]

The associated step-event probability is 1-exp(-softplus(z))=sigmoid(z). Unlike the continuous likelihood, this objective is exact for integer-step events under the stated censoring convention. Keeping both variants tests this modeling mismatch without silently changing the primary replication. Both retain positive horizon survival without an explicit tail logit. Predictions are provided only on the 0,...,30 grid.

Sources: [Wang and Sun, SurvTRACE, Sections 3-4](https://arxiv.org/html/2110.00855v2), [authors' PCH loss](https://github.com/RyanWangZf/SurvTRACE/blob/main/survtrace/losses.py), [authors' public training path](https://github.com/RyanWangZf/SurvTRACE/blob/main/survtrace/train_utils.py).

## Running the comparison

**For the extracted `Run_1_results`, use `Glazier_Hard24_Add_New_Models.ipynb`.** It trains only the three new comparators on the copied original split and reuses 460 archived predictions, including the old direct curves, Glazier methods and oracle sampler. The combined comparison therefore retains the original M=10,000 and evaluation patients. Both new-model training and old prediction reuse are separate from a new all-model run below. Completed new checkpoints can be resumed. Old runtime values retain their original hardware provenance; use the `reused_prediction` column when interpreting timings.

Open `Glazier_Weibull_HardRule24.ipynb` in this folder. It defaults to smoke mode. Run cells in order; then set `MODE='full'` and use the new full output folder. Full mode keeps the original 50,000-person split, M=10,000, five sampling seeds, 300 evaluation patients and landmarks 0,20,23,24. All three new predictors return deterministic direct curves; M does not control their accuracy or runtime.

To reuse the completed hard24 run, set `REUSE_FROM` to the old result folder or its `checkpoints` subfolder. Reuse verifies the rule, cohort size, seed, scenario and generator hashes, then copies the cohort and unchanged models to a **new** folder. The old DynamicDeepHit and SurvTransformer checkpoint labels are mapped to their Lite names. New history models are fitted on the copied training split. Old result files and scores are not edited. Reuse applies only to a compatible hard24 cohort, not the earlier no-hard-rule sampling-budget runs.

`HISTORY_OPTIONS` in the notebook exposes architecture, likelihood-task weights, learning rate, regularization, batch size, burn-in and selection settings. Full mode uses at most 200 joint-training epochs and patience 15. Every training prefix is used each epoch, which is substantially more work than the Lite models' one record per patient. Validation scoring defaults to a fixed maximum of 2,048 prefixes; increase `validation_prefix_limit` for a more extensive validation comparison. Options and source hashes are locked into the run configuration. Change the output folder when changing them.

For rigorous performance claims, tune these explicit defaults on validation data with comparable search budgets and repeat training seeds. The supplied smoke run verifies execution and information boundaries; it does not settle model rankings or replicate the original clinical studies' hyperparameter searches.

## Files and outputs

- `source/history_models.py`: both architectures, both SurvTRACE likelihoods, conditional probability calculations and auxiliary losses.
- `source/history_training.py`: prefix dataset, training, validation and direct prediction.
- `source/extension_workflow.py`: Lite labels, new model dispatch, safe checkpoint restore and observed-history prediction routing.
- `hard_rule_workflow.py`: compatible checkpoint reuse, all-model evaluation and plots.
- `reuse_results.py`: strict saved-prediction validation/import, with legacy-to-Lite mapping.
- `test_reuse_results.py`: cache integrity, mismatch rejection, new-only prediction dispatch and resumability.
- `test_history_models.py`: hand-calculated probability/loss checks, history/padding/leakage checks, training/restore checks and legacy-name protection.

New runs save the cohort, checkpoints, per-epoch training CSVs, settings and hashes, individual predictions, reference curves, summary tables, plots and a results ZIP. The unchanged Lite source filenames and internal class names remain for checkpoint compatibility; their experiment labels are the requested Lite names. Archived v1 runs can be restored for reading but cannot be trained in place through the new API.

The generator and reference recursion are unchanged. Their observed-process KM target and unresolved biological censoring interpretation are unchanged. This code is not a claim of biological identification or of reproducing all features of either published clinical experiment.
