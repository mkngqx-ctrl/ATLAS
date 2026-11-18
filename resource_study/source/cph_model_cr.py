"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading a saved competing-risks cohort (written by a generate_cohort_cr +
# train_test_val_split_cr driver, mirroring sfsa_data_generation.main())
# ---------------------------------------------------------------------------
def load_cohort_cr_npz(path):
    """
    Competing-risks analogue of cph_model.py's `load_cohort_npz`. Also
    carries `states` (the full 6-state trajectories from
    `generate_cohort_cr`) and a `cause` column (0=censored, 1=disease,
    2=other) in place of the single-risk file's binary `event` column.
    """
    data = np.load(path)
    return {
        "covariate": data["covariate"],
        "states": data["states"],
        "duration": data["duration"],
        "cause": data["cause"],
        "train_index": data["train_index"],
        "test_index": data["test_index"],
        "val_index": data["val_index"],
    }


# ---------------------------------------------------------------------------
# Fitting: two cause-specific Cox models
# ---------------------------------------------------------------------------
def fit_cph_cr(covariate, duration, cause, covariate_name="x", penalizer=0.0):
    """
    Cause-specific-hazards Cox fit. Fits ONE ordinary `CoxPHFitter` per
    cause via the base file's `fit_cph`, each time treating every OTHER
    cause's deaths as censored (the standard competing-risks
    "cause-specific hazard" trick -- see module docstring for why this is a
    valid, separately-identifiable hazard even though no individual ever
    has more than one observed failure time).

    Returns (cph_disease, cph_other).
    """
    cause = np.asarray(cause)
    event_disease = (cause == 1).astype(int)
    event_other = (cause == 2).astype(int)
    # `fit_cph` is defined in cph_model.py, assumed already pasted/run in an
    # earlier cell (see module docstring) -- reused verbatim so both
    # cause-specific fits go through the exact same lifelines call as the
    # base single-risk model.
    cph_disease = fit_cph(covariate, duration, event_disease, covariate_name, penalizer)
    cph_other = fit_cph(covariate, duration, event_other, covariate_name, penalizer)
    return cph_disease, cph_other


# ---------------------------------------------------------------------------
# Prediction: combine the two cause-specific cumulative hazards into CIFs
# ---------------------------------------------------------------------------
def predict_cif_cph(cph_pair, covariate_values, eval_times, covariate_name="x"):
    """
    Predict cause-specific CIF_k(t | x) for each value in `covariate_values`
    at each time in `eval_times`, by combining the two cause-specific Cox
    models' cumulative hazards exactly the way `aalen_johansen_cif` combines
    the nonparametric ones (see module docstring for the formula).

    `cph_pair` is the `(cph_disease, cph_other)` tuple `fit_cph_cr` returns;
    there is no `meta` argument here, matching the base file's special-cased
    `predict_survival(cph, covariate_values, eval_times, covariate_name=...)`
    signature (CPH carries no trained-network `meta` dict).

    Returns an (n, 2, len(eval_times)) array: axis-1 index 0 is the disease
    CIF, index 1 is the other-cause CIF (array-index convention, NOT the
    `cause`-code value). Values are clipped to [0, 1] and, defensively, any
    (individual, time) cell where disease+other would exceed 1+1e-6 (possible
    in principle from the discretization above, though expected to be
    negligible given this scenario's small per-step death probabilities) is
    rescaled proportionally, with a printed warning if that ever triggers.
    """
    cph_disease, cph_other = cph_pair
    query = pd.DataFrame({covariate_name: np.asarray(covariate_values, dtype=np.float64)})
    # lifelines returns (n_times, n_individuals); transpose to (n, n_times)
    # to match this codebase's convention everywhere else.
    Lam_d = cph_disease.predict_cumulative_hazard(query, times=eval_times).to_numpy().T
    Lam_o = cph_other.predict_cumulative_hazard(query, times=eval_times).to_numpy().T

    S = np.exp(-(Lam_d + Lam_o))  # overall survival; S(t=0) = 1 exactly (no time has elapsed)
    S_prev = np.concatenate([np.ones((S.shape[0], 1)), S[:, :-1]], axis=1)  # pre-jump S(t_{j-1})
    dLam_d = np.diff(Lam_d, axis=1, prepend=0.0)
    dLam_o = np.diff(Lam_o, axis=1, prepend=0.0)
    cif_d = np.cumsum(S_prev * dLam_d, axis=1)
    cif_o = np.cumsum(S_prev * dLam_o, axis=1)

    cif_d = np.clip(cif_d, 0.0, 1.0)
    cif_o = np.clip(cif_o, 0.0, 1.0)
    total = cif_d + cif_o
    over = total > 1 + 1e-6
    n_over = int(over.sum())
    if n_over:
        print(f"  [predict_cif_cph] clipped {n_over} (individual, time) cell(s) where "
              f"cif_disease + cif_other exceeded 1 + 1e-6 (max excess "
              f"{float((total - 1)[over].max()):.2e}); rescaling those cells proportionally.")
        scale = np.ones_like(total)
        scale[over] = 1.0 / total[over]
        cif_d = cif_d * scale
        cif_o = cif_o * scale

    return np.stack([cif_d, cif_o], axis=1)  # (n, 2, len(eval_times))


def mae_per_timestep(predicted, ground_truth):
    """MAE_t: mean absolute error between predicted and ground-truth curves
    at each time step, averaged over individuals. Both arrays (n, n_times)."""
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario
# ---------------------------------------------------------------------------
def run_cph_cr_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                             demo_values=None, seed=4242):
    """
    Competing-risks analogue of `run_cph_for_scenario`: fits the two
    cause-specific Cox models, combines them into CIFs via
    `predict_cif_cph`, then scores against
    `ground_truth_cif_for_validation_set` (the Aalen-Johansen ground truth
    for overall survival AND both cause-specific CIFs) instead of the
    single-risk `ground_truth_km_for_validation_set`. Also reports MAE_t
    for the derived overall-survival curve `1 - cif_disease - cif_other`,
    since that is the natural CPH-vs-ground-truth comparison a reader would
    want alongside the two cause-specific curves.

    Returns (cph_pair, eval_times, mae_t_survival, mae_t_disease,
    mae_t_other, demo_curves).
    """
    cohort = load_cohort_cr_npz(cohort_path)
    train_idx = cohort["train_index"]
    val_idx = cohort["val_index"]

    cph_pair = fit_cph_cr(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["cause"][train_idx],
        covariate_name=scenario.covariate_name,
    )
    cph_disease, cph_other = cph_pair
    print(f"\n=== CPH (competing risks) fit on {scenario.name} scenario "
          f"(covariate: {scenario.covariate_name}) ===")
    print("-- disease-cause model (other cause treated as censored) --")
    print(cph_disease.summary[["coef", "exp(coef)", "p"]])
    print("-- other-cause model (disease treated as censored) --")
    print(cph_other.summary[["coef", "exp(coef)", "p"]])

    eval_times = np.arange(0, T_K + 1)

    # --- MAE_t on a validation subsample ---
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted_cif = predict_cif_cph(cph_pair, sample_covariate, eval_times, covariate_name=scenario.covariate_name)
    predicted_survival = 1.0 - predicted_cif[:, 0, :] - predicted_cif[:, 1, :]

    ground_truth = ground_truth_cif_for_validation_set(  # defined in the SFSA-CR cell already run
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
    )

    mae_t_survival = mae_per_timestep(predicted_survival, ground_truth["survival"])
    mae_t_disease = mae_per_timestep(predicted_cif[:, 0, :], ground_truth["cif_disease"])
    mae_t_other = mae_per_timestep(predicted_cif[:, 1, :], ground_truth["cif_other"])

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gt_n_matched}-person ground truth each):")
    print(f"  mean MAE (survival):   {mae_t_survival.mean():.4f}")
    print(f"  mean MAE (disease CIF): {mae_t_disease.mean():.4f}")
    print(f"  mean MAE (other CIF):   {mae_t_other.mean():.4f}")
    for t in (0, 10, 20, 30):
        if t < len(eval_times):
            print(f"  t={t}: survival MAE={mae_t_survival[t]:.4f}  "
                  f"disease MAE={mae_t_disease[t]:.4f}  other MAE={mae_t_other[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t_survival, label="Survival", color="tab:cyan")
    plt.plot(eval_times, mae_t_disease, label="Disease CIF", color="tab:red", linestyle="--")
    plt.plot(eval_times, mae_t_other, label="Other CIF", color="tab:gray", linestyle=":")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"CPH (competing risks) MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- demo individuals (3 rows: overall survival, disease CIF, other CIF) ---
    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    demo_cif = predict_cif_cph(cph_pair, demo_values, eval_times, covariate_name=scenario.covariate_name)
    fig, axes = plt.subplots(3, len(demo_values), figsize=(5 * len(demo_values), 11), sharex=True)
    if len(demo_values) == 1:
        axes = axes.reshape(3, 1)  # plt.subplots squeezes size-1 dims away by default

    demo_curves = {}
    for col, v in enumerate(demo_values):
        gt = ground_truth_cif(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        sf_pred = 1.0 - demo_cif[col, 0, :] - demo_cif[col, 1, :]
        demo_curves[v] = {
            "ground_truth_survival": gt["survival"], "ground_truth_disease": gt["cif_disease"],
            "ground_truth_other": gt["cif_other"],
            "survival": sf_pred, "disease": demo_cif[col, 0, :], "other": demo_cif[col, 1, :],
        }

        axes[0, col].step(gt["eval_times"], gt["survival"], where="post", label="Ground truth", color="tab:red")
        axes[0, col].step(eval_times, sf_pred, where="post", label="CPH", color="tab:cyan")
        axes[0, col].set_title(f"{scenario.covariate_name}={v}\nOverall survival")
        axes[0, col].legend(fontsize=8)

        axes[1, col].step(gt["eval_times"], gt["cif_disease"], where="post", color="tab:red")
        axes[1, col].step(eval_times, demo_cif[col, 0, :], where="post", color="tab:cyan")
        axes[1, col].set_title("Disease CIF")

        axes[2, col].step(gt["eval_times"], gt["cif_other"], where="post", color="tab:red")
        axes[2, col].step(eval_times, demo_cif[col, 1, :], where="post", color="tab:cyan")
        axes[2, col].set_title("Other CIF")
        axes[2, col].set_xlabel("Time")

    axes[0, 0].set_ylabel("Survival probability")
    axes[1, 0].set_ylabel("CIF")
    axes[2, 0].set_ylabel("CIF")
    plt.suptitle(f"CPH (competing risks) vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return cph_pair, eval_times, mae_t_survival, mae_t_disease, mae_t_other, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios. `cph_cr_relu` / `cph_cr_weibull` end up being the
# `(cph_disease, cph_other)` tuples themselves (the first element of
# `run_cph_cr_for_scenario`'s return), matching the base file's CPH
# special-casing (no `_model` infix) and the model-comparison harness's
# `{model_prefix}_cr_{relu|weibull}` naming convention with `model_prefix =
# "cph_cr"`.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    cph_cr_relu, eval_times_cr_relu, mae_t_survival_cr_relu, mae_t_disease_cr_relu, mae_t_other_cr_relu, demo_cr_relu = run_cph_cr_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort_cr.npz",
    )
    cph_cr_weibull, eval_times_cr_weibull, mae_t_survival_cr_weibull, mae_t_disease_cr_weibull, mae_t_other_cr_weibull, demo_cr_weibull = run_cph_cr_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort_cr.npz",
    )
