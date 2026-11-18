"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter


# ---------------------------------------------------------------------------
# Loading a saved cohort back in (written by sfsa_data_generation.main())
# ---------------------------------------------------------------------------
def load_cohort_npz(path):
    data = np.load(path)
    return {
        "covariate": data["covariate"],
        "duration": data["duration"],
        "event": data["event"],
        "train_index": data["train_index"],
        "test_index": data["test_index"],
        "val_index": data["val_index"],
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_cph(covariate, duration, event, covariate_name="x", penalizer=0.0):
    """
    Fit a Cox Proportional Hazards model on one scenario's training split.

    lifelines wants one row per individual in a DataFrame with a duration
    column, a 0/1 event column, and the covariate column(s) -- so we
    assemble that here. `penalizer` adds a bit of L2 regularization if the
    fit is unstable (rarely needed with a single covariate, but cheap
    insurance).
    """
    df = pd.DataFrame({
        covariate_name: np.asarray(covariate, dtype=np.float64),
        "duration": np.asarray(duration, dtype=np.float64),
        "event": np.asarray(event, dtype=np.int64),
    })
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="duration", event_col="event")
    return cph


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_survival(cph, covariate_values, eval_times, covariate_name="x"):
    """
    Predict S(t | x) for each value in `covariate_values`, at each time in
    `eval_times`. Returns an array of shape (n_individuals, n_times) so it
    lines up directly with `ground_truth_km_for_validation_set`'s output.
    """
    query = pd.DataFrame({covariate_name: np.asarray(covariate_values, dtype=np.float64)})
    sf = cph.predict_survival_function(query, times=eval_times)
    # lifelines returns a DataFrame shaped (n_times, n_individuals); transpose
    # to match the (n_individuals, n_times) convention used elsewhere here.
    return sf.to_numpy().T


def mae_per_timestep(predicted, ground_truth):
    """
    MAE_t (paper, just after Section 3.2): mean absolute error between
    predicted and ground-truth survival probability at each time step,
    averaged over individuals. Both arrays are (n_individuals, n_times).
    """
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario
# ---------------------------------------------------------------------------
def run_cph_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                          demo_values=None, seed=4242):
    """
    Fit CPH on the scenario's training split, then:
      1. print the fitted coefficient / hazard ratio,
      2. compute an MAE_t curve on a random subsample of the validation set
         (the paper uses the *full* validation set, ~5000 individuals here;
         each one needs a fresh 10,000-person ground-truth simulation, which
         is the expensive step described in Section 4 -- n_mae_sample lets
         you trade accuracy of the MAE curve for runtime; bump it toward
         5000 for a closer match to the paper once you know things work),
      3. plot predicted vs. ground-truth survival curves for the paper's
         demo covariate values (Figure 4/5 style).

    Returns (cph, eval_times, mae_t, demo_curves).
    """
    cohort = load_cohort_npz(cohort_path)
    train_idx = cohort["train_index"]
    val_idx = cohort["val_index"]

    cph = fit_cph(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["event"][train_idx],
        covariate_name=scenario.covariate_name,
    )
    print(f"\n=== CPH fit on {scenario.name} scenario (covariate: {scenario.covariate_name}) ===")
    print(cph.summary[["coef", "exp(coef)", "p"]])
    # exp(coef) is the hazard ratio: how much a one-unit increase in the
    # covariate multiplies risk. >1 means higher covariate -> higher hazard
    # (worse survival); <1 means higher covariate -> lower hazard.

    eval_times = np.arange(0, sdg_T_K() + 1)

    # --- MAE_t on a validation subsample ---
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted = predict_survival(cph, sample_covariate, eval_times, covariate_name=scenario.covariate_name)
    ground_truth = ground_truth_km_for_validation_set(  # defined in the SFSA cell already run
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=sdg_T_K()
    )
    mae_t = mae_per_timestep(predicted, ground_truth)

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gt_n_matched}-person ground truth each):")
    print(f"  mean MAE across all t: {mae_t.mean():.4f}")
    for t in (0, 10, 20, 30):
        print(f"  MAE at t={t}: {mae_t[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t, label="CPH")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"CPH MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- demo individuals (Figure 4 / Figure 5 style) ---
    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    fig, axes = plt.subplots(1, len(demo_values), figsize=(5 * len(demo_values), 4), sharey=True)
    demo_curves = {}
    for ax, v in zip(np.atleast_1d(axes), demo_values):
        t_gt, sf_gt = ground_truth_km(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=sdg_T_K())
        sf_pred = predict_survival(cph, [v], eval_times, covariate_name=scenario.covariate_name)[0]
        demo_curves[v] = {"ground_truth": sf_gt, "cph": sf_pred}

        ax.step(t_gt, sf_gt, where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, sf_pred, where="post", label="CPH", color="tab:cyan")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.set_xlabel("Time")
        ax.legend()
    axes_list = np.atleast_1d(axes)
    axes_list[0].set_ylabel("Survival probability")
    plt.suptitle(f"CPH vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return cph, eval_times, mae_t, demo_curves


def sdg_T_K():
    """Small indirection so this file works whether your notebook's SFSA
    cell defined the constant as T_K (module) or you're calling this from a
    context where it's named differently -- edit here if needed."""
    return T_K


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    cph_relu, eval_times_relu, mae_t_relu, demo_relu = run_cph_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort.npz",
    )
    cph_weibull, eval_times_weibull, mae_t_weibull, demo_weibull = run_cph_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort.npz",
    )
