"""
generate_cohorts_sm.py
========================

Small, one-time utility that was missing from the semi-Markov extension:
`sfsa_semi_markov.py` defines `generate_cohort_sm` / `train_test_val_split_sm`
but (unlike `sfsa_data_generation.py`, which ships its own `main()` that saves
`{scenario}_cohort.npz`) never assembles and saves the semi-Markov cohort to
disk. Every `*_model_sm.py` file and `model_comparison_semimarkov.py` load
from `/content/{scenario}_cohort_sm.npz` — this script is what produces
those two files.

Paste-into-Colab-cell convention, same as the rest of this codebase: run
`sfsa_data_generation.py` then `sfsa_semi_markov.py` in earlier cells first
(this script uses `RELU_SCENARIO`, `WEIBULL_SCENARIO`, `T_K`,
`DEFAULT_HAZARD_PARAMS`, `generate_cohort_sm`, `train_test_val_split_sm`,
`validate_hazard_monotonicity` directly from those, assumed already in the
global namespace), then run this.

Usage:
    python3 generate_cohorts_sm.py
or, in a Colab cell, just paste the body below `if __name__ == "__main__":`.
"""

import numpy as np

# Fresh seeds for the semi-Markov cohorts (distinct from the base
# generator's SEEDS dict and from the competing-risks cohort's SEEDS_CR, so
# all three cohort families are independently reproducible).
SEEDS_SM = {
    "ReLU_cohort_sm": 21001,
    "ReLU_split_sm": 21002,
    "Weibull_cohort_sm": 22001,
    "Weibull_split_sm": 22002,
}


def build_and_save_cohort_sm(scenario, cohort_seed, split_seed, n_cohort=50_000,
                              t_k=None, hazard_kwargs=None, out_dir="/content/"):
    """Generate one scenario's semi-Markov cohort, run the hazard-monotonicity
    sanity check on a small subsample, split, and save to
    `{out_dir}/{scenario_slug}_cohort_sm.npz` — same `.npz` schema as the base
    (non-semi-Markov) cohort, since generate_cohort_sm's output format is
    deliberately unchanged from generate_cohort (see sfsa_semi_markov.py's
    own module docstring)."""
    t_k = T_K if t_k is None else t_k
    hazard_kwargs = dict(DEFAULT_HAZARD_PARAMS) if hazard_kwargs is None else dict(hazard_kwargs)

    cohort = generate_cohort_sm(scenario, n=n_cohort, seed=cohort_seed, t_k=t_k, **hazard_kwargs)

    subsample_n = min(500, n_cohort)
    validate_hazard_monotonicity(cohort["states"][:subsample_n], t_k=t_k, **hazard_kwargs)

    split = train_test_val_split_sm(cohort, seed=split_seed)

    n = cohort["covariate"].shape[0]
    train_index, test_index, val_index = split["train"]["index"], split["test"]["index"], split["val"]["index"]

    scenario_slug = scenario.name.lower()
    out_path = f"{out_dir}{scenario_slug}_cohort_sm.npz"
    np.savez_compressed(
        out_path,
        covariate=cohort["covariate"],
        states=cohort["states"],
        duration=cohort["duration"],
        event=cohort["event"],
        train_index=train_index,
        test_index=test_index,
        val_index=val_index,
        seed=cohort_seed,
    )
    n_dead = int(cohort["event"].sum())
    n_censored = n - n_dead
    print(f"[{scenario.name}] saved {out_path}  (n={n}: "
          f"observed death={n_dead} [{n_dead/n:.3f}], "
          f"censored={n_censored} [{n_censored/n:.3f}]), "
          f"hazard_kwargs={hazard_kwargs}")
    return out_path


def main(n_cohort=50_000, out_dir="/content/", hazard_kwargs=None):
    build_and_save_cohort_sm(RELU_SCENARIO, SEEDS_SM["ReLU_cohort_sm"], SEEDS_SM["ReLU_split_sm"],
                              n_cohort=n_cohort, out_dir=out_dir, hazard_kwargs=hazard_kwargs)
    build_and_save_cohort_sm(WEIBULL_SCENARIO, SEEDS_SM["Weibull_cohort_sm"], SEEDS_SM["Weibull_split_sm"],
                              n_cohort=n_cohort, out_dir=out_dir, hazard_kwargs=hazard_kwargs)


if __name__ == "__main__":
    main(n_cohort=50_000, out_dir="/content/")
