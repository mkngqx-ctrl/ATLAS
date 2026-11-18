"""
generate_cohorts_cr.py
=======================

Small, one-time utility that was missing from the competing-risks extension:
`sfsa_competing_risks.py` defines `generate_cohort_cr` / `train_test_val_split_cr`
but (unlike `sfsa_data_generation.py`, which ships its own `main()` that saves
`{scenario}_cohort.npz`) never assembles and saves the competing-risks cohort
to disk. Every `*_model_cr.py` file and `model_comparison_competing_risks.py`
load from `/content/{scenario}_cohort_cr.npz` — this script is what produces
those two files.

Paste-into-Colab-cell convention, same as the rest of this codebase: run
`sfsa_data_generation.py` then `sfsa_competing_risks.py` in earlier cells
first (this script uses `RELU_SCENARIO`, `WEIBULL_SCENARIO`, `T_K`,
`generate_cohort_cr`, `train_test_val_split_cr`, `validate_generation_invariants_cr`
directly from those, assumed already in the global namespace), then run this.

Usage:
    python3 generate_cohorts_cr.py
or, in a Colab cell, just paste the body below `if __name__ == "__main__":`.
"""

import numpy as np

# Fresh seeds for the competing-risks cohorts (distinct from the base
# generator's SEEDS dict in sfsa_data_generation.py, to keep the two cohorts
# independently reproducible rather than accidentally correlated).
SEEDS_CR = {
    "ReLU_cohort_cr": 11001,
    "ReLU_split_cr": 11002,
    "Weibull_cohort_cr": 12001,
    "Weibull_split_cr": 12002,
}


def build_and_save_cohort_cr(scenario, cohort_seed, split_seed, n_cohort=50_000,
                              t_k=None, cause_a_share_fn=None, out_dir="/content/"):
    """Generate one scenario's competing-risks cohort, run the invariant
    check on a small subsample (mirrors sfsa_data_generation.main()'s own
    pattern), split, and save to `{out_dir}/{scenario_slug}_cohort_cr.npz`."""
    t_k = T_K if t_k is None else t_k
    kwargs = {} if cause_a_share_fn is None else {"cause_a_share_fn": cause_a_share_fn}

    cohort = generate_cohort_cr(scenario, n=n_cohort, seed=cohort_seed, t_k=t_k, **kwargs)

    # Invariant check on a small subsample, same spirit as the base
    # generator's main() (full-cohort checks would be slow and are
    # unnecessary to catch a systematic generation bug).
    subsample_n = min(500, n_cohort)
    validate_generation_invariants_cr(cohort["states"][:subsample_n], t_k=t_k)

    split = train_test_val_split_cr(cohort, seed=split_seed)

    n = cohort["covariate"].shape[0]
    train_index, test_index, val_index = split["train"]["index"], split["test"]["index"], split["val"]["index"]

    scenario_slug = scenario.name.lower()
    out_path = f"{out_dir}{scenario_slug}_cohort_cr.npz"
    np.savez_compressed(
        out_path,
        covariate=cohort["covariate"],
        states=cohort["states"],
        duration=cohort["duration"],
        cause=cohort["cause"],
        train_index=train_index,
        test_index=test_index,
        val_index=val_index,
        seed=cohort_seed,
    )
    n_disease = int((cohort["cause"] == 1).sum())
    n_other = int((cohort["cause"] == 2).sum())
    n_censored = int((cohort["cause"] == 0).sum())
    print(f"[{scenario.name}] saved {out_path}  (n={n}: "
          f"disease={n_disease} [{n_disease/n:.3f}], "
          f"other={n_other} [{n_other/n:.3f}], "
          f"censored={n_censored} [{n_censored/n:.3f}])")
    return out_path


def main(n_cohort=50_000, out_dir="/content/"):
    build_and_save_cohort_cr(RELU_SCENARIO, SEEDS_CR["ReLU_cohort_cr"], SEEDS_CR["ReLU_split_cr"],
                              n_cohort=n_cohort, out_dir=out_dir)
    build_and_save_cohort_cr(WEIBULL_SCENARIO, SEEDS_CR["Weibull_cohort_cr"], SEEDS_CR["Weibull_split_cr"],
                              n_cohort=n_cohort, out_dir=out_dir)


if __name__ == "__main__":
    main(n_cohort=50_000, out_dir="/content/")
