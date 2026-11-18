"""
sfsa_semi_markov_extras.py

Small companion module to `sfsa_semi_markov.py`: adds the PROSPECTIVE
(t=0) Kaplan-Meier ground-truth helpers that module does not ship.

WHY THIS IS NEEDED:
`sfsa_semi_markov.py` only exports LANDMARK ground truth
(`ground_truth_km_landmark_sm`, `ground_truth_km_landmark_sm_for_validation_set`)
-- by design, per its own module docstring: "at t=0 every individual's
sojourn time is trivially 0/1, so this change alone (without landmark
evaluation) reproduces the 'no information asymmetry' non-result." That's
true, but it also means there is currently no way to compute a PROSPECTIVE
(t=0) ground-truth curve under the semi-Markov hazard dynamics at all --
and a prospective comparison is still useful, precisely BECAUSE it is
expected to reproduce that null result: it's the documented sanity check
that every model looks similar when nobody (model or ground truth) has any
sojourn-time information to exploit yet, which is what makes the later
landmark-protocol divergence meaningful by contrast instead of an
artifact. This module fills that one gap.

`ground_truth_km_sm` / `ground_truth_km_sm_for_validation_set` below are
built exactly the way the base generator's own prospective ground truth is
(`sfsa_data_generation.ground_truth_km` / `ground_truth_km_for_validation_set`),
just swapping in the semi-Markov stepping function: simulate `n_matched`
individuals sharing one covariate value from t=0 via `_simulate_states_sm`
(instead of `_simulate_states`), then take the Kaplan-Meier estimate of the
resulting (duration, event) pairs.

The per-individual seeding idiom in `ground_truth_km_sm_for_validation_set`
is copied VERBATIM from
`sfsa_semi_markov.ground_truth_km_landmark_sm_for_validation_set` for
consistency: spawn one child `np.random.SeedSequence` per individual and
pass its `.entropy` straight through as the concrete integer seed (not,
e.g., `.generate_state(1)[0]` -- that would be a different, merely
plausible-looking idiom; this file uses the one the real codebase actually
uses).

Paste/import order: this module does real `from ... import ...` statements
(matching `sfsa_semi_markov.py`'s own style) against `sfsa_data_generation.py`
and `sfsa_semi_markov.py`, so those two must already be importable (saved
as files alongside this one, or already executed earlier in the same
notebook/script) before this module is used.
"""

from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter

from sfsa_data_generation import make_rng, _extract_duration_event, T_K
from sfsa_semi_markov import _simulate_states_sm, DEFAULT_HAZARD_PARAMS


# ---------------------------------------------------------------------------
# Prospective (t=0) ground truth under semi-Markov dynamics
# ---------------------------------------------------------------------------
def ground_truth_km_sm(scenario, covariate_value, seed, n_matched=10_000,
                        t_k=T_K, **hazard_kwargs):
    """
    Prospective (t=0) KM ground truth under semi-Markov dynamics -- the
    direct semi-Markov analogue of `sfsa_data_generation.ground_truth_km`,
    using `_simulate_states_sm` instead of `_simulate_states`.

    Simulates `n_matched` individuals all sharing `covariate_value`, from
    t=0, with the SAME semi-Markov hazard-ramp dynamics used to generate
    the training cohort -- pass the cohort's hazard parameters through
    `**hazard_kwargs` (e.g. `**DEFAULT_HAZARD_PARAMS`, or whatever was
    actually used to call `generate_cohort_sm`); an empty `hazard_kwargs`
    falls back to `sojourn_boosted_transition_matrix`'s own defaults, which
    ARE `DEFAULT_HAZARD_PARAMS` -- so passing nothing is only correct if
    the cohort was generated with the defaults too.

    Returns (eval_times, survival_probability) for t = 0..t_k, matching
    `ground_truth_km`'s signature exactly.
    """
    rng = make_rng(seed)
    covariate = np.full(n_matched, float(covariate_value))
    states = _simulate_states_sm(covariate, scenario, rng, t_k=t_k, **hazard_kwargs)
    duration, event = _extract_duration_event(states, t_k)

    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=event)
    eval_times = np.arange(0, t_k + 1)
    sf = kmf.survival_function_at_times(eval_times).to_numpy()
    return eval_times, sf


def ground_truth_km_sm_for_validation_set(scenario, val_covariates, base_seed,
                                           n_matched=10_000, t_k=T_K, **hazard_kwargs):
    """
    Prospective ground-truth KM curve for every individual in a validation
    set, under semi-Markov dynamics -- the semi-Markov analogue of
    `sfsa_data_generation.ground_truth_km_for_validation_set`. Returns an
    array of shape (len(val_covariates), t_k + 1).

    Per-individual deterministic seeding copied verbatim from
    `sfsa_semi_markov.ground_truth_km_landmark_sm_for_validation_set`: spawn
    one child `SeedSequence` per individual via
    `np.random.SeedSequence(base_seed).spawn(n)`, then pass each child's
    `.entropy` straight through as the concrete seed -- so re-running this
    function reproduces the exact same ground-truth curves.
    """
    seed_seq = np.random.SeedSequence(base_seed)
    child_seeds = seed_seq.spawn(len(val_covariates))
    curves = np.empty((len(val_covariates), t_k + 1), dtype=np.float64)
    for i, (cov, child_seed) in enumerate(zip(val_covariates, child_seeds)):
        _, sf = ground_truth_km_sm(
            scenario, float(cov), int(child_seed.generate_state(1)[0]), n_matched, t_k, **hazard_kwargs,
        )
        curves[i] = sf
    return curves
