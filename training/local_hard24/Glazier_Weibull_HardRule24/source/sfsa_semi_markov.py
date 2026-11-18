# HardRule24: retain t=20 ramp; from t=24, Sicker -> Sicker becomes Dead.
# Original descriptive text below predates this additional override.
"""
sfsa_semi_markov.py

Item 2 from glazier_differentiation_designs.md: a semi-Markov,
sojourn-time-dependent-hazard extension of the SFSA generator, meant to be
evaluated together with Item 1's landmark/dynamic-prediction protocol
(sfsa_data_generation.ground_truth_km_landmark*) -- see the caveat in the
design doc: at t=0 every individual's sojourn time is trivially 0/1, so
this change alone (without landmark evaluation) reproduces the "no
information asymmetry" non-result. The whole point is for a landmark-style
evaluator to condition on an individual's OBSERVED sojourn time and expose
exactly the signal that only a model which tracks/generates real history
(Glazier) can use.

Design:

  - The base generator's only history-dependence is an ad hoc hard rule:
    "N consecutive Sicker states forces Dead" (N=5 before step 20, N=3
    from step 20 on). That is a discontinuous step function of the
    Sicker-specific run length, and it is the ONLY place sojourn time
    matters anywhere in the base chain -- Healthy->Sick and Sick->Sicker
    are otherwise perfectly memoryless (first-order Markov in state alone).
  - This module generalizes that into a genuine semi-Markov chain: the
    Sick->Sicker and Sicker->Dead transition probabilities now rise
    SMOOTHLY with `sojourn` -- the number of consecutive steps an
    individual has already spent in their CURRENT state (Sick or Sicker,
    respectively) -- via a Weibull-shaped hazard ramp (increasing hazard
    with duration, same functional family as the paper's own Weibull
    scenario), instead of only a sudden jump to forced-death at a fixed
    run length.
  - The old rule's reference run-lengths (5 before step 20, 3 from step 20
    on) are kept as the ramp's *reference scale*: the added Sicker->Dead
    hazard reaches ~0.97 (near-certain) right around the old threshold,
    so the two behave similarly in the large-sojourn limit, but now values
    BELOW the old threshold carry real, smoothly-varying information
    instead of being indistinguishable from "nothing has happened yet."
  - The hard deterministic override is removed entirely in favor of this
    probabilistic ramp -- this is now a genuine (if extreme near the old
    reference point) semi-Markov hazard, not a Markov chain with a
    bolted-on deterministic rule. There is consequently no hard upper
    bound on an individual's Sicker run length any more (just vanishing
    probability of exceeding it) -- a deliberate, realistic relaxation.
  - Reallocating probability mass: boosting Sick->Sicker or Sicker->Dead
    takes the added mass proportionally from that row's OTHER live-state
    entries (never touching the tiny background-death or censoring
    columns), so rows still sum to exactly 1 and nothing goes negative.

This changes the chain's dynamics relative to the base generator (unlike
Item 3, which preserved the base calibration exactly by construction), so
the resulting mean/median time-to-death and censoring rates are NOT
guaranteed to match the paper's reported targets -- `cohort_summary` (from
sfsa_data_generation, directly reusable since states/duration/event share
the same encoding) should be used to check where they land, and the
hazard-ramp parameters below are there to be recalibrated if an exact
match matters for the paper's narrative.
"""

from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter

from sfsa_data_generation import (
    HEALTHY, SICK, SICKER, DEAD, CENSORED, N_STATES, T_K, STATE_NAMES,
    Scenario, RELU_SCENARIO, WEIBULL_SCENARIO,
    make_rng, sample_covariate, transition_matrix, _extract_duration_event,
    cohort_summary,
)

# ---------------------------------------------------------------------------
# Hazard-ramp defaults (see module docstring for the rationale)
# ---------------------------------------------------------------------------
DEFAULT_HAZARD_PARAMS = dict(
    lambda_sick_to_sicker=0.35, kappa_sick_to_sicker=2.5,
    lambda_sicker_to_dead=0.97, kappa_sicker_to_dead=3.0,
)


# ---------------------------------------------------------------------------
# Sojourn-boosted transition matrix
# ---------------------------------------------------------------------------
def sojourn_boosted_transition_matrix(
    F: np.ndarray, p: float, sojourn: np.ndarray, abs_step: int,
    lambda_sick_to_sicker: float = DEFAULT_HAZARD_PARAMS["lambda_sick_to_sicker"],
    kappa_sick_to_sicker: float = DEFAULT_HAZARD_PARAMS["kappa_sick_to_sicker"],
    lambda_sicker_to_dead: float = DEFAULT_HAZARD_PARAMS["lambda_sicker_to_dead"],
    kappa_sicker_to_dead: float = DEFAULT_HAZARD_PARAMS["kappa_sicker_to_dead"],
) -> np.ndarray:
    """
    Build the base (n, 5, 5) transition matrix (identical formulas to
    `sfsa_data_generation.transition_matrix`), then smoothly boost:
        Sick  -> Sicker  by  extra_s = min(lambda_s * (sojourn/ref)^kappa_s, cap)
        Sicker-> Dead    by  extra_k = min(lambda_k * (sojourn/ref)^kappa_k, cap)
    where `ref` = 5 if abs_step < 20 else 3 (the base generator's own
    time-dependent reference run-length), and `sojourn` is each
    individual's number of consecutive steps already spent in their
    CURRENT state (so the Sick-row boost only "fires" for an individual
    actually in Sick, the Sicker-row boost only for one actually in
    Sicker -- selection happens naturally when the caller indexes this
    matrix by each individual's own `prev` state).

    The added mass is taken proportionally from that row's other live
    outgoing entries (never from the tiny background-death or censoring
    columns), so every row still sums to exactly 1.
    """
    T = transition_matrix(F, p).copy()
    n = T.shape[0]
    sojourn = np.asarray(sojourn, dtype=np.float64)
    ref = np.where(np.asarray(abs_step) < 20, 5.0, 3.0)

    # --- Sick -> Sicker ---
    extra_s = np.minimum(lambda_sick_to_sicker * (sojourn / ref) ** kappa_sick_to_sicker, 0.90)
    live_mass_s = T[:, SICK, HEALTHY] + T[:, SICK, SICK]
    extra_s = np.minimum(extra_s, 0.95 * live_mass_s)
    scale_s = np.divide(live_mass_s - extra_s, live_mass_s, out=np.ones_like(live_mass_s), where=live_mass_s > 1e-12)
    T[:, SICK, HEALTHY] *= scale_s
    T[:, SICK, SICK] *= scale_s
    T[:, SICK, SICKER] += extra_s

    # --- Sicker -> Dead ---
    extra_k = np.minimum(lambda_sicker_to_dead * (sojourn / ref) ** kappa_sicker_to_dead, 0.99)
    live_mass_k = T[:, SICKER, SICK] + T[:, SICKER, SICKER]
    extra_k = np.minimum(extra_k, 0.95 * live_mass_k)
    scale_k = np.divide(live_mass_k - extra_k, live_mass_k, out=np.ones_like(live_mass_k), where=live_mass_k > 1e-12)
    T[:, SICKER, SICK] *= scale_k
    T[:, SICKER, SICKER] *= scale_k
    T[:, SICKER, DEAD] += extra_k

    row_sums = T.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise ValueError(f"semi-Markov transition matrix rows do not sum to 1 for {len(bad)} individual(s)")
    if (T < -1e-9).any():
        raise ValueError("semi-Markov transition matrix has negative entries")
    return T


# ---------------------------------------------------------------------------
# Core vectorized simulation (mirrors _simulate_from / _simulate_states)
# ---------------------------------------------------------------------------
def _simulate_from_sm(
    current_state: np.ndarray, sojourn: np.ndarray, covariate: np.ndarray,
    scenario: Scenario, rng: np.random.Generator, start_step_1indexed: int, n_steps: int,
    **hazard_kwargs,
) -> np.ndarray:
    """
    Semi-Markov analogue of `sfsa_data_generation._simulate_from`: same
    stepping loop, except the transition matrix is rebuilt every step
    (since the Sick/Sicker hazard boost depends on each individual's
    CURRENT `sojourn`, which changes every step) via
    `sojourn_boosted_transition_matrix`, and there is no deterministic
    forced-death override -- the boosted Sicker->Dead probability itself
    carries that risk.
    """
    n = covariate.shape[0]
    F = scenario.F(covariate)
    prev = np.asarray(current_state, dtype=np.int8).copy()
    soj = np.asarray(sojourn, dtype=np.int32).copy()
    new_states = np.empty((n, n_steps), dtype=np.int8)

    for i in range(n_steps):
        abs_step = start_step_1indexed + 1 + i
        T = sojourn_boosted_transition_matrix(F, scenario.p, soj, abs_step, **hazard_kwargs)
        cumT = np.cumsum(T, axis=-1)

        u = rng.random(n)
        row_cum = cumT[np.arange(n), prev]
        nxt = np.clip((u[:, None] > row_cum).sum(axis=1), 0, N_STATES - 1)

        # Additional post-transition hard rule, active from absolute step 24.
        # The proposed second consecutive Sicker state becomes Dead.
        forced = (abs_step >= 24) & (prev == SICKER) & (nxt == SICKER)
        nxt = np.where(forced, DEAD, nxt)

        absorbed = np.isin(prev, (DEAD, CENSORED))
        nxt = np.where(absorbed, prev, nxt)

        same_state = (nxt == prev) & (~absorbed)
        soj = np.where(absorbed, soj, np.where(same_state, soj + 1, 1))

        new_states[:, i] = nxt
        prev = nxt

    return new_states


def _simulate_states_sm(
    covariate: np.ndarray, scenario: Scenario, rng: np.random.Generator, t_k: int = T_K,
    **hazard_kwargs,
) -> np.ndarray:
    """Semi-Markov analogue of `sfsa_data_generation._simulate_states`:
    same initial-state draw (t=1, sojourn=1 for everyone -- they have just
    entered that state), then delegates to `_simulate_from_sm`."""
    n = covariate.shape[0]
    init_state = rng.choice(np.array([HEALTHY, SICK, SICKER]), size=n, p=[2 / 3, 1 / 4, 1 / 12])
    init_sojourn = np.ones(n, dtype=np.int32)

    if t_k <= 1:
        return init_state.reshape(n, 1).astype(np.int8)[:, :t_k]

    rest = _simulate_from_sm(
        init_state, init_sojourn, covariate, scenario, rng,
        start_step_1indexed=1, n_steps=t_k - 1, **hazard_kwargs,
    )
    return np.concatenate([init_state.reshape(n, 1).astype(np.int8), rest], axis=1)


def generate_cohort_sm(scenario: Scenario, n: int, seed: int, t_k: int = T_K, **hazard_kwargs) -> dict:
    """Semi-Markov analogue of `sfsa_data_generation.generate_cohort`."""
    rng = make_rng(seed)
    covariate = sample_covariate(scenario, n, rng)
    states = _simulate_states_sm(covariate, scenario, rng, t_k, **hazard_kwargs)
    duration, event = _extract_duration_event(states, t_k)
    return {"scenario": scenario.name, "covariate": covariate, "states": states,
            "duration": duration, "event": event, "seed": seed}


def train_test_val_split_sm(cohort: dict, seed: int, fracs: tuple = (0.8, 0.1, 0.1)) -> dict:
    assert abs(sum(fracs) - 1.0) < 1e-9
    n = cohort["covariate"].shape[0]
    rng = make_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_test = int(round(fracs[1] * n))
    train_idx, test_idx, val_idx = idx[:n_train], idx[n_train:n_train + n_test], idx[n_train + n_test:]

    def subset(indices):
        return {"covariate": cohort["covariate"][indices], "states": cohort["states"][indices],
                "duration": cohort["duration"][indices], "event": cohort["event"][indices], "index": indices}
    return {"train": subset(train_idx), "test": subset(test_idx), "val": subset(val_idx)}


# ---------------------------------------------------------------------------
# Sojourn time from an OBSERVED prefix (generalizes
# sfsa_data_generation.consecutive_sicker_run_length to any state, not
# just Sicker -- Item 2's hazard depends on sojourn while Sick too)
# ---------------------------------------------------------------------------
def current_state_sojourn_time(states_prefix: np.ndarray) -> np.ndarray:
    """
    Given an (n, L) array of OBSERVED states ending at a landmark time
    (column L-1 is each individual's state AT the landmark), return each
    row's sojourn time: the number of consecutive trailing columns equal
    to that same landmark state (>= 1, since the landmark column itself
    always counts).
    """
    states_prefix = np.asarray(states_prefix)
    n, L = states_prefix.shape
    landmark_state = states_prefix[:, -1]
    run = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=bool)
    for col in range(L - 1, -1, -1):
        same = states_prefix[:, col] == landmark_state
        run = np.where(active & same, run + 1, run)
        active = active & same
    return run


# ---------------------------------------------------------------------------
# Landmark / dynamic-prediction ground truth (Item 1, applied to this
# semi-Markov chain -- see module docstring: this is the evaluation mode
# that actually exposes sojourn-time information)
# ---------------------------------------------------------------------------
def ground_truth_km_landmark_sm(
    scenario: Scenario, covariate_value: float, current_state: int, sojourn: int,
    landmark_step: int, seed: int, n_matched: int = 2_000, t_k: int = T_K,
    **hazard_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Semi-Markov analogue of `sfsa_data_generation.ground_truth_km_landmark`:
    continue the chain for `n_matched` individuals sharing this covariate
    value and exactly at (current_state, sojourn) as of absolute step
    `landmark_step`, and KM-estimate the resulting (duration, event) pairs.
    """
    if current_state in (DEAD, CENSORED):
        raise ValueError(
            f"ground_truth_km_landmark_sm requires a live current_state (Healthy/Sick/Sicker); "
            f"got {STATE_NAMES.get(current_state, current_state)!r} at landmark_step={landmark_step}."
        )
    eval_times = np.arange(0, t_k + 1)
    local_t_k = t_k - landmark_step
    if local_t_k <= 0:
        return eval_times, np.ones_like(eval_times, dtype=np.float64)

    rng = make_rng(seed)
    covariate = np.full(n_matched, float(covariate_value))
    init_state = np.full(n_matched, current_state, dtype=np.int8)
    init_sojourn = np.full(n_matched, sojourn, dtype=np.int32)

    new_states = _simulate_from_sm(
        init_state, init_sojourn, covariate, scenario, rng,
        start_step_1indexed=landmark_step, n_steps=local_t_k, **hazard_kwargs,
    )
    local_duration, event = _extract_duration_event(new_states, local_t_k)
    duration = landmark_step + local_duration

    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=event)
    sf = kmf.survival_function_at_times(eval_times).to_numpy()
    return eval_times, sf


def ground_truth_km_landmark_sm_for_validation_set(
    scenario: Scenario, val_covariates: np.ndarray, val_states_prefix: np.ndarray,
    landmark_step: int, base_seed: int, n_matched: int = 2_000, t_k: int = T_K,
    **hazard_kwargs,
) -> np.ndarray:
    """
    Semi-Markov analogue of
    `sfsa_data_generation.ground_truth_km_landmark_for_validation_set`.
    `val_states_prefix` is each individual's ACTUALLY OBSERVED state
    sequence up to and including the landmark (shape
    (len(val_covariates), landmark_step)); every row's last column must be
    a live state (filter out Dead/Censored individuals first).
    """
    val_states_prefix = np.asarray(val_states_prefix)
    if val_states_prefix.shape[1] != landmark_step:
        raise ValueError(
            f"val_states_prefix must have exactly landmark_step={landmark_step} columns "
            f"(got {val_states_prefix.shape[1]})."
        )
    current_state = val_states_prefix[:, -1]
    absorbed = np.isin(current_state, (DEAD, CENSORED))
    if absorbed.any():
        raise ValueError(
            f"{int(absorbed.sum())} of {len(current_state)} individuals are already "
            f"Dead/Censored at landmark_step={landmark_step} -- filter these out first."
        )
    sojourn = current_state_sojourn_time(val_states_prefix)

    seed_seq = np.random.SeedSequence(base_seed)
    child_seeds = seed_seq.spawn(len(val_covariates))
    curves = np.empty((len(val_covariates), t_k + 1), dtype=np.float64)
    for i, (cov, cur_state, soj, child_seed) in enumerate(
        zip(val_covariates, current_state, sojourn, child_seeds)
    ):
        _, sf = ground_truth_km_landmark_sm(
            scenario, cov, int(cur_state), int(soj), landmark_step,
            int(child_seed.generate_state(1)[0]), n_matched, t_k, **hazard_kwargs,
        )
        curves[i] = sf
    return curves


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
def validate_hazard_monotonicity(
    scenario: Scenario, covariate_value: float, abs_step: int, seed: int,
    n_matched: int = 20_000, max_sojourn: int = 8, **hazard_kwargs,
) -> dict:
    """
    Empirically confirms the semi-Markov hazard actually increases with
    sojourn time (replacing the base generator's hard-rule invariant check,
    which no longer applies since there is no deterministic override
    here): for each sojourn value 1..max_sojourn, start `n_matched`
    individuals at Sicker with that sojourn and take exactly one more step;
    the empirical P(-> Dead) should be non-decreasing in sojourn, and
    close to `lambda_sicker_to_dead` by the reference run-length.
    """
    rng = make_rng(seed)
    covariate = np.full(n_matched, float(covariate_value))
    F = scenario.F(covariate)
    p_dead = []
    for soj in range(1, max_sojourn + 1):
        init_state = np.full(n_matched, SICKER, dtype=np.int8)
        init_sojourn = np.full(n_matched, soj, dtype=np.int32)
        out = _simulate_from_sm(init_state, init_sojourn, covariate, scenario, rng,
                                 start_step_1indexed=abs_step - 1, n_steps=1, **hazard_kwargs)
        p_dead.append(float((out[:, 0] == DEAD).mean()))
    return {"sojourn_values": list(range(1, max_sojourn + 1)), "p_dead": p_dead}
