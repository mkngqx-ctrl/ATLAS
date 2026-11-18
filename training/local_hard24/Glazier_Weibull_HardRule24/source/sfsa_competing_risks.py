"""
sfsa_competing_risks.py

Item 3 from glazier_differentiation_designs.md: a competing-risks extension
of the SFSA / Markov-chain "sick and sicker" generator in
sfsa_data_generation.py.

Design (see the design doc for the full rationale; summary below):

  - "Dead" is split into two cause-specific absorbing states:
        DEAD_DISEASE -- death attributable to the modeled disease process
        DEAD_OTHER    -- background/competing mortality, unrelated to it
  - The *total* per-step death probability out of Healthy/Sick/Sicker is
    left EXACTLY as in the base scenario (Eq. 19's 0.001 / 0.002 / 0.004
    terms) -- only how that probability mass is split between the two
    causes changes. This means the base scenario's calibration (dropout
    rate, mean/median time-to-death, total censoring rate) is preserved
    bit-for-bit; competing risks is a strict refinement, not a different
    generator.
  - The split is state- and covariate-dependent (`default_cause_a_share`):
    death while Healthy is mostly attributed to the "other" cause (a
    healthy individual's death is unlikely to be the modeled disease);
    death while Sicker is mostly attributed to "disease"; death while Sick
    is split roughly evenly. A mild covariate modulation (via a z-scored
    sigmoid, scenario-agnostic) pushes individuals further into the
    covariate's upper tail slightly more toward the disease cause.
  - The existing forced-termination rule (5 consecutive Sicker states
    before step 20, or 3 at/after step 20, forces death) is, by
    construction, always attributed to DEAD_DISEASE -- it explicitly
    represents disease progression, not background mortality, so there is
    no "share" to apply here.

Why this is the right lever for the Glazier-vs-baselines comparison (see
glazier_differentiation_designs.md, item 3): Glazier's next-state
prediction is already a categorical softmax over states, so adding one
more absorbing state is a near one-line change (`n_states += 1`) with no
loss-function change at all. Every other model in the 8-model comparison
needs real surgery to handle cause-specific outcomes:
  - CPH needs separate cause-specific Cox models (treating the other
    cause's events as censoring in each submodel) or a from-scratch
    competing-risks Cox/Fine-Gray extension.
  - DRSA, Dynamic-DeepHit, and SurvTransformer each have a single
    softmax-over-T_K-bins head; competing risks needs that turned into a
    softmax over (bins x causes), with a correspondingly restructured
    ranking/NLL loss.
  - WTTE-RNN's single-Weibull-per-individual output has no natural
    extension to multiple causes at all -- it would need a genuine
    redesign (e.g. a finite mixture of cause-specific Weibulls, or a
    competing-Weibull hazard decomposition).

This module only builds the DATA SIDE of that comparison (generator +
ground truth); adapting each of the five non-Glazier model architectures
is a separate, per-model design decision left for the modeling side of the
comparison.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sfsa_data_generation import (
    HEALTHY, SICK, SICKER, N_STATES as _N_STATES_BASE, T_K,
    Scenario, RELU_SCENARIO, WEIBULL_SCENARIO,
    make_rng, sample_covariate,
)

# ---------------------------------------------------------------------------
# Competing-risks state encoding
# ---------------------------------------------------------------------------
# HEALTHY/SICK/SICKER keep their base-module indices (0/1/2) so that any
# code comparing/plotting the *live* states is unaffected. DEAD becomes two
# cause-specific absorbing states, and CENSORED moves to the last index.
DEAD_DISEASE, DEAD_OTHER, CENSORED = 3, 4, 5
N_STATES_CR = 6
STATE_NAMES_CR = {
    HEALTHY: "Healthy", SICK: "Sick", SICKER: "Sicker",
    DEAD_DISEASE: "Dead (disease)", DEAD_OTHER: "Dead (other cause)",
    CENSORED: "Censored",
}
N_CAUSES = 2  # 1 = disease, 2 = other -- see `cause` arrays below


# ---------------------------------------------------------------------------
# Cause-split function
# ---------------------------------------------------------------------------
def default_cause_a_share(
    covariate: np.ndarray, current_state: np.ndarray, scenario: Scenario
) -> np.ndarray:
    """
    Fraction (in [0, 1]) of a row's total death-probability mass to
    attribute to the DISEASE cause (the rest goes to the OTHER cause).

    State-dependent base rate: death while Healthy is mostly background
    mortality (low disease share); death while Sicker is mostly disease
    progression (high disease share); Sick is roughly split down the
    middle. A small covariate modulation (z-scored through a sigmoid, so
    it is bounded and scenario-agnostic regardless of whether the raw F(.)
    happens to exceed [0, 1], which it can for WEIBULL_SCENARIO) nudges
    individuals further into the covariate's upper tail slightly more
    toward the disease cause.
    """
    current_state = np.asarray(current_state)
    base = np.select(
        [current_state == HEALTHY, current_state == SICK, current_state == SICKER],
        [0.10, 0.50, 0.85],
    )
    z = (np.asarray(covariate, dtype=np.float64) - scenario.covariate_mean) / scenario.covariate_sd
    modulation = 0.10 * (1.0 / (1.0 + np.exp(-z)) - 0.5)  # in (-0.05, +0.05)
    return np.clip(base + modulation, 0.0, 1.0)


CauseShareFn = Callable[[np.ndarray, np.ndarray, Scenario], np.ndarray]


# ---------------------------------------------------------------------------
# Competing-risks transition matrix
# ---------------------------------------------------------------------------
def competing_risks_transition_matrix(
    covariate: np.ndarray, scenario: Scenario, cause_a_share_fn: CauseShareFn = default_cause_a_share,
) -> np.ndarray:
    """
    Build the (n, 6, 6) competing-risks transition matrix. Identical to
    `sfsa_data_generation.transition_matrix` except the single Dead column
    (index 3 there) is split into DEAD_DISEASE (3) and DEAD_OTHER (4), and
    CENSORED moves from index 4 to index 5. Every live-state-to-live-state
    entry, and every row's *total* probability of dying, is numerically
    identical to the base 5-state matrix -- only the death-cause split is
    new. See module docstring.
    """
    F = scenario.F(np.asarray(covariate, dtype=np.float64))
    F = np.atleast_1d(F)
    covariate = np.atleast_1d(np.asarray(covariate, dtype=np.float64))
    p = scenario.p
    n = F.shape[0]
    T = np.zeros((n, N_STATES_CR, N_STATES_CR), dtype=np.float64)

    dead_mass = {HEALTHY: 0.001, SICK: 0.002, SICKER: 0.004}

    # Row 0: Healthy
    share_h = cause_a_share_fn(covariate, np.full(n, HEALTHY), scenario)
    T[:, 0, 0] = F - (1 / 3) * p - 0.001
    T[:, 0, 1] = 0.5 * (1 - F) - (1 / 3) * p
    T[:, 0, 2] = 0.5 * (1 - F) - (1 / 3) * p
    T[:, 0, DEAD_DISEASE] = dead_mass[HEALTHY] * share_h
    T[:, 0, DEAD_OTHER] = dead_mass[HEALTHY] * (1 - share_h)
    T[:, 0, CENSORED] = p

    # Row 1: Sick
    share_s = cause_a_share_fn(covariate, np.full(n, SICK), scenario)
    T[:, 1, 0] = (2 / 3) * F - (1 / 3) * p
    T[:, 1, 1] = (1 / 3) - (1 / 3) * F - (1 / 3) * p - 0.002
    T[:, 1, 2] = (2 / 3) - (1 / 3) * F - (1 / 3) * p
    T[:, 1, DEAD_DISEASE] = dead_mass[SICK] * share_s
    T[:, 1, DEAD_OTHER] = dead_mass[SICK] * (1 - share_s)
    T[:, 1, CENSORED] = p

    # Row 2: Sicker
    share_k = cause_a_share_fn(covariate, np.full(n, SICKER), scenario)
    T[:, 2, 0] = 0.0
    T[:, 2, 1] = (1 / 8) * F - (2 / 3) * p + 0.306
    T[:, 2, 2] = 0.69 - (1 / 8) * F - (1 / 3) * p
    T[:, 2, DEAD_DISEASE] = dead_mass[SICKER] * share_k
    T[:, 2, DEAD_OTHER] = dead_mass[SICKER] * (1 - share_k)
    T[:, 2, CENSORED] = p

    # Absorbing states
    T[:, DEAD_DISEASE, DEAD_DISEASE] = 1.0
    T[:, DEAD_OTHER, DEAD_OTHER] = 1.0
    T[:, CENSORED, CENSORED] = 1.0

    row_sums = T.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise ValueError(
            f"Competing-risks transition matrix rows do not sum to 1 for {len(bad)} "
            f"individual(s) (first offender index {bad[0]}, F={F[bad[0]]:.4f}, p={p})."
        )
    if (T < -1e-9).any():
        raise ValueError(
            "Competing-risks transition matrix has negative entries -- F or p is out "
            "of the valid range for this scenario."
        )
    return T


# ---------------------------------------------------------------------------
# Core vectorized simulation (mirrors _simulate_from / _simulate_states)
# ---------------------------------------------------------------------------
def _simulate_from_cr(
    current_state: np.ndarray,
    consecutive_sicker: np.ndarray,
    covariate: np.ndarray,
    scenario: Scenario,
    rng: np.random.Generator,
    start_step_1indexed: int,
    n_steps: int,
    cause_a_share_fn: CauseShareFn = default_cause_a_share,
) -> np.ndarray:
    """
    Competing-risks analogue of `sfsa_data_generation._simulate_from`.
    Identical per-step Markov-transition and forced-termination logic,
    except (a) the transition matrix is the 6-state competing-risks one,
    and (b) a forced death (from the consecutive-Sicker run-length rule)
    is always routed to DEAD_DISEASE -- it represents disease progression
    by construction, not background mortality, so there is no cause-share
    to apply there.
    """
    n = covariate.shape[0]
    T = competing_risks_transition_matrix(covariate, scenario, cause_a_share_fn)
    cumT = np.cumsum(T, axis=-1)

    new_states = np.empty((n, n_steps), dtype=np.int8)
    prev = np.asarray(current_state, dtype=np.int8).copy()
    consecutive = np.asarray(consecutive_sicker, dtype=np.int32).copy()

    for i in range(n_steps):
        abs_step = start_step_1indexed + 1 + i
        u = rng.random(n)
        row_cum = cumT[np.arange(n), prev]
        nxt = np.clip((u[:, None] > row_cum).sum(axis=1), 0, N_STATES_CR - 1)

        absorbed = np.isin(prev, (DEAD_DISEASE, DEAD_OTHER, CENSORED))
        nxt = np.where(absorbed, prev, nxt)

        threshold = 5 if abs_step < 20 else 3
        is_sicker = nxt == SICKER
        consecutive = np.where(is_sicker, consecutive + 1, 0)

        force_death = (~absorbed) & (consecutive >= threshold)
        nxt = np.where(force_death, DEAD_DISEASE, nxt)  # forced death -> always disease cause
        consecutive = np.where(force_death, 0, consecutive)

        new_states[:, i] = nxt
        prev = nxt

    return new_states


def _simulate_states_cr(
    covariate: np.ndarray, scenario: Scenario, rng: np.random.Generator, t_k: int = T_K,
    cause_a_share_fn: CauseShareFn = default_cause_a_share,
) -> np.ndarray:
    """Competing-risks analogue of `sfsa_data_generation._simulate_states`.
    Same initial-state draw (t=1), then delegates to `_simulate_from_cr`."""
    n = covariate.shape[0]
    init_state = rng.choice(np.array([HEALTHY, SICK, SICKER]), size=n, p=[2 / 3, 1 / 4, 1 / 12])
    init_consecutive_sicker = (init_state == SICKER).astype(np.int32)

    if t_k <= 1:
        return init_state.reshape(n, 1).astype(np.int8)[:, :t_k]

    rest = _simulate_from_cr(
        init_state, init_consecutive_sicker, covariate, scenario, rng,
        start_step_1indexed=1, n_steps=t_k - 1, cause_a_share_fn=cause_a_share_fn,
    )
    return np.concatenate([init_state.reshape(n, 1).astype(np.int8), rest], axis=1)


def _extract_duration_cause(states: np.ndarray, t_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (n, t_k) competing-risks state trajectories into (duration,
    cause) pairs: duration = 1-indexed time of the first absorbing event
    (either death cause, or censoring) or t_k if none occurs; cause = 0 if
    no death was observed by t_k (administrative censoring or random
    censoring), 1 if DEAD_DISEASE, 2 if DEAD_OTHER. This is the standard
    (duration, event_type) competing-risks format, with cause=0 meaning
    "right-censored" exactly as event=0 does in the single-risk case.
    """
    n = states.shape[0]
    dead_a = states == DEAD_DISEASE
    dead_b = states == DEAD_OTHER
    censored = states == CENSORED

    has_a, has_b, has_c = dead_a.any(1), dead_b.any(1), censored.any(1)
    time_a = np.where(has_a, dead_a.argmax(1) + 1, t_k + 1)
    time_b = np.where(has_b, dead_b.argmax(1) + 1, t_k + 1)
    time_c = np.where(has_c, censored.argmax(1) + 1, t_k + 1)

    duration = np.minimum(np.minimum(np.minimum(time_a, time_b), time_c), t_k)

    cause = np.zeros(n, dtype=np.int8)
    a_first = has_a & (time_a <= time_b) & (time_a <= time_c) & (time_a <= t_k)
    b_first = (~a_first) & has_b & (time_b <= time_c) & (time_b <= t_k)
    cause[a_first] = 1
    cause[b_first] = 2
    return duration, cause


def generate_cohort_cr(
    scenario: Scenario, n: int, seed: int, t_k: int = T_K,
    cause_a_share_fn: CauseShareFn = default_cause_a_share,
) -> dict:
    """Competing-risks analogue of `sfsa_data_generation.generate_cohort`."""
    rng = make_rng(seed)
    covariate = sample_covariate(scenario, n, rng)
    states = _simulate_states_cr(covariate, scenario, rng, t_k, cause_a_share_fn)
    duration, cause = _extract_duration_cause(states, t_k)
    return {
        "scenario": scenario.name, "covariate": covariate, "states": states,
        "duration": duration, "cause": cause, "seed": seed,
    }


def train_test_val_split_cr(cohort: dict, seed: int, fracs: tuple = (0.8, 0.1, 0.1)) -> dict:
    assert abs(sum(fracs) - 1.0) < 1e-9
    n = cohort["covariate"].shape[0]
    rng = make_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_test = int(round(fracs[1] * n))
    train_idx, test_idx, val_idx = idx[:n_train], idx[n_train:n_train + n_test], idx[n_train + n_test:]

    def subset(indices):
        return {
            "covariate": cohort["covariate"][indices], "states": cohort["states"][indices],
            "duration": cohort["duration"][indices], "cause": cohort["cause"][indices],
            "index": indices,
        }
    return {"train": subset(train_idx), "test": subset(test_idx), "val": subset(val_idx)}


# ---------------------------------------------------------------------------
# Ground truth: Aalen-Johansen cumulative incidence functions
# ---------------------------------------------------------------------------
def aalen_johansen_cif(
    duration: np.ndarray, cause: np.ndarray, t_k: int = T_K, n_causes: int = N_CAUSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nonparametric Aalen-Johansen estimator for the overall (all-cause)
    survival function and the cause-specific cumulative incidence
    functions, under right censoring (cause == 0).

    At each observed event time tau (any cause):
        S(tau)    = S(tau-1) * (1 - d_total(tau) / n_at_risk(tau))
        CIF_k(tau) = CIF_k(tau-1) + S(tau-1) * (d_k(tau) / n_at_risk(tau))
    using the survival value from JUST BEFORE tau on the right-hand side
    of both updates (the defining property of Aalen-Johansen: CIF
    increments use the pre-jump survival probability). Censored
    observations only ever reduce `n_at_risk` at later times, which is the
    usual nonparametric treatment of independent censoring; no NumPy RNG
    is involved here, this is a deterministic estimator applied to
    already-simulated (duration, cause) pairs.

    Returns (eval_times, overall_survival, cif) where `cif` has shape
    (n_causes, t_k + 1): cif[k-1] is cause k's CIF curve, and
    overall_survival + cif.sum(axis=0) == 1 at every eval time (up to
    floating-point error) whenever every individual is either still alive,
    dead, or censored by then -- i.e. always, by construction.
    """
    eval_times = np.arange(0, t_k + 1)
    duration = np.asarray(duration)
    cause = np.asarray(cause)

    surv_curve = np.ones(t_k + 1, dtype=np.float64)
    cif_curves = np.zeros((n_causes + 1, t_k + 1), dtype=np.float64)  # index 0 unused

    event_times = np.unique(duration[cause > 0])
    S = 1.0
    cif_running = np.zeros(n_causes + 1, dtype=np.float64)
    for tau in event_times:
        tau = int(tau)
        n_at_risk = int((duration >= tau).sum())
        if n_at_risk == 0:
            continue
        d_total = int(((duration == tau) & (cause > 0)).sum())
        for k in range(1, n_causes + 1):
            d_k = int(((duration == tau) & (cause == k)).sum())
            if d_k:
                cif_running[k] += S * (d_k / n_at_risk)
        S_new = S * (1 - d_total / n_at_risk)
        if tau <= t_k:
            surv_curve[tau:] = S_new
            for k in range(1, n_causes + 1):
                cif_curves[k, tau:] = cif_running[k]
        S = S_new

    return eval_times, surv_curve, cif_curves[1:]


def ground_truth_cif(
    scenario: Scenario, covariate_value: float, seed: int, n_matched: int = 10_000, t_k: int = T_K,
    cause_a_share_fn: CauseShareFn = default_cause_a_share,
) -> dict:
    """
    Competing-risks analogue of `sfsa_data_generation.ground_truth_km`:
    simulate `n_matched` individuals sharing `covariate_value`, then
    Aalen-Johansen-estimate the overall survival curve and both
    cause-specific CIF curves as the ground truth.
    """
    rng = make_rng(seed)
    covariate = np.full(n_matched, float(covariate_value))
    states = _simulate_states_cr(covariate, scenario, rng, t_k, cause_a_share_fn)
    duration, cause = _extract_duration_cause(states, t_k)
    eval_times, sf, cif = aalen_johansen_cif(duration, cause, t_k)
    return {"eval_times": eval_times, "survival": sf, "cif_disease": cif[0], "cif_other": cif[1]}


def ground_truth_cif_for_validation_set(
    scenario: Scenario, val_covariates: np.ndarray, base_seed: int,
    n_matched: int = 10_000, t_k: int = T_K, cause_a_share_fn: CauseShareFn = default_cause_a_share,
) -> dict:
    """Per-individual ground truth over a validation set, one independently
    seeded simulated-cohort match per individual (mirrors
    `ground_truth_km_for_validation_set`)."""
    seed_seq = np.random.SeedSequence(base_seed)
    child_seeds = seed_seq.spawn(len(val_covariates))
    sf = np.empty((len(val_covariates), t_k + 1))
    cif_disease = np.empty((len(val_covariates), t_k + 1))
    cif_other = np.empty((len(val_covariates), t_k + 1))
    for i, (cov, child_seed) in enumerate(zip(val_covariates, child_seeds)):
        out = ground_truth_cif(scenario, cov, int(child_seed.generate_state(1)[0]), n_matched, t_k, cause_a_share_fn)
        sf[i], cif_disease[i], cif_other[i] = out["survival"], out["cif_disease"], out["cif_other"]
    return {"survival": sf, "cif_disease": cif_disease, "cif_other": cif_other}


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
def validate_generation_invariants_cr(states: np.ndarray, t_k: int = T_K) -> None:
    """
    Asserts, over a batch of generated competing-risks trajectories, that:
      1. DEAD_DISEASE, DEAD_OTHER, and CENSORED are all absorbing.
      2. No individual has a run of Sicker states reaching the
         time-dependent threshold without the run being terminated by a
         forced transition -- and that forced transition is ALWAYS to
         DEAD_DISEASE specifically (never DEAD_OTHER).
    """
    n, T = states.shape
    for absorb_state in (DEAD_DISEASE, DEAD_OTHER, CENSORED):
        hit = states == absorb_state
        any_hit = hit.any(axis=1)
        first_hit = np.where(any_hit, hit.argmax(axis=1), T)
        for t in range(T):
            must_be_absorbed = t >= first_hit
            if not np.all(states[must_be_absorbed, t] == absorb_state):
                raise AssertionError(f"Absorbing-state violation for state {absorb_state} at t={t + 1}")

    for i in range(n):
        run = 0
        for t in range(T):
            s = states[i, t]
            threshold = 5 if (t + 1) < 20 else 3
            if s == SICKER:
                run += 1
                if run >= threshold:
                    raise AssertionError(
                        f"Individual {i}: {run} consecutive Sicker states at t={t + 1} "
                        f"(threshold {threshold}) without forced death."
                    )
            elif s == DEAD_DISEASE:
                if run >= threshold:
                    pass  # correctly forced, and correctly disease-cause
                break
            elif s == DEAD_OTHER:
                if run >= threshold:
                    raise AssertionError(
                        f"Individual {i}: a forced termination (run={run} >= {threshold}) at "
                        f"t={t + 1} was attributed to DEAD_OTHER, not DEAD_DISEASE."
                    )
                break
            elif s == CENSORED:
                break
            else:
                run = 0
