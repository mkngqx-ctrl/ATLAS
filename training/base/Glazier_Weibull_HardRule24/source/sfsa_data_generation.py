"""
sfsa_data_generation.py

Reproduces the Stochastic Finite State Automata (SFSA) / Markov-chain "sick and
sicker" data-generation process from:

    "Deep Learning Based Survival Analysis for Sequential Data"
    (Section 3: Methods -> 3.1 Finite State Automata and Markov Data Generation
    Model, 3.2 Simulation, 3.2.1 ReLU Data Generation Model, 3.2.2 Weibull Data
    Generation Model)

Stack: NumPy for the vectorized Markov-chain simulation, PyTorch only for
global seeding (so it stays deterministic once you plug in the downstream
DL survival models), and lifelines for the Kaplan-Meier ground-truth curves.

Everything here is seeded end-to-end: call `set_seed(GLOBAL_SEED)` once, then
every cohort/split/ground-truth call additionally takes its own explicit
`seed` argument so that individual pieces are independently reproducible
regardless of call order (see `make_rng`).

Known open question (paper does not specify this): the covariate is drawn
from a "truncated normal distribution" for each scenario, but no mean/SD is
given -- only the plotting ranges in the paper's Figure 2 (age ~ [30, 120],
calcium ~ [7, 13]). RELU_SCENARIO / WEIBULL_SCENARIO below ship with a
starting guess for mean/SD; `calibrate_covariate_params` does a local search
to bring the resulting censoring rates / mean time-to-death close to the
paper's reported validation numbers. Treat the calibrated values as a
best-effort match, not a guarantee of recovering the authors' true
parameters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
from scipy.stats import truncnorm
from lifelines import KaplanMeierFitter

# ---------------------------------------------------------------------------
# State encoding (paper Section 3.1 / Figure 1)
HEALTHY, SICK, SICKER, DEAD, CENSORED = 0, 1, 2, 3, 4
STATE_NAMES = {HEALTHY: "Healthy", SICK: "Sick", SICKER: "Sicker",
               DEAD: "Dead", CENSORED: "Censored"}
N_STATES = 5
T_K = 30  # time steps per individual (paper Section 3.2)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fix every global RNG (python `random`, NumPy legacy global state, and
    PyTorch) so the whole pipeline -- including downstream model training --
    is reproducible. Call this once at the top of your script/notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int) -> np.random.Generator:
    """A dedicated NumPy Generator seeded from `seed`, independent of the
    global NumPy random state. All data-generation functions below use one
    of these (never the global `np.random` state) so that a given `seed`
    always reproduces the same cohort regardless of what else has run.
    """
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Transition matrix (Eq. 19)
# ---------------------------------------------------------------------------
def transition_matrix(F: np.ndarray, p: float) -> np.ndarray:
    """
    Build the (N, 5, 5) transition matrix T(F, p) from Eq. 19.

    F : array of shape (N,), one covariate-derived value per individual.
    p : scalar per-time-step random-censoring probability for the scenario.

    Row/column order: Healthy(0), Sick(1), Sicker(2), Dead(3), Censored(4).
    """
    F = np.atleast_1d(np.asarray(F, dtype=np.float64))
    n = F.shape[0]
    T = np.zeros((n, N_STATES, N_STATES), dtype=np.float64)

    # Row 0: Healthy
    T[:, 0, 0] = F - (1 / 3) * p - 0.001
    T[:, 0, 1] = 0.5 * (1 - F) - (1 / 3) * p
    T[:, 0, 2] = 0.5 * (1 - F) - (1 / 3) * p
    T[:, 0, 3] = 0.001
    T[:, 0, 4] = p

    # Row 1: Sick
    T[:, 1, 0] = (2 / 3) * F - (1 / 3) * p
    T[:, 1, 1] = (1 / 3) - (1 / 3) * F - (1 / 3) * p - 0.002
    T[:, 1, 2] = (2 / 3) - (1 / 3) * F - (1 / 3) * p
    T[:, 1, 3] = 0.002
    T[:, 1, 4] = p

    # Row 2: Sicker
    T[:, 2, 0] = 0.0
    T[:, 2, 1] = (1 / 8) * F - (2 / 3) * p + 0.306
    T[:, 2, 2] = 0.69 - (1 / 8) * F - (1 / 3) * p
    T[:, 2, 3] = 0.004
    T[:, 2, 4] = p

    # Row 3: Dead (absorbing)
    T[:, 3, 3] = 1.0

    # Row 4: Censored (absorbing)
    T[:, 4, 4] = 1.0

    row_sums = T.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise ValueError(
            f"Transition matrix rows do not sum to 1 for {len(bad)} individual(s) "
            f"(first offender index {bad[0]}, F={F[bad[0]]:.4f}, p={p}). "
            "F is likely outside the range that keeps all entries valid probabilities."
        )
    if (T < -1e-9).any():
        raise ValueError(
            "Transition matrix has negative entries -- F or p is out of the "
            "valid range for this scenario."
        )
    return T


# ---------------------------------------------------------------------------
# Scenario definitions (paper Section 3.2.1 / 3.2.2)
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    covariate_name: str
    F: Callable[[np.ndarray], np.ndarray]
    p: float                       # per-time-step censoring probability
    covariate_bounds: tuple        # (low, high) truncation bounds
    covariate_mean: float          # ASSUMPTION -- see module docstring
    covariate_sd: float            # ASSUMPTION -- see module docstring
    # paper's reported validation-set targets, used for sanity-checking /
    # calibrating the covariate distribution:
    target_dropout_censoring_rate: float
    target_total_censoring_rate: float
    target_mean_ttd: float
    target_median_ttd: float


def relu_F(age: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=np.float64)
    return np.where(age <= 70, 0.75, -0.012 * age + 1.59)


def weibull_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (beta / alpha) * (x / alpha) ** (beta - 1) * np.exp(-((x / alpha) ** beta))


def weibull_F(calcium: np.ndarray) -> np.ndarray:
    return 3 * weibull_pdf(calcium, alpha=10, beta=5) + 0.2


RELU_SCENARIO = Scenario(
    name="ReLU", covariate_name="age", F=relu_F, p=0.0011,
    covariate_bounds=(30.0, 120.0), covariate_mean=70.0, covariate_sd=20.0,
    target_dropout_censoring_rate=0.022, target_total_censoring_rate=0.291,
    target_mean_ttd=17.1, target_median_ttd=20.0,
)

WEIBULL_SCENARIO = Scenario(
    name="Weibull", covariate_name="calcium", F=weibull_F, p=0.003,
    covariate_bounds=(7.0, 13.0), covariate_mean=10.0, covariate_sd=1.5,
    target_dropout_censoring_rate=0.055, target_total_censoring_rate=0.258,
    target_mean_ttd=16.6, target_median_ttd=18.0,
)


def sample_covariate(scenario: Scenario, n: int, rng: np.random.Generator) -> np.ndarray:
    low, high = scenario.covariate_bounds
    a = (low - scenario.covariate_mean) / scenario.covariate_sd
    b = (high - scenario.covariate_mean) / scenario.covariate_sd
    return truncnorm.rvs(
        a, b, loc=scenario.covariate_mean, scale=scenario.covariate_sd,
        size=n, random_state=rng,
    )


# ---------------------------------------------------------------------------
# Core vectorized Markov-chain simulation
# ---------------------------------------------------------------------------
def _simulate_states(
    covariate: np.ndarray, scenario: Scenario, rng: np.random.Generator, t_k: int = T_K
) -> np.ndarray:
    """
    Simulate the full (n, t_k) state trajectory for every individual, given
    their (already-sampled) covariate values. Implements:
      - the initial-state draw (t=1) from Multinomial(S=(0,1,2), p=(2/3,1/4,1/12)),
      - subsequent draws from each individual's own transition-matrix row,
      - Dead/Censored absorption,
      - the time-dependent forced-termination rule: 5 consecutive Sicker
        states before time step 20 forces death; 3 consecutive Sicker states
        at/after time step 20 forces death.
    """
    n = covariate.shape[0]
    F = scenario.F(covariate)
    T = transition_matrix(F, scenario.p)
    cumT = np.cumsum(T, axis=-1)  # (n, 5, 5)

    states = np.empty((n, t_k), dtype=np.int8)

    # t = 1 (index 0): initial state drawn only from {Healthy, Sick, Sicker}
    states[:, 0] = rng.choice(
        np.array([HEALTHY, SICK, SICKER]), size=n, p=[2 / 3, 1 / 4, 1 / 12]
    )
    consecutive_sicker = (states[:, 0] == SICKER).astype(np.int32)

    for t in range(1, t_k):
        prev = states[:, t - 1]
        u = rng.random(n)
        row_cum = cumT[np.arange(n), prev]                    # (n, 5)
        nxt = np.clip((u[:, None] > row_cum).sum(axis=1), 0, N_STATES - 1)

        absorbed = np.isin(prev, (DEAD, CENSORED))
        nxt = np.where(absorbed, prev, nxt)

        step_1indexed = t + 1
        threshold = 5 if step_1indexed < 20 else 3
        is_sicker = nxt == SICKER
        consecutive_sicker = np.where(is_sicker, consecutive_sicker + 1, 0)

        force_death = (~absorbed) & (consecutive_sicker >= threshold)
        nxt = np.where(force_death, DEAD, nxt)
        consecutive_sicker = np.where(force_death, 0, consecutive_sicker)

        states[:, t] = nxt

    return states


def _extract_duration_event(states: np.ndarray, t_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert (n, t_k) state trajectories into (duration, event) pairs:
    duration = 1-indexed time of death, or of censoring/drop-out, or t_k if
    neither happens (administrative censoring at end of follow-up);
    event = 1 if death was observed, else 0.
    """
    n = states.shape[0]
    died = states == DEAD
    censored = states == CENSORED
    has_died = died.any(axis=1)
    has_censored = censored.any(axis=1)

    death_time = np.where(has_died, died.argmax(axis=1) + 1, t_k + 1)
    censor_time = np.where(has_censored, censored.argmax(axis=1) + 1, t_k + 1)

    duration = np.minimum(np.minimum(death_time, censor_time), t_k)
    event = (has_died & (death_time <= censor_time) & (death_time <= t_k)).astype(np.int8)
    return duration, event


def generate_cohort(scenario: Scenario, n: int, seed: int, t_k: int = T_K) -> dict:
    """Generate `n` individuals under `scenario`, deterministically from `seed`."""
    rng = make_rng(seed)
    covariate = sample_covariate(scenario, n, rng)
    states = _simulate_states(covariate, scenario, rng, t_k)
    duration, event = _extract_duration_event(states, t_k)
    return {
        "scenario": scenario.name,
        "covariate": covariate,
        "states": states,
        "duration": duration,
        "event": event,
        "seed": seed,
    }


def train_test_val_split(cohort: dict, seed: int, fracs: tuple = (0.8, 0.1, 0.1)) -> dict:
    """80/10/10 train/test/validation split (paper Section 3.2), deterministic
    given `seed`."""
    assert abs(sum(fracs) - 1.0) < 1e-9
    n = cohort["covariate"].shape[0]
    rng = make_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_test = int(round(fracs[1] * n))
    train_idx, test_idx, val_idx = idx[:n_train], idx[n_train:n_train + n_test], idx[n_train + n_test:]

    def subset(indices):
        return {
            "covariate": cohort["covariate"][indices],
            "states": cohort["states"][indices],
            "duration": cohort["duration"][indices],
            "event": cohort["event"][indices],
            "index": indices,
        }

    return {"train": subset(train_idx), "test": subset(test_idx), "val": subset(val_idx)}


# ---------------------------------------------------------------------------
# Summary statistics / calibration against the paper's reported numbers
# ---------------------------------------------------------------------------
def cohort_summary(cohort: dict) -> dict:
    dropped_out = (cohort["states"] == CENSORED).any(axis=1)
    event, duration = cohort["event"], cohort["duration"]
    died = event == 1
    return {
        "n": int(len(event)),
        "dropout_censoring_rate": float(dropped_out.mean()),
        "total_censoring_rate": float((event == 0).mean()),
        "mean_ttd": float(duration[died].mean()) if died.any() else float("nan"),
        "median_ttd": float(np.median(duration[died])) if died.any() else float("nan"),
    }


def calibrate_covariate_params(
    scenario: Scenario, seed: int, n_calib: int = 20_000, max_iter: int = 25
) -> dict:
    """
    Best-effort local search over (mean, sd) of the truncated-normal covariate
    distribution to match the paper's reported dropout-censoring rate and
    mean time-to-death for this scenario (see module docstring -- the paper
    never states these parameters). Returns the calibrated values; does not
    mutate the module-level scenario constants.

    NOTE: this is an under-determined problem -- two summary statistics
    cannot uniquely pin down two distribution parameters, so different
    starting points can converge to different (mean, sd) pairs that all
    reproduce the target statistics about equally well. `mean` is bounded to
    the scenario's covariate range and `sd` to a plausible range to keep the
    result from drifting to a degenerate corner solution; still treat the
    output as "a" plausible calibration, not "the" one the paper used.
    """
    from scipy.optimize import minimize

    low, high = scenario.covariate_bounds
    span = high - low

    def loss(params):
        mean, log_sd = params
        sd = float(np.exp(log_sd))
        trial = replace(scenario, covariate_mean=float(mean), covariate_sd=sd)
        cohort = generate_cohort(trial, n_calib, seed)
        s = cohort_summary(cohort)
        err_dropout = s["dropout_censoring_rate"] - scenario.target_dropout_censoring_rate
        err_ttd = (s["mean_ttd"] - scenario.target_mean_ttd) / scenario.target_mean_ttd
        return err_dropout ** 2 + err_ttd ** 2

    x0 = [scenario.covariate_mean, np.log(scenario.covariate_sd)]
    bounds = [(low, high), (np.log(span / 50), np.log(span))]
    res = minimize(loss, x0, method="Nelder-Mead", bounds=bounds,
                    options={"maxiter": max_iter, "xatol": 0.5, "fatol": 1e-6})
    mean, log_sd = res.x
    return {"covariate_mean": float(mean), "covariate_sd": float(np.exp(log_sd)),
            "final_loss": float(res.fun)}


# ---------------------------------------------------------------------------
# Ground-truth Kaplan-Meier curves (paper Section 3.2, end / Section 4)
# ---------------------------------------------------------------------------
def ground_truth_km(
    scenario: Scenario, covariate_value: float, seed: int,
    n_matched: int = 10_000, t_k: int = T_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ground truth for a single covariate value: simulate `n_matched`
    individuals who all share that exact covariate value, then take the
    Kaplan-Meier estimate of their (duration, event) pairs as the ground
    truth survival curve (paper: "we estimate the ground truth of survival
    probabilities over time for each individual by generating 10,000 similar
    observations with the same covariate and getting the KM estimate").

    Returns (eval_times, survival_probability) for t = 0..t_k.
    """
    rng = make_rng(seed)
    covariate = np.full(n_matched, float(covariate_value))
    states = _simulate_states(covariate, scenario, rng, t_k)
    duration, event = _extract_duration_event(states, t_k)

    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=event)
    eval_times = np.arange(0, t_k + 1)
    sf = kmf.survival_function_at_times(eval_times).to_numpy()
    return eval_times, sf


def ground_truth_km_for_validation_set(
    scenario: Scenario, val_covariates: np.ndarray, base_seed: int,
    n_matched: int = 10_000, t_k: int = T_K,
) -> np.ndarray:
    """
    Ground-truth KM curve for every individual in a validation set (paper
    Section 4: ground truth is computed per validation covariate). This is
    the expensive step the paper describes -- O(len(val_covariates) *
    n_matched) simulated individuals. Returns an array of shape
    (len(val_covariates), t_k + 1).

    A distinct, deterministic seed is derived per validation individual (via
    NumPy's SeedSequence) so re-running this function reproduces the exact
    same ground-truth curves.
    """
    seed_seq = np.random.SeedSequence(base_seed)
    child_seeds = seed_seq.spawn(len(val_covariates))
    curves = np.empty((len(val_covariates), t_k + 1), dtype=np.float64)
    for i, (cov, child_seed) in enumerate(zip(val_covariates, child_seeds)):
        _, sf = ground_truth_km(scenario, cov, int(child_seed.generate_state(1)[0]), n_matched, t_k)
        curves[i] = sf
    return curves


# ---------------------------------------------------------------------------
# Correctness checks against the paper's stated generation rules
# ---------------------------------------------------------------------------
def validate_generation_invariants(states: np.ndarray, t_k: int = T_K) -> None:
    """
    Asserts, over a (small) batch of generated trajectories, that:
      1. Dead(3) and Censored(4) are absorbing (never left once entered).
      2. No individual has a run of Sicker(2) states reaching the
         time-dependent threshold (5 before step 20, 3 from step 20 on)
         without the run being terminated by a forced transition to Dead.
    Intended as a fast regression test on a subsample (e.g. 200-1000
    individuals), not the full 50,000-individual cohort.
    """
    n, T = states.shape

    for absorb_state in (DEAD, CENSORED):
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
            elif s in (DEAD, CENSORED):
                break
            else:
                run = 0


# ---------------------------------------------------------------------------
# End-to-end demonstration
# ---------------------------------------------------------------------------
GLOBAL_SEED = 20260909  # fixed top-level seed for the whole pipeline

SEEDS = {
    "ReLU_cohort": 1001,
    "ReLU_split": 1002,
    "Weibull_cohort": 2001,
    "Weibull_split": 2002,
    "ground_truth_base": 9001,
    "invariant_check_subsample": 5001,
}

# Paper's clinically-of-interest demonstration individuals (Section 4 / Figures 4-5)
RELU_DEMO_AGES = [65, 70, 75]
WEIBULL_DEMO_CALCIUMS = [8.0, 12.5, 13.0]


def main(n_cohort: int = 50_000, run_calibration: bool = False, run_ground_truth: bool = False):
    set_seed(GLOBAL_SEED)

    for scenario, cohort_seed, split_seed in (
        (RELU_SCENARIO, SEEDS["ReLU_cohort"], SEEDS["ReLU_split"]),
        (WEIBULL_SCENARIO, SEEDS["Weibull_cohort"], SEEDS["Weibull_split"]),
    ):
        print(f"\n=== {scenario.name} data generation model (covariate: {scenario.covariate_name}) ===")

        if run_calibration:
            calib = calibrate_covariate_params(scenario, seed=cohort_seed)
            print(f"  Calibrated covariate params: {calib}")
            scenario = replace(
                scenario, covariate_mean=calib["covariate_mean"], covariate_sd=calib["covariate_sd"]
            )

        cohort = generate_cohort(scenario, n=n_cohort, seed=cohort_seed)

        # Fast correctness check on a subsample (the full check is O(n * t_k) in
        # pure Python and not needed for every individual).
        rng = make_rng(SEEDS["invariant_check_subsample"])
        check_idx = rng.choice(n_cohort, size=min(500, n_cohort), replace=False)
        validate_generation_invariants(cohort["states"][check_idx])
        print("  [OK] absorbing-state and termination-rule invariants hold on a 500-sample check.")

        summary = cohort_summary(cohort)
        print(f"  Simulated : dropout censoring={summary['dropout_censoring_rate']:.3%}, "
              f"total censoring={summary['total_censoring_rate']:.3%}, "
              f"mean TTD={summary['mean_ttd']:.1f}, median TTD={summary['median_ttd']:.1f}")
        print(f"  Paper target: dropout censoring={scenario.target_dropout_censoring_rate:.3%}, "
              f"total censoring={scenario.target_total_censoring_rate:.3%}, "
              f"mean TTD={scenario.target_mean_ttd:.1f}, median TTD={scenario.target_median_ttd:.1f}")

        splits = train_test_val_split(cohort, seed=split_seed)
        for name, subset in splits.items():
            print(f"  {name}: n={len(subset['covariate'])}")

        out_path = f"/content/{scenario.name.lower()}_cohort.npz"
        np.savez_compressed(
            out_path,
            covariate=cohort["covariate"], states=cohort["states"],
            duration=cohort["duration"], event=cohort["event"],
            train_index=splits["train"]["index"], test_index=splits["test"]["index"],
            val_index=splits["val"]["index"], seed=cohort_seed,
        )
        print(f"  Saved cohort + split indices to {out_path}")

        if run_ground_truth:
            demo_values = RELU_DEMO_AGES if scenario.name == "ReLU" else WEIBULL_DEMO_CALCIUMS
            for v in demo_values:
                _, sf = ground_truth_km(scenario, v, seed=SEEDS["ground_truth_base"] + int(v * 10))
                print(f"  Ground-truth S(t) for {scenario.covariate_name}={v}: "
                      f"S(0)={sf[0]:.3f}, S(15)={sf[15]:.3f}, S(30)={sf[30]:.3f}")


if __name__ == "__main__":
    # Defaults keep runtime short; pass run_calibration=True / run_ground_truth=True
    # and n_cohort=50_000 to match the paper's full setup once you're ready to
    # spend the compute.
    main(n_cohort=50_000, run_calibration=False, run_ground_truth=True)