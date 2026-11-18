"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# --- numpy trapz/trapezoid compatibility (renamed in numpy>=2.0) ----------
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ===========================================================================
# CONFIGURATION -- same knobs, same defaults, as model_comparison.py (see
# that file's docstring for the runtime-cost rationale; nothing about
# competing risks changes any of these numbers).
# ===========================================================================
N_EVAL = None            # None = full test split; set an int (e.g. 500) to subsample for a faster run
GT_N_MATCHED = 1000       # matched-individual count per ground-truth Aalen-Johansen curve
GLAZIER_N_SAMPLES = 150   # generated trajectories per individual, for EACH Glazier variant's predictions
CONCORDANCE_MAX_N = 2000  # concordance is an O(n^2) pairwise computation -- subsampled independently of N_EVAL
DEMO_SEED = 4242
OUT_DIR = "/content/"     # where PNGs / .tex / .csv files are saved


# ===========================================================================
# Loading a saved competing-risks cohort. See module docstring -- nothing in
# this codebase writes this file for you yet; `sfsa_competing_risks.py`'s
# `generate_cohort_cr` + `train_test_val_split_cr` build the arrays, you add
# the np.savez call with this exact schema.
# ===========================================================================
def load_cohort_cr_npz(path):
    data = np.load(path)
    return {
        "covariate": data["covariate"],
        "states": data["states"],      # full 6-state trajectories (not needed below, kept for parity/future use)
        "duration": data["duration"],
        "cause": data["cause"],        # 0 = censored, 1 = disease, 2 = other (see sfsa_competing_risks.py)
        "train_index": data["train_index"],
        "test_index": data["test_index"],
        "val_index": data["val_index"],
    }


# ===========================================================================
# Metrics -- mae_per_timestep and concordance_index_np are REUSED VERBATIM
# from model_comparison.py (copy, not reimplementation); the two
# competing-risks metrics below are direct generalizations of that file's
# concordance index and integrated_brier_score, one cause at a time.
# ===========================================================================
def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


def concordance_index_np(duration, event, risk_score, max_pairs_n=None, seed=0):
    """
    Harrell's C (verbatim copy from model_comparison.py -- see that file's
    docstring for the verification note). A pair (i, j) is "comparable" if
    i's observed duration is strictly shorter than j's AND i's shorter
    duration was an actual observed event (not censoring). Concordant if
    the higher-risk individual (i) also has the higher risk score. Ties in
    risk score get half credit. O(n^2) -- `max_pairs_n` randomly subsamples
    first if the population is larger than that.
    """
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event)
    risk_score = np.asarray(risk_score, dtype=np.float64)
    n = len(duration)
    if max_pairs_n is not None and n > max_pairs_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_pairs_n, replace=False)
        duration, event, risk_score = duration[idx], event[idx], risk_score[idx]
        n = max_pairs_n

    d_i = duration[:, None]
    d_j = duration[None, :]
    e_i = event[:, None]
    r_i = risk_score[:, None]
    r_j = risk_score[None, :]

    comparable = (d_i < d_j) & (e_i == 1)
    concordant = comparable & (r_i > r_j)
    tied = comparable & (r_i == r_j)
    n_comparable = int(comparable.sum())
    if n_comparable == 0:
        return float("nan"), 0
    c_index = (concordant.sum() + 0.5 * tied.sum()) / n_comparable
    return float(c_index), n_comparable


def concordance_index_cr(duration, cause, cif_k_over_time, eval_times, which_cause, max_pairs_n=None, seed=0):
    """
    Competing-risks concordance for ONE cause `k` (`which_cause` is a raw
    `cause` CODE -- 1 for disease, 2 for other, matching `cause`'s own
    encoding, NOT the CIF array's 0/1 axis-index convention). A thin
    wrapper, not a reimplementation: builds the binary "did THIS individual
    experience cause k" event indicator and a single summary risk score per
    individual -- the trapezoidal area under their own predicted cause-k
    CIF curve, i.e. "expected cumulative-incidence burden of cause k" (the
    CIF analogue of the base harness's `-RMST`: a bigger area under a
    rising CIF curve means the model thinks this individual picks up cause
    k earlier/more surely, so it should rank as HIGHER risk, unlike RMST
    where a bigger area meant LOWER risk -- hence no sign flip here) --
    then hands both off to the existing, already-verified
    `concordance_index_np` completely unmodified.

    cif_k_over_time: (n, len(eval_times)) predicted CIF curve for cause k
    ONLY (already sliced out of the model's (n, 2, len(eval_times)) output
    -- this function is cause-index-agnostic, it doesn't care which axis
    position the caller sliced from).
    """
    event_k = (np.asarray(cause) == which_cause).astype(int)
    risk_score = _trapz(cif_k_over_time, eval_times, axis=1)
    return concordance_index_np(duration, event_k, risk_score, max_pairs_n=max_pairs_n, seed=seed)


def integrated_brier_score_cr(cif_k_pred, eval_times, duration, cause, which_cause, censoring_survival_fn, eps=None):
    """Cause-specific IPCW; G(T-) for any observed event, NaN if unsupported."""
    if which_cause not in (1,2): raise ValueError('cause must be 1 or 2')
    pred=np.asarray(cif_k_pred,dtype=float); times=np.asarray(eval_times,dtype=float)
    duration=np.asarray(duration,dtype=float);cause=np.asarray(cause)
    if pred.shape != (len(duration),len(times)): raise ValueError('shape mismatch')
    if len(times)<2 or np.any(np.diff(times)<=0): raise ValueError('increasing grid required')
    gl=np.asarray(censoring_survival_fn(np.nextafter(duration,-np.inf)))
    bs=np.full(len(times),np.nan)
    for j,t in enumerate(times):
        gt=float(censoring_survival_fn(np.array([t]))[0]); done=(duration<=t)&(cause>0); alive=duration>t
        if not np.isfinite(gt) or gt<=0 or np.any(~np.isfinite(gl[done])) or np.any(gl[done]<=0):continue
        v=np.zeros(len(duration));v[done]=((cause[done]==which_cause)-pred[done,j])**2/gl[done]
        v[alive]=pred[alive,j]**2/gt;bs[j]=v.mean()
    return bs,float(np.trapezoid(bs,times)/(times[-1]-times[0]))


# ===========================================================================
# Model registry -- auto-detects which trained CR models exist in the
# notebook's global namespace, exactly like model_comparison.py's own
# registry, just with the `_cr`-suffixed naming convention and
# `predict_cif_*` functions from the design doc's interface table.
# ===========================================================================
_GENERIC_MODEL_SPECS_CR = [
    {"label": "DRSA", "color": "tab:brown",
     "predict_fn_name": "predict_cif_drsa", "model_prefix": "drsa_model_cr", "meta_prefix": "drsa_meta_cr"},
    {"label": "WTTE-RNN", "color": "tab:orange",
     "predict_fn_name": "predict_cif_wtte", "model_prefix": "wtte_model_cr", "meta_prefix": "wtte_meta_cr"},
    {"label": "Dynamic-DeepHit", "color": "tab:green",
     "predict_fn_name": "predict_cif_dh", "model_prefix": "dh_model_cr", "meta_prefix": "dh_meta_cr"},
    {"label": "SurvTransformer", "color": "tab:olive",
     "predict_fn_name": "predict_cif_survtransformer",
     "model_prefix": "survtransformer_model_cr", "meta_prefix": "survtransformer_meta_cr"},
]
_SAMPLING_MODEL_SPECS_CR = [
    {"label": "Glazier-GRU", "color": "tab:purple",
     "predict_fn_name": "predict_cif_glazier",
     "model_prefix": "glazier_model_cr", "meta_prefix": "glazier_meta_cr"},
    {"label": "Glazier-RNN", "color": "tab:pink",
     "predict_fn_name": "predict_cif_glazier_rnn",
     "model_prefix": "glazier_rnn_model_cr", "meta_prefix": "glazier_rnn_meta_cr"},
    {"label": "Glazier-Transformer", "color": "tab:gray",
     "predict_fn_name": "predict_cif_glazier_transformer",
     "model_prefix": "glazier_transformer_model_cr", "meta_prefix": "glazier_transformer_meta_cr"},
]
_SAMPLING_LABELS_CR = {spec["label"] for spec in _SAMPLING_MODEL_SPECS_CR}


def build_model_registry_cr(scenario, glazier_n_samples=GLAZIER_N_SAMPLES, glazier_seed=DEMO_SEED):
    suffix = "relu" if scenario.name == "ReLU" else "weibull"
    g = globals()
    registry = []

    # CPH is a special case, like the base harness: the variable is a
    # (cph_disease, cph_other) tuple (no "_model" infix), and "meta" is
    # unused (None) -- see cph_model_cr.py's predict_cif_cph.
    cph_var = f"cph_cr_{suffix}"
    if cph_var in g and "predict_cif_cph" in g:
        cph_pair = g[cph_var]
        pf = g["predict_cif_cph"]
        registry.append({
            "label": "CPH", "color": "tab:cyan",
            "predict_fn": lambda cov, t, _cph=cph_pair, _pf=pf: _pf(_cph, cov, t, covariate_name=scenario.covariate_name),
            "model_obj": cph_pair, "meta_obj": None, "is_sampling": False,
        })
    else:
        print(f"  [skip] CPH: '{cph_var}' or predict_cif_cph not found in your notebook -- run the CPH (CR) cell first.")

    for spec in _GENERIC_MODEL_SPECS_CR:
        model_var = f"{spec['model_prefix']}_{suffix}"
        meta_var = f"{spec['meta_prefix']}_{suffix}"
        fn_name = spec["predict_fn_name"]
        if model_var in g and meta_var in g and fn_name in g:
            model_obj, meta_obj, pf = g[model_var], g[meta_var], g[fn_name]
            registry.append({
                "label": spec["label"], "color": spec["color"],
                "predict_fn": lambda cov, t, _m=model_obj, _me=meta_obj, _pf=pf: _pf(_m, _me, cov, t),
                "model_obj": model_obj, "meta_obj": meta_obj, "is_sampling": False,
            })
        else:
            print(f"  [skip] {spec['label']}: '{model_var}'/'{meta_var}' not found -- run that model's CR cell first.")

    for spec in _SAMPLING_MODEL_SPECS_CR:
        model_var = f"{spec['model_prefix']}_{suffix}"
        meta_var = f"{spec['meta_prefix']}_{suffix}"
        fn_name = spec["predict_fn_name"]
        if model_var in g and meta_var in g and fn_name in g:
            model_obj, meta_obj, pf = g[model_var], g[meta_var], g[fn_name]
            registry.append({
                "label": spec["label"], "color": spec["color"],
                "predict_fn": lambda cov, t, _m=model_obj, _me=meta_obj, _pf=pf: _pf(
                    _m, _me, cov, t, n_samples=glazier_n_samples, seed=glazier_seed),
                "model_obj": model_obj, "meta_obj": meta_obj, "is_sampling": True,
            })
        else:
            print(f"  [skip] {spec['label']}: '{model_var}'/'{meta_var}' not found -- run that model's CR cell first.")

    return registry


def predict_curves_with_progress_cr(predict_fn, covariate_values, eval_times, label, chunk_size=500):
    """
    Chunked-prediction helper, analogous to model_comparison.py's
    `predict_curves_with_progress`, but for a `predict_fn` that returns
    `(chunk_n, N_CAUSES, len(eval_times))` cause-stacked CIF arrays instead
    of a single `(chunk_n, len(eval_times))` survival array.
    """
    n = len(covariate_values)
    out = np.empty((n, N_CAUSES, len(eval_times)))
    start_time = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = predict_fn(covariate_values[start:end], eval_times)
        elapsed = time.time() - start_time
        rate = end / elapsed if elapsed > 0 else float("inf")
        remaining = (n - end) / rate if rate > 0 else float("nan")
        print(f"    [{label}] {end}/{n} individuals predicted "
              f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
    return out


# ===========================================================================
# LaTeX table generation -- REUSED VERBATIM from model_comparison.py.
# ===========================================================================
def _escape_latex(s):
    return str(s).replace("\\", "").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def to_latex_table(df, caption, label, float_format="{:.4f}"):
    cols = list(df.columns)
    col_spec = "l" + "c" * (len(cols) - 1)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(_escape_latex(c) for c in cols) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(float_format.format(v))
            else:
                cells.append(_escape_latex(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{_escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ===========================================================================
# Plotting -- same dpi=150 / tight_layout / color-table style as
# model_comparison.py, extended to three curves (survival, disease CIF,
# other CIF) per demo covariate value instead of one.
# ===========================================================================
def plot_cif_comparison_cr(scenario, eval_times, demo_values, ground_truth_demo, predicted_demo_by_model, out_path):
    """
    3 rows (overall survival, disease CIF, other-cause CIF) x
    len(demo_values) columns, sharing x across all axes and y within each
    row (survival lives in [0, 1]; the two CIFs are typically much smaller
    and on different scales from survival and from each other, so each row
    gets its own y-scale rather than forcing one global sharey like the
    base harness's single-row `plot_survival_comparison`).

    ground_truth_demo: list of dicts (one per demo value) with keys
    "survival"/"cif_disease"/"cif_other" -- the exact dict shape
    `ground_truth_cif` already returns, so no reshaping needed at the call
    site.
    predicted_demo_by_model: list of {"label", "color", "survival",
    "cif_disease", "cif_other"}, each value an (len(demo_values),
    len(eval_times)) array.
    """
    n_cols = len(demo_values)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.5 * n_cols, 4.0 * 3), sharex=True, sharey="row")
    axes = np.atleast_2d(axes).reshape(3, n_cols)

    row_specs = [("survival", "Survival probability"), ("cif_disease", "Disease CIF"), ("cif_other", "Other-cause CIF")]
    for col_idx, (v, gt) in enumerate(zip(demo_values, ground_truth_demo)):
        for row_idx, (key, _ylabel) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            ax.step(eval_times, gt[key], where="post", label="Ground truth", color="tab:red", linewidth=2)
            for entry in predicted_demo_by_model:
                ax.step(eval_times, entry[key][col_idx], where="post",
                         label=entry["label"], color=entry["color"], alpha=0.85)
            if row_idx == 0:
                ax.set_title(f"{scenario.covariate_name}={v}")
            if row_idx == 2:
                ax.set_xlabel("Time")
            ax.legend(fontsize=7)
    for row_idx, (_key, ylabel) in enumerate(row_specs):
        axes[row_idx, 0].set_ylabel(ylabel)

    plt.suptitle(f"All models vs. ground truth (competing risks) -- {scenario.name} scenario")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


def plot_mae_comparison_cr(scenario_label, eval_times, mae_curves, out_path):
    """
    1x3 line-plot panels (survival / disease CIF / other CIF), one line per
    model per panel -- analogous to model_comparison.py's single-axes
    `plot_mae_comparison`, just split three ways since there are now three
    MAE_t curves per model instead of one.

    mae_curves: dict with keys "survival"/"disease"/"other", each a list of
    {"label", "color", "mae_t"} dicts (one entry per model) -- mirrors the
    base harness's `mae_curves` list shape, one per curve type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_specs = [("survival", "Survival"), ("disease", "Disease CIF"), ("other", "Other-cause CIF")]
    for ax, (key, title) in zip(axes, panel_specs):
        for entry in mae_curves[key]:
            ax.plot(eval_times, entry["mae_t"], label=entry["label"], color=entry["color"])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean Absolute Prediction Error")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.suptitle(f"MAE$_t$ comparison (competing risks) -- {scenario_label} scenario, full test set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


def plot_metric_bars_cr(scenario_label, metrics_df, out_path):
    """
    1x2 bar charts: Integrated Brier Score (Disease) and Integrated Brier
    Score (Other) -- analogous to model_comparison.py's `plot_metric_bars`,
    but with the two cause-specific IBS numbers in place of the base
    harness's single Integrated-MAE/Integrated-Brier pair (MAE is already
    broken out into its own three-panel figure above, so this figure is
    reserved for the two headline calibration numbers).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(metrics_df["Model"], metrics_df["Integrated Brier (Disease)"], color="tab:blue")
    axes[0].set_ylabel("Integrated Brier Score (Disease)")
    axes[0].set_title("Integrated Brier (Disease) -- lower is better")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(metrics_df["Model"], metrics_df["Integrated Brier (Other)"], color="tab:red")
    axes[1].set_ylabel("Integrated Brier Score (Other)")
    axes[1].set_title("Integrated Brier (Other) -- lower is better")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle(f"Summary metrics (competing risks) -- {scenario_label} scenario")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


# ===========================================================================
# End-to-end comparison for one scenario
# ===========================================================================
# Cause bookkeeping used throughout run_comparison_for_scenario_cr: "label"
# is the CSV-column-friendly name, "axis_index" is the position in every
# model's (n, 2, len(eval_times)) CIF output (0=disease, 1=other -- the
# array-position convention, see module docstring), "cause_code" is the raw
# `cause` array value (1=disease, 2=other) used by concordance_index_cr /
# integrated_brier_score_cr's event indicators.
CAUSE_SPECS_CR = (
    {"key": "disease", "label": "Disease", "axis_index": 0, "cause_code": 1},
    {"key": "other", "label": "Other", "axis_index": 1, "cause_code": 2},
)


def run_comparison_for_scenario_cr(scenario, cohort_path, n_eval=N_EVAL, gt_n_matched=GT_N_MATCHED,
                                    glazier_n_samples=GLAZIER_N_SAMPLES, concordance_max_n=CONCORDANCE_MAX_N,
                                    demo_values=None, seed=DEMO_SEED, out_dir=OUT_DIR):
    cohort = load_cohort_cr_npz(cohort_path)
    test_idx = cohort["test_index"]
    train_idx = cohort["train_index"]

    if n_eval is not None and n_eval < len(test_idx):
        rng = np.random.default_rng(seed)
        eval_idx = rng.choice(test_idx, size=n_eval, replace=False)
    else:
        eval_idx = test_idx

    covariate_eval = cohort["covariate"][eval_idx]
    duration_eval = cohort["duration"][eval_idx]
    cause_eval = cohort["cause"][eval_idx]
    eval_times = np.arange(0, T_K + 1)

    print(f"\n{'=' * 70}\n{scenario.name} scenario (competing risks) -- evaluating on "
          f"{len(eval_idx)} test individuals\n{'=' * 70}")

    registry = build_model_registry_cr(scenario, glazier_n_samples=glazier_n_samples, glazier_seed=seed)
    if not registry:
        print("  No trained CR models found in the notebook namespace -- nothing to compare. "
              "Run at least one model's CR cell before this one.")
        return None

    # --- ground truth (simulated, Aalen-Johansen), computed ONCE per
    # scenario and shared across every model ---
    print(f"\n  Simulating ground-truth Aalen-Johansen CIF curves ({gt_n_matched} matched individuals each) "
          f"for {len(eval_idx)} test individuals -- this is the expensive shared step...")
    t0 = time.time()
    ground_truth = ground_truth_cif_for_validation_set(  # reused from the sfsa_competing_risks.py cell
        scenario, covariate_eval, base_seed=seed, n_matched=gt_n_matched, t_k=T_K,
        cause_a_share_fn=default_cause_a_share,
    )
    print(f"  done in {time.time() - t0:.1f}s.")
    # Stack into (n, 2, T_K+1) so every cause-specific metric below can
    # index by axis position exactly like a model's own predicted output --
    # same array-position convention (0=disease, 1=other) everywhere.
    ground_truth_cif_stack = np.stack([ground_truth["cif_disease"], ground_truth["cif_other"]], axis=1)

    # --- censoring KM, fit on the TRAINING split, for IPCW weights. Here
    # "was censored" is cause==0 (cause>0 means an actual death was
    # observed, of either cause) -- same idea as `1 - cohort["event"]` in
    # the base harness, just using cause==0 as the censoring indicator ---
    censoring_kmf = KaplanMeierFitter()  # reused from the sfsa_data_generation.py cell (lifelines)
    censoring_kmf.fit(cohort["duration"][train_idx], event_observed=(cohort["cause"][train_idx] == 0).astype(int))
    censoring_survival_fn = lambda t: censoring_kmf.survival_function_at_times(t).to_numpy()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    metrics_rows = []
    mae_curves = {"survival": [], "disease": [], "other": []}
    predicted_demo_by_model = []

    for entry in registry:
        label = entry["label"]
        print(f"\n  -- {label} --")
        if label == "Glazier-Transformer":
            chunk_size = 20   # slowest variant (O(T^2) generation) -- smaller chunks for more responsive progress prints
        elif label in _SAMPLING_LABELS_CR:
            chunk_size = 50   # other Glazier variants -- generative, still much slower than discriminative models
        else:
            chunk_size = 500  # discriminative models -- one forward pass per individual

        predicted = predict_curves_with_progress_cr(
            entry["predict_fn"], covariate_eval, eval_times, label, chunk_size=chunk_size,
        )  # (n_eval, 2, len(eval_times))

        predicted_disease = predicted[:, 0, :]
        predicted_other = predicted[:, 1, :]
        # S(t) = 1 - CIF_disease(t) - CIF_other(t), exact by the CIF
        # partition's definition -- clipped only for float-roundoff safety
        # (the interface contract allows cif_disease+cif_other up to
        # 1+1e-6, which would otherwise make survival very slightly negative).
        predicted_survival = np.clip(1.0 - predicted_disease - predicted_other, 0.0, 1.0)

        mae_t_survival = mae_per_timestep(predicted_survival, ground_truth["survival"])
        mae_t_disease = mae_per_timestep(predicted_disease, ground_truth_cif_stack[:, 0, :])
        mae_t_other = mae_per_timestep(predicted_other, ground_truth_cif_stack[:, 1, :])

        row = {
            "Model": label,
            "Mean MAE (Survival)": mae_t_survival.mean(),
            "Mean MAE (Disease CIF)": mae_t_disease.mean(),
            "Mean MAE (Other CIF)": mae_t_other.mean(),
            "Integrated MAE (Survival)": _trapz(mae_t_survival, eval_times),
            "Integrated MAE (Disease CIF)": _trapz(mae_t_disease, eval_times),
            "Integrated MAE (Other CIF)": _trapz(mae_t_other, eval_times),
            "N Eval": len(eval_idx),
        }

        cif_per_cause = {0: predicted_disease, 1: predicted_other}
        c_index_by_label = {}
        ibs_by_label = {}
        for spec in CAUSE_SPECS_CR:
            cif_k = cif_per_cause[spec["axis_index"]]
            c_index, n_comparable = concordance_index_cr(
                duration_eval, cause_eval, cif_k, eval_times, which_cause=spec["cause_code"],
                max_pairs_n=concordance_max_n, seed=seed,
            )
            bs_t, ibs = integrated_brier_score_cr(
                cif_k[:, np.asarray(eval_times) < 30], np.asarray(eval_times)[np.asarray(eval_times) < 30], duration_eval, cause_eval, which_cause=spec["cause_code"],
                censoring_survival_fn=censoring_survival_fn,
            )
            row[f"Concordance ({spec['label']})"] = c_index
            row[f"Integrated Brier ({spec['label']})"] = ibs
            row[f"N Comparable Pairs ({spec['label']})"] = n_comparable
            c_index_by_label[spec["label"]] = c_index
            ibs_by_label[spec["label"]] = ibs

        metrics_rows.append(row)
        mae_curves["survival"].append({"label": label, "color": entry["color"], "mae_t": mae_t_survival})
        mae_curves["disease"].append({"label": label, "color": entry["color"], "mae_t": mae_t_disease})
        mae_curves["other"].append({"label": label, "color": entry["color"], "mae_t": mae_t_other})

        demo_pred = entry["predict_fn"](np.asarray(demo_values, dtype=np.float64), eval_times)  # (n_demo, 2, T+1)
        demo_disease = demo_pred[:, 0, :]
        demo_other = demo_pred[:, 1, :]
        demo_survival = np.clip(1.0 - demo_disease - demo_other, 0.0, 1.0)
        predicted_demo_by_model.append({
            "label": label, "color": entry["color"],
            "survival": demo_survival, "cif_disease": demo_disease, "cif_other": demo_other,
        })

        print(f"    MAE(surv)={row['Mean MAE (Survival)']:.4f}  "
              f"MAE(disease)={row['Mean MAE (Disease CIF)']:.4f}  MAE(other)={row['Mean MAE (Other CIF)']:.4f}  "
              f"C(disease)={c_index_by_label['Disease']:.4f}  C(other)={c_index_by_label['Other']:.4f}  "
              f"IBS(disease)={ibs_by_label['Disease']:.4f}  IBS(other)={ibs_by_label['Other']:.4f}")

    # Reorder columns to exactly match the documented CSV layout (dict
    # insertion order above already does this, `pd.DataFrame` just
    # preserves it, but spelled out explicitly here so the on-disk column
    # order can't silently drift if the loop above is ever edited).
    csv_columns = [
        "Model", "Mean MAE (Survival)", "Mean MAE (Disease CIF)", "Mean MAE (Other CIF)",
        "Integrated MAE (Survival)", "Integrated MAE (Disease CIF)", "Integrated MAE (Other CIF)",
        "Concordance (Disease)", "Concordance (Other)",
        "Integrated Brier (Disease)", "Integrated Brier (Other)",
        "N Eval", "N Comparable Pairs (Disease)", "N Comparable Pairs (Other)",
    ]
    metrics_df = pd.DataFrame(metrics_rows)[csv_columns]

    # --- ground truth curves at the demo covariate values (for the CIF figure) ---
    ground_truth_demo = []
    for v in demo_values:
        gt_demo = ground_truth_cif(  # reused from the sfsa_competing_risks.py cell
            scenario, v, seed=seed, n_matched=max(gt_n_matched, 2000), t_k=T_K,
            cause_a_share_fn=default_cause_a_share,
        )
        ground_truth_demo.append(gt_demo)

    os.makedirs(out_dir, exist_ok=True)
    scenario_slug = scenario.name.lower()

    print(f"\n  --- {scenario.name} (competing risks) comparison table ---")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}_cr.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    latex_cols = [
        "Model", "Mean MAE (Survival)", "Mean MAE (Disease CIF)", "Mean MAE (Other CIF)",
        "Concordance (Disease)", "Concordance (Other)",
        "Integrated Brier (Disease)", "Integrated Brier (Other)",
    ]
    latex_str = to_latex_table(
        metrics_df[latex_cols],
        caption=f"Competing-risks model comparison on the full {scenario.name} test set (n={len(eval_idx)}).",
        label=f"tab:comparison_cr_{scenario_slug}",
    )
    tex_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}_cr.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"  Saved {tex_path} -- paste this directly into your Overleaf document:\n")
    print(latex_str)

    plot_mae_comparison_cr(scenario.name, eval_times, mae_curves,
                            os.path.join(out_dir, f"fig_mae_comparison_{scenario_slug}_cr.png"))
    plot_cif_comparison_cr(scenario, eval_times, demo_values, ground_truth_demo, predicted_demo_by_model,
                            os.path.join(out_dir, f"fig_cif_comparison_{scenario_slug}_cr.png"))
    plot_metric_bars_cr(scenario.name, metrics_df,
                         os.path.join(out_dir, f"fig_metric_bars_{scenario_slug}_cr.png"))

    return {
        "metrics_df": metrics_df,
        "mae_curves": mae_curves,
        "ground_truth": ground_truth,
        "eval_times": eval_times,
        "eval_idx": eval_idx,
    }


# ===========================================================================
# Run for both scenarios
# ===========================================================================
if __name__ == '__main__':
    results_relu_cr = run_comparison_for_scenario_cr(RELU_SCENARIO, "/content/relu_cohort_cr.npz")
    results_weibull_cr = run_comparison_for_scenario_cr(WEIBULL_SCENARIO, "/content/weibull_cohort_cr.npz")

    if results_relu_cr is not None and results_weibull_cr is not None:
        combined_df_cr = pd.concat(
            [results_relu_cr["metrics_df"].assign(Scenario="ReLU"),
             results_weibull_cr["metrics_df"].assign(Scenario="Weibull")],
            ignore_index=True,
        )
        combined_csv_path_cr = os.path.join(OUT_DIR, "comparison_table_combined_cr.csv")
        combined_df_cr.to_csv(combined_csv_path_cr, index=False)
        print(f"\nSaved combined competing-risks table across both scenarios: {combined_csv_path_cr}")
