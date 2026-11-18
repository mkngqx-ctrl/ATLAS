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
# that file's docstring for the runtime-cost rationale; unchanged here since
# the semi-Markov cohorts are the same size/shape as the base ones).
# ===========================================================================
N_EVAL = None             # None = full test split; set an int to subsample for a faster run
GT_N_MATCHED = 1000        # matched-individual count per ground-truth KM curve (prospective)
GLAZIER_N_SAMPLES = 150    # generated trajectories per individual, per Glazier variant
CONCORDANCE_MAX_N = 2000   # concordance is O(n^2) -- subsampled independently of N_EVAL
DEMO_SEED = 4242
OUT_DIR = "/content/"      # where PNGs / .tex / .csv files are saved

# Landmark-run configuration -- separate, smaller-by-default knobs (see
# run_landmark_comparison_for_scenario_sm), same defaults as the base harness.
LANDMARK_STEPS = (10, 15, 20)
LANDMARK_N_EVAL = 300
LANDMARK_GT_N_MATCHED = 500
LANDMARK_GLAZIER_N_SAMPLES = 100


# ===========================================================================
# Loading a saved semi-Markov cohort -- identical .npz schema to the base
# cohort (generate_cohort_sm returns the same dict shape as generate_cohort),
# so this loader is reused verbatim from model_comparison.py.
# ===========================================================================
def load_cohort_npz(path):
    data = np.load(path)
    return {
        "covariate": data["covariate"],
        "states": data["states"],  # needed for the landmark comparison below
        "duration": data["duration"],
        "event": data["event"],
        "train_index": data["train_index"],
        "test_index": data["test_index"],
        "val_index": data["val_index"],
    }


# ===========================================================================
# Metric functions -- reused VERBATIM from model_comparison.py. See that
# file's module docstring for what each metric means and how it was
# verified; unchanged here since none of this math is specific to how the
# cohort's states were generated.
# ===========================================================================
def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


def restricted_mean_survival_time(S, eval_times):
    """Area under each individual's own predicted survival curve -- used
    as the summary "risk score" for the concordance index below (higher
    RMST = predicted to survive longer = lower risk)."""
    return _trapz(S, eval_times, axis=1)


def concordance_index_np(duration, event, risk_score, max_pairs_n=None, seed=0):
    """
    A pair (i, j) is "comparable" if individual i's observed duration is
    strictly shorter than j's AND i's shorter duration was an actual
    observed event (not censoring) -- the standard Harrell's-C definition.
    Concordant if the higher-risk individual (i) also has the higher risk
    score. Ties in risk score get half credit. O(n^2) -- `max_pairs_n`
    randomly subsamples first if the population is larger than that.
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


def integrated_brier_score(S_pred, eval_times, duration, event, censoring_survival_fn, eps=None):
    """IPCW with G(T-); unsupported times are NaN, never epsilon clipped.
    IBS uses the supplied grid and is NaN if any requested time lacks support.
    For the preserved administrative horizon use times strictly below 30.
    """
    S_pred=np.asarray(S_pred, dtype=float); times=np.asarray(eval_times,dtype=float)
    duration=np.asarray(duration,dtype=float); event=np.asarray(event)
    if S_pred.shape != (len(duration),len(times)): raise ValueError('shape mismatch')
    if len(times)<2 or np.any(np.diff(times)<=0): raise ValueError('increasing grid required')
    gleft=np.asarray(censoring_survival_fn(np.nextafter(duration,-np.inf)))
    bs=np.full(len(times),np.nan)
    for j,t in enumerate(times):
        gt=float(censoring_survival_fn(np.array([t]))[0])
        failed=(duration<=t)&(event==1); alive=duration>t
        if not np.isfinite(gt) or gt<=0 or np.any(~np.isfinite(gleft[failed])) or np.any(gleft[failed]<=0): continue
        v=np.zeros(len(duration)); v[failed]=S_pred[failed,j]**2/gleft[failed]
        v[alive]=(1-S_pred[alive,j])**2/gt; bs[j]=v.mean()
    return bs, float(np.trapezoid(bs,times)/(times[-1]-times[0]))


# ===========================================================================
# Model registry -- auto-detects which trained `*_sm` models exist in the
# notebook's global namespace, exactly like model_comparison.py's
# `build_model_registry`, but looking for the `_sm` infix BEFORE the
# scenario suffix (`{prefix}_sm_{relu|weibull}` instead of
# `{prefix}_{relu|weibull}`) -- same pattern as this codebase's competing-
# risks (`_cr`) files. The spec lists themselves (labels, colors,
# prefixes, predict_fn names) are otherwise IDENTICAL to the base harness's,
# since training data differs but the prediction math/interface does not
# (see module docstring) -- only the predict_fn NAMES for the prospective
# protocol are unchanged; the landmark protocol uses separate, new
# `*_landmark_sm` predict functions for the 3 Glazier variants (see
# `_GLAZIER_LANDMARK_FN_NAMES` below), since the 5 discriminative models'
# landmark prediction reuses `predict_survival_landmark_conditional_batch`
# rather than a model-specific function at all.
# ===========================================================================
_GENERIC_MODEL_SPECS_SM = [
    {"label": "DRSA", "color": "tab:brown",
     "predict_fn_name": "predict_survival_drsa", "model_prefix": "drsa_model", "meta_prefix": "drsa_meta"},
    {"label": "WTTE-RNN", "color": "tab:orange",
     "predict_fn_name": "predict_survival_wtte", "model_prefix": "wtte_model", "meta_prefix": "wtte_meta"},
    {"label": "Dynamic-DeepHit", "color": "tab:green",
     "predict_fn_name": "predict_survival_dh", "model_prefix": "dh_model", "meta_prefix": "dh_meta"},
    {"label": "SurvTransformer", "color": "tab:olive",
     "predict_fn_name": "predict_survival_survtransformer",
     "model_prefix": "survtransformer_model", "meta_prefix": "survtransformer_meta"},
]
_SAMPLING_MODEL_SPECS_SM = [
    {"label": "Glazier-GRU", "color": "tab:purple",
     "predict_fn_name": "predict_survival_glazier",
     "model_prefix": "glazier_model", "meta_prefix": "glazier_meta"},
    {"label": "Glazier-RNN", "color": "tab:pink",
     "predict_fn_name": "predict_survival_glazier_rnn",
     "model_prefix": "glazier_rnn_model", "meta_prefix": "glazier_rnn_meta"},
    {"label": "Glazier-Transformer", "color": "tab:gray",
     "predict_fn_name": "predict_survival_glazier_transformer",
     "model_prefix": "glazier_transformer_model", "meta_prefix": "glazier_transformer_meta"},
]
_SAMPLING_LABELS_SM = {spec["label"] for spec in _SAMPLING_MODEL_SPECS_SM}
_DISCRIMINATIVE_LABELS_SM = {"CPH", "DRSA", "WTTE-RNN", "Dynamic-DeepHit", "SurvTransformer"}

# New landmark-only predict functions, one per Glazier variant (see
# glazier_model_sm.py / glazier_rnn_model_sm.py / glazier_transformer_model_sm.py
# -- each adds its own variant of this, since the GRU/RNN variants thread an
# explicit hidden state through the observed prefix while the Transformer
# variant re-encodes the whole growing history at every step; see
# semi_markov_design.md section 3).
_GLAZIER_LANDMARK_FN_NAMES = {
    "Glazier-GRU": "predict_survival_glazier_landmark_sm",
    "Glazier-RNN": "predict_survival_glazier_rnn_landmark_sm",
    "Glazier-Transformer": "predict_survival_glazier_transformer_landmark_sm",
}


def build_model_registry_sm(scenario, glazier_n_samples=GLAZIER_N_SAMPLES, glazier_seed=DEMO_SEED):
    """Same auto-detection logic as model_comparison.py's build_model_registry,
    with the `_sm` infix: variables must be named `{prefix}_sm_{relu|weibull}`
    (e.g. `drsa_model_sm_relu`, `glazier_meta_sm_weibull`, `cph_sm_relu`)."""
    suffix = "sm_relu" if scenario.name == "ReLU" else "sm_weibull"
    g = globals()
    registry = []

    # CPH special case, same as the base harness: global `predict_survival`
    # function (unchanged name -- CPH's prediction math never changed).
    cph_var = f"cph_{suffix}"
    if cph_var in g and "predict_survival" in g:
        cph_obj = g[cph_var]
        pf = g["predict_survival"]
        registry.append({
            "label": "CPH", "color": "tab:cyan",
            "predict_fn": lambda cov, t, _cph=cph_obj, _pf=pf: _pf(_cph, cov, t, covariate_name=scenario.covariate_name),
            "model_obj": cph_obj, "meta_obj": None, "is_sampling": False,
        })
    else:
        print(f"  [skip] CPH: '{cph_var}' or predict_survival not found in your notebook -- run the CPH-sm cell first.")

    for spec in _GENERIC_MODEL_SPECS_SM:
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
            print(f"  [skip] {spec['label']}: '{model_var}'/'{meta_var}' not found -- run that model's sm cell first.")

    for spec in _SAMPLING_MODEL_SPECS_SM:
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
            print(f"  [skip] {spec['label']}: '{model_var}'/'{meta_var}' not found -- run that model's sm cell first.")

    return registry


def predict_curves_with_progress(predict_fn, covariate_values, eval_times, label, chunk_size=500):
    """Reused verbatim from model_comparison.py -- chunked prediction with
    progress/ETA printing, mainly useful for the (much slower) Glazier
    variants."""
    n = len(covariate_values)
    out = np.empty((n, len(eval_times)))
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
# LaTeX table generation -- reused verbatim from model_comparison.py.
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
# Plotting -- reused from model_comparison.py, almost verbatim. The only
# adaptation: plot_survival_comparison gains an optional `tag` suffix for
# its suptitle (e.g. " (semi-Markov, prospective)") so the two protocols'
# figures are distinguishable at a glance without changing the function's
# call signature for any existing caller. plot_mae_comparison and
# plot_metric_bars need no change at all -- both already take a free-form
# `scenario_label` string, so the protocol tag is simply folded into the
# string passed in by each driver below.
# ===========================================================================
def plot_mae_comparison(scenario_label, eval_times, mae_curves, out_path):
    plt.figure(figsize=(7, 5))
    for entry in mae_curves:
        plt.plot(eval_times, entry["mae_t"], label=entry["label"], color=entry["color"])
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"MAE$_t$ comparison -- {scenario_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


def plot_survival_comparison(scenario, eval_times, demo_values, ground_truth_demo, predicted_demo_by_model,
                              out_path, tag=""):
    fig, axes = plt.subplots(1, len(demo_values), figsize=(5.5 * len(demo_values), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, v, sf_gt in zip(axes, demo_values, ground_truth_demo):
        ax.step(eval_times, sf_gt, where="post", label="Ground truth", color="tab:red", linewidth=2)
        for entry in predicted_demo_by_model:
            ax.step(eval_times, entry["curves"][list(demo_values).index(v)], where="post",
                     label=entry["label"], color=entry["color"], alpha=0.85)
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.set_xlabel("Time")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Survival probability")
    plt.suptitle(f"All models vs. ground truth -- {scenario.name} scenario{tag}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


def plot_metric_bars(scenario_label, metrics_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(metrics_df["Model"], metrics_df["Integrated MAE (AUC)"], color="tab:blue")
    axes[0].set_ylabel("Integrated MAE (AUC)")
    axes[0].set_title("Integrated MAE -- lower is better")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(metrics_df["Model"], metrics_df["Integrated Brier Score"], color="tab:red")
    axes[1].set_ylabel("Integrated Brier Score")
    axes[1].set_title("Integrated Brier Score -- lower is better")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle(f"Summary metrics -- {scenario_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


# ===========================================================================
# PROTOCOL 1: PROSPECTIVE comparison for one scenario ("S(t|covariate) from
# t=0") -- nearly identical to model_comparison.py's run_comparison_for_scenario,
# differing only in: (a) the registry is built from the `_sm`-suffixed
# globals, (b) ground truth comes from ground_truth_km_sm_for_validation_set
# (semi-Markov hazard dynamics) instead of the base ground_truth_km_for_validation_set.
#
# THIS IS THE DOCUMENTED NULL-RESULT SANITY CHECK (see module docstring): at
# t=0 every individual's sojourn time is trivially 1 (they have just entered
# their initial state), so there is no sojourn-time signal for ANY model to
# exploit yet -- all 8 models are expected to land close together on every
# metric here. If they do NOT -- if some model dramatically outperforms the
# others on this run -- that is a red flag about training/calibration, not
# evidence the semi-Markov extension "worked"; the real test is the landmark
# protocol below.
# ===========================================================================
def run_comparison_for_scenario_sm(scenario, cohort_path, cohort_train_for_censoring=None,
                                    n_eval=N_EVAL, gt_n_matched=GT_N_MATCHED,
                                    glazier_n_samples=GLAZIER_N_SAMPLES,
                                    concordance_max_n=CONCORDANCE_MAX_N,
                                    demo_values=None, seed=DEMO_SEED, out_dir=OUT_DIR,
                                    hazard_kwargs=None):
    if hazard_kwargs is None:
        hazard_kwargs = dict(DEFAULT_HAZARD_PARAMS)  # reused from the sfsa_semi_markov cell

    cohort = load_cohort_npz(cohort_path)
    test_idx = cohort["test_index"]
    train_idx = cohort["train_index"]

    if n_eval is not None and n_eval < len(test_idx):
        rng = np.random.default_rng(seed)
        eval_idx = rng.choice(test_idx, size=n_eval, replace=False)
    else:
        eval_idx = test_idx

    covariate_eval = cohort["covariate"][eval_idx]
    duration_eval = cohort["duration"][eval_idx]
    event_eval = cohort["event"][eval_idx]
    eval_times = np.arange(0, T_K + 1)

    print(f"\n{'=' * 70}\n{scenario.name} scenario (semi-Markov, PROSPECTIVE protocol) -- "
          f"evaluating on {len(eval_idx)} test individuals\n{'=' * 70}")

    registry = build_model_registry_sm(scenario, glazier_n_samples=glazier_n_samples, glazier_seed=seed)
    if not registry:
        print("  No trained semi-Markov models found in the notebook namespace -- nothing to compare. "
              "Run at least one model's *_sm cell before this one.")
        return None

    # --- ground truth (simulated under the semi-Markov hazard dynamics), computed ONCE per scenario ---
    print(f"\n  Simulating prospective ground-truth KM curves ({gt_n_matched} matched individuals each) "
          f"for {len(eval_idx)} test individuals -- this is the expensive shared step...")
    t0 = time.time()
    ground_truth = ground_truth_km_sm_for_validation_set(  # reused from the sfsa_semi_markov_extras cell
        scenario, covariate_eval, base_seed=seed, n_matched=gt_n_matched, t_k=T_K, **hazard_kwargs
    )
    print(f"  done in {time.time() - t0:.1f}s.")

    # --- censoring KM, fit on the TRAINING split, for IPCW Brier-score weights ---
    censoring_kmf = KaplanMeierFitter()  # reused from the SFSA cell (lifelines)
    censoring_kmf.fit(cohort["duration"][train_idx], event_observed=1 - cohort["event"][train_idx])
    censoring_survival_fn = lambda t: censoring_kmf.survival_function_at_times(t).to_numpy()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    metrics_rows = []
    mae_curves = []
    predicted_demo_by_model = []

    for entry in registry:
        label = entry["label"]
        print(f"\n  -- {label} --")
        if label == "Glazier-Transformer":
            chunk_size = 20
        elif label in _SAMPLING_LABELS_SM:
            chunk_size = 50
        else:
            chunk_size = 500
        predicted = predict_curves_with_progress(
            entry["predict_fn"], covariate_eval, eval_times, label, chunk_size=chunk_size,
        )

        mae_t = mae_per_timestep(predicted, ground_truth)
        rmst = restricted_mean_survival_time(predicted, eval_times)
        risk_score = -rmst
        c_index, n_comparable = concordance_index_np(
            duration_eval, event_eval, risk_score, max_pairs_n=concordance_max_n, seed=seed
        )
        bs_t, ibs = integrated_brier_score(predicted[:, np.asarray(eval_times) < 30], np.asarray(eval_times)[np.asarray(eval_times) < 30], duration_eval, event_eval, censoring_survival_fn)

        row = {
            "Model": label,
            "Mean MAE": mae_t[np.asarray(eval_times) > 0].mean(),
            "Max MAE": mae_t.max(),
            "Integrated MAE (AUC)": _trapz(mae_t, eval_times),
            "MAE@t=10": mae_t[10] if len(mae_t) > 10 else float("nan"),
            "MAE@t=20": mae_t[20] if len(mae_t) > 20 else float("nan"),
            "MAE@t=30": mae_t[30] if len(mae_t) > 30 else float("nan"),
            "Concordance Index": c_index,
            "Integrated Brier Score": ibs,
            "N Eval": len(eval_idx),
            "N Comparable Pairs": n_comparable,
        }
        metrics_rows.append(row)
        mae_curves.append({"label": label, "color": entry["color"], "mae_t": mae_t})

        demo_pred = entry["predict_fn"](np.asarray(demo_values, dtype=np.float64), eval_times)
        predicted_demo_by_model.append({"label": label, "color": entry["color"], "curves": demo_pred})

        print(f"    Mean MAE={row['Mean MAE']:.4f}  Max MAE={row['Max MAE']:.4f}  "
              f"Integrated MAE={row['Integrated MAE (AUC)']:.4f}  "
              f"C-index={c_index:.4f}  IBS={ibs:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)

    # --- ground truth curves at the demo covariate values (for the survival-curve figure) ---
    ground_truth_demo = []
    for v in demo_values:
        _, sf_gt = ground_truth_km_sm(scenario, v, seed=seed, n_matched=max(gt_n_matched, 2000),
                                       t_k=T_K, **hazard_kwargs)
        ground_truth_demo.append(sf_gt)

    os.makedirs(out_dir, exist_ok=True)
    scenario_slug = scenario.name.lower()
    protocol_label = f"{scenario.name} (semi-Markov, prospective protocol)"

    print(f"\n  --- {protocol_label} comparison table ---")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}_sm_prospective.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    latex_cols = ["Model", "Mean MAE", "Max MAE", "Integrated MAE (AUC)", "Concordance Index", "Integrated Brier Score"]
    latex_str = to_latex_table(
        metrics_df[latex_cols],
        caption=f"Semi-Markov prospective-protocol model comparison on the {scenario.name} test set "
                f"(n={len(eval_idx)}).",
        label=f"tab:comparison_sm_prospective_{scenario_slug}",
    )
    tex_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}_sm_prospective.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"  Saved {tex_path} -- paste this directly into your Overleaf document:\n")
    print(latex_str)

    plot_mae_comparison(protocol_label, eval_times, mae_curves,
                         os.path.join(out_dir, f"fig_mae_sm_prospective_{scenario_slug}.png"))
    plot_survival_comparison(scenario, eval_times, demo_values, ground_truth_demo, predicted_demo_by_model,
                              os.path.join(out_dir, f"fig_survival_sm_prospective_{scenario_slug}.png"),
                              tag=" (semi-Markov, prospective)")
    plot_metric_bars(protocol_label, metrics_df,
                      os.path.join(out_dir, f"fig_metrics_sm_prospective_{scenario_slug}.png"))

    # --- explicit null-result sanity-check note (see module + function docstring) ---
    mae_spread = metrics_df["Mean MAE"].max() - metrics_df["Mean MAE"].min()
    print(
        f"\n  NOTE (documented sanity check, not a differentiation result): this is the PROSPECTIVE "
        f"protocol run -- every model predicts S(t) from t=0, before any individual has accumulated "
        f"an informative sojourn time. All {len(metrics_df)} models are EXPECTED to score similarly "
        f"here (current Mean-MAE spread across models: {mae_spread:.4f}); this run exists to confirm "
        f"the semi-Markov cohort behaves like an ordinary survival cohort absent landmark conditioning, "
        f"NOT to demonstrate that any model is better than another. The real test of the "
        f"information-asymmetry hypothesis this generator is built to expose is the LANDMARK protocol "
        f"(run_landmark_comparison_for_scenario_sm) below."
    )

    return {
        "metrics_df": metrics_df,
        "mae_curves": mae_curves,
        "ground_truth": ground_truth,
        "eval_times": eval_times,
        "eval_idx": eval_idx,
        "protocol": "prospective",
    }


# ===========================================================================
# PROTOCOL 2: LANDMARK comparison for one scenario ("S(t | covariate,
# OBSERVED states up to landmark step L)") -- THIS is the actual point of
# the semi-Markov generator (see module docstring). Closely mirrors
# model_comparison.py's run_landmark_comparison_for_scenario.
# ===========================================================================

# --- classical actuarial landmarking for the 5 discriminative models ------
# Copied VERBATIM from model_comparison.py. This is deliberate, not a
# limitation to "fix": none of CPH/DRSA/WTTE-RNN/Dynamic-DeepHit/
# SurvTransformer ever see the state sequence -- their whole input is
# [covariate, step_index] -- so the best any of them can structurally do at
# a landmark is S(t|x)/S(L|x), the classical actuarial conditioning formula.
# It depends only on the covariate and the single bit "still alive at the
# landmark," and is therefore IDENTICAL for two individuals sharing a
# covariate value regardless of how long either has actually sojourned in
# their current (possibly very different) observed state -- exactly the
# sojourn-time blindness this comparison is designed to expose.
def predict_survival_landmark_conditional_batch(predict_fn, covariate_values, landmark_step, eval_times, t_k=T_K):
    """
    S(t | covariate, alive at landmark) = S(t | covariate) / S(landmark | covariate),
    clipped to <= 1, forced to 1 for t <= landmark_step.
    """
    covariate_values = np.atleast_1d(np.asarray(covariate_values, dtype=np.float64))
    full_times = np.arange(0, t_k + 1)
    s_full = predict_fn(covariate_values, full_times)                     # (n, t_k+1)
    s_landmark = s_full[:, landmark_step]
    if np.any(~np.isfinite(s_landmark)) or np.any(s_landmark <= 0):
        raise ValueError('landmark ratio undefined where predicted S(L) is not positive')            # (n,)
    out = np.empty((len(covariate_values), len(eval_times)), dtype=np.float64)
    for j, t in enumerate(eval_times):
        t = int(t)
        if t <= landmark_step:
            out[:, j] = 1.0
        else:
            out[:, j] = np.minimum(s_full[:, t] / s_landmark, 1.0)
    return out


def run_landmark_comparison_for_scenario_sm(scenario, cohort_path, landmark_steps=LANDMARK_STEPS,
                                             n_eval=LANDMARK_N_EVAL, gt_n_matched=LANDMARK_GT_N_MATCHED,
                                             glazier_n_samples=LANDMARK_GLAZIER_N_SAMPLES,
                                             seed=DEMO_SEED, out_dir=OUT_DIR, hazard_kwargs=None):
    """
    For each landmark step L: restrict to test individuals actually alive
    (not yet Dead/Censored) at L, compute the semi-Markov landmark ground
    truth conditioned on each individual's OWN observed (state, sojourn) at
    L (`ground_truth_km_landmark_sm_for_validation_set`, which internally
    derives sojourn via `current_state_sojourn_time` -- both reused from the
    sfsa_semi_markov cell), score the 5 discriminative models via the
    sojourn-blind `predict_survival_landmark_conditional_batch` above, and
    score the 3 Glazier variants via their own NEW `*_landmark_sm` predict
    functions (one per variant -- see _GLAZIER_LANDMARK_FN_NAMES), which
    condition generation on each individual's TRUE observed prefix
    (`states_prefix_L`) instead of a blank START_TOKEN.

    THIS is the headline comparison: the line plot this function produces
    (Mean MAE vs. landmark step, one line per model) should show the 5
    discriminative models staying roughly flat/poor across landmark steps
    (structurally sojourn-blind) while the 3 Glazier models' error should
    drop as L increases (they see more of the true trajectory, hence more
    of the true sojourn-time signal the semi-Markov hazard ramp is
    sensitive to).

    COST WARNING: same shape as the base harness's landmark comparison --
    every landmark step re-pays each Glazier variant's generation cost, so
    the defaults here are deliberately smaller than the prospective
    protocol's (see LANDMARK_* constants).
    """
    if hazard_kwargs is None:
        hazard_kwargs = dict(DEFAULT_HAZARD_PARAMS)

    cohort = load_cohort_npz(cohort_path)
    test_idx = cohort["test_index"]

    if n_eval is not None and n_eval < len(test_idx):
        rng = np.random.default_rng(seed)
        base_idx = rng.choice(test_idx, size=n_eval, replace=False)
    else:
        base_idx = test_idx

    print(f"\n{'=' * 70}\n{scenario.name} scenario (semi-Markov, LANDMARK protocol)\n{'=' * 70}")

    registry = build_model_registry_sm(scenario, glazier_n_samples=glazier_n_samples, glazier_seed=seed)
    if not registry:
        print("  No trained semi-Markov models found -- nothing to compare.")
        return None

    g = globals()
    all_rows = []
    mae_vs_landmark = {entry["label"]: [] for entry in registry}

    for L in landmark_steps:
        current_state_at_L = cohort["states"][base_idx, L - 1]
        alive_mask = ~np.isin(current_state_at_L, (DEAD, CENSORED))  # reused from the SFSA cell
        alive_idx = base_idx[alive_mask]
        n_alive = len(alive_idx)
        print(f"\n  -- landmark L={L}: {n_alive}/{len(base_idx)} eval individuals still alive --")
        if n_alive == 0:
            print("     (none left alive at this landmark -- skipping)")
            continue

        covariate_L = cohort["covariate"][alive_idx]
        states_prefix_L = cohort["states"][alive_idx, :L]
        local_eval_times = np.arange(L + 1, T_K + 1)

        print(f"     Simulating semi-Markov landmark ground truth ({gt_n_matched} matched individuals each, "
              f"conditioned on each individual's OWN observed sojourn time)...")
        t0 = time.time()
        ground_truth_L = ground_truth_km_landmark_sm_for_validation_set(  # reused from the sfsa_semi_markov cell
            scenario, covariate_L, states_prefix_L, L, base_seed=seed, n_matched=gt_n_matched, t_k=T_K,
            **hazard_kwargs,
        )
        print(f"     done in {time.time() - t0:.1f}s.")
        ground_truth_scoped = ground_truth_L[:, L + 1:]  # score only t in (L, T_K]

        for entry in registry:
            label = entry["label"]
            if entry.get("is_sampling"):
                # Look up this variant's OWN landmark predict function by name
                # (NOT entry["predict_fn"], which is wrapped for the
                # prospective covariate+eval_times+n_samples+seed signature --
                # the landmark signature additionally needs each individual's
                # own observed prefix and the landmark step itself, which
                # must stay paired with the matching covariate value, so we
                # chunk both by hand rather than through
                # predict_curves_with_progress's single-array chunking).
                fn_name = _GLAZIER_LANDMARK_FN_NAMES[label]
                if fn_name not in g:
                    print(f"    [skip] {label}: '{fn_name}' not found -- run that model's sm cell first.")
                    continue
                landmark_pf = g[fn_name]
                chunk_size = 10 if label == "Glazier-Transformer" else 25
                n_this = len(covariate_L)
                predicted_scoped = np.empty((n_this, len(local_eval_times)))
                t0_chunk = time.time()
                for start in range(0, n_this, chunk_size):
                    end = min(start + chunk_size, n_this)
                    predicted_scoped[start:end] = landmark_pf(
                        entry["model_obj"], entry["meta_obj"],
                        covariate_L[start:end], states_prefix_L[start:end], L, local_eval_times,
                        n_samples=glazier_n_samples, seed=seed, t_k=T_K,
                    )
                    elapsed = time.time() - t0_chunk
                    rate = end / elapsed if elapsed > 0 else float("inf")
                    remaining = (n_this - end) / rate if rate > 0 else float("nan")
                    print(f"    [{label} (L={L})] {end}/{n_this} individuals predicted "
                          f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
            else:
                # Classical actuarial landmarking -- deliberately sojourn-blind, see above.
                predicted_scoped = predict_survival_landmark_conditional_batch(
                    entry["predict_fn"], covariate_L, L, local_eval_times, t_k=T_K,
                )

            mae_t_scoped = mae_per_timestep(predicted_scoped, ground_truth_scoped)
            row = {
                "Model": label, "Landmark": L, "N Alive": n_alive,
                "Mean MAE": mae_t_scoped.mean(), "Max MAE": mae_t_scoped.max(),
                "Integrated MAE (AUC)": _trapz(mae_t_scoped, local_eval_times),
            }
            all_rows.append(row)
            mae_vs_landmark[label].append((L, row["Mean MAE"]))
            print(f"    [{label}] Mean MAE={row['Mean MAE']:.4f}  Max MAE={row['Max MAE']:.4f}  "
                  f"Integrated MAE={row['Integrated MAE (AUC)']:.4f}")

    if not all_rows:
        print("  No landmark had any individuals still alive -- nothing to report.")
        return None

    landmark_df = pd.DataFrame(all_rows)
    os.makedirs(out_dir, exist_ok=True)
    scenario_slug = scenario.name.lower()

    print(f"\n  --- {scenario.name} semi-Markov landmark comparison table ---")
    print(landmark_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}_sm_landmark.csv")
    landmark_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    latex_str = to_latex_table(
        landmark_df[["Model", "Landmark", "N Alive", "Mean MAE", "Max MAE", "Integrated MAE (AUC)"]],
        caption=f"Semi-Markov landmark-protocol comparison on the {scenario.name} test set.",
        label=f"tab:comparison_sm_landmark_{scenario_slug}",
    )
    tex_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}_sm_landmark.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"  Saved {tex_path} -- paste this directly into your Overleaf document:\n")
    print(latex_str)

    # --- the headline figure: Mean MAE vs. landmark step, one line per model ---
    plt.figure(figsize=(7, 5))
    for entry in registry:
        label = entry["label"]
        pts = mae_vs_landmark.get(label, [])
        if not pts:
            continue
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=label, color=entry["color"])
    plt.xlabel("Landmark step L")
    plt.ylabel("Mean MAE over t in (L, T_K]")
    plt.title(f"Landmark-conditional Mean MAE vs. landmark depth -- {scenario.name} scenario (semi-Markov)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"fig_landmark_mae_sm_{scenario_slug}.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"  Saved {plot_path}")

    return {"landmark_df": landmark_df, "mae_vs_landmark": mae_vs_landmark, "protocol": "landmark"}


# ===========================================================================
# Combined prospective-vs.-landmark contrast summary -- the actual deliverable
# of running both protocols side by side (see module docstring). Computes,
# per model, how Mean MAE moves between the shallowest and deepest landmark
# step actually evaluated, and contrasts the discriminative-model group's
# average movement against the Glazier group's.
# ===========================================================================
def summarize_landmark_contrast(landmark_df, scenario_name):
    """Descriptive within-model changes; no predetermined family comparison."""
    lines=[f"Landmark results: {scenario_name}",
           "Changes combine different residual horizons and eligible populations; no isolated history or architecture effect is inferred."]
    for label, sub in landmark_df.groupby('Model'):
        sub=sub.sort_values('Landmark');first=sub.iloc[0];last=sub.iloc[-1]
        delta=float(last['Mean MAE']-first['Mean MAE'])
        lines.append(f"{label}: L={first['Landmark']} MAE={first['Mean MAE']:.4f}; L={last['Landmark']} MAE={last['Mean MAE']:.4f}; change={delta:+.4f}")
    text='\n'.join(lines);print(text);return text


# ===========================================================================
# Run both protocols for both scenarios (module-scope, Colab-cell style --
# no argparse/CLI, no if __name__ == "__main__" guard, matching this
# codebase's existing convention of "paste the whole file as one cell").
# ===========================================================================
if __name__ == '__main__':
    results_relu_sm_prospective = run_comparison_for_scenario_sm(RELU_SCENARIO, "/content/relu_cohort_sm.npz")
    results_weibull_sm_prospective = run_comparison_for_scenario_sm(WEIBULL_SCENARIO, "/content/weibull_cohort_sm.npz")

    if results_relu_sm_prospective is not None and results_weibull_sm_prospective is not None:
        combined_prospective_df = pd.concat(
            [results_relu_sm_prospective["metrics_df"].assign(Scenario="ReLU"),
             results_weibull_sm_prospective["metrics_df"].assign(Scenario="Weibull")],
            ignore_index=True,
        )
        combined_prospective_csv_path = os.path.join(OUT_DIR, "comparison_table_combined_sm_prospective.csv")
        combined_prospective_df.to_csv(combined_prospective_csv_path, index=False)
        print(f"\nSaved combined semi-Markov prospective table across both scenarios: {combined_prospective_csv_path}")

    results_relu_sm_landmark = run_landmark_comparison_for_scenario_sm(RELU_SCENARIO, "/content/relu_cohort_sm.npz")
    results_weibull_sm_landmark = run_landmark_comparison_for_scenario_sm(WEIBULL_SCENARIO, "/content/weibull_cohort_sm.npz")

    # --- the contrast IS the deliverable -- printed clearly, not buried in separate unrelated CSVs ---
    _summary_blocks = [
        "# Semi-Markov prospective vs. landmark protocol -- summary\n",
        "Prospective protocol (t=0, sanity check): all models are expected to score\n"
        "similarly, since no individual has an informative sojourn time yet -- see\n"
        "each scenario's printed NOTE above and the *_sm_prospective.csv tables.\n",
        "Landmark protocol (the actual test): the contrast below is the headline\n"
        "result this generator is built to produce.\n",
    ]
    if results_relu_sm_landmark is not None:
        _summary_blocks.append(summarize_landmark_contrast(results_relu_sm_landmark["landmark_df"], "ReLU"))
    if results_weibull_sm_landmark is not None:
        _summary_blocks.append(summarize_landmark_contrast(results_weibull_sm_landmark["landmark_df"], "Weibull"))

    _summary_path = os.path.join(OUT_DIR, "sm_comparison_summary.md")
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(_summary_path, "w") as f:
            f.write("\n\n".join(_summary_blocks) + "\n")
        print(f"\nSaved combined prospective-vs-landmark contrast summary: {_summary_path}")
    except OSError as exc:
        print(f"\n[warn] could not write {_summary_path}: {exc}")
