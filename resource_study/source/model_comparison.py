"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- numpy trapz/trapezoid compatibility (renamed in numpy>=2.0) ----------
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ===========================================================================
# CONFIGURATION -- the knobs you'll actually want to tune (see docstring)
# ===========================================================================
N_EVAL = None            # None = full test split; set an int (e.g. 500) to subsample for a faster run
GT_N_MATCHED = 1000       # matched-individual count per ground-truth KM curve
GLAZIER_N_SAMPLES = 150   # generated trajectories per individual, for EACH Glazier variant's predictions (paid 3x now)
CONCORDANCE_MAX_N = 2000  # concordance is an O(n^2) pairwise computation -- subsampled independently of N_EVAL
DEMO_SEED = 4242
OUT_DIR = "/content/"     # where PNGs / .tex files are saved


# ===========================================================================
# Loading a saved cohort (written by sfsa_data_generation.main())
# ===========================================================================
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


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


def restricted_mean_survival_time(S, eval_times):
    """Area under each individual's own predicted survival curve -- used
    as the summary "risk score" for the concordance index below (higher
    RMST = predicted to survive longer = lower risk)."""
    return _trapz(S, eval_times, axis=1)


# ===========================================================================
# Concordance index (Harrell's C, RMST-based) -- verified against a
# brute-force per-pair reference; see module docstring.
# ===========================================================================
def concordance_index_np(duration, event, risk_score, max_pairs_n=None, seed=0):
    """
    A pair (i, j) is "comparable" if individual i's observed duration is
    strictly shorter than j's AND i's shorter duration was an actual
    observed event (not censoring) -- the standard Harrell's-C definition.
    Concordant if the higher-risk individual (i) also has the higher risk
    score. Ties in risk score get half credit. O(n^2) -- `max_pairs_n`
    randomly subsamples first if the population is larger than that, since
    the pairwise comparison matrix would otherwise get very large.
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


# ===========================================================================
# Integrated Brier Score, IPCW-weighted (Graf et al. 1999) -- verified
# against a brute-force per-individual reference; see module docstring.
# ===========================================================================
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
# Model registry -- auto-detects which trained models exist in the
# notebook's global namespace (i.e. which cells you've already run) and
# wraps each one behind a uniform predict_fn(covariate_values, eval_times)
# signature, so everything below can treat every model identically.
#
# Two spec lists: _GENERIC_MODEL_SPECS covers discriminative, one-forward-
# pass models (DRSA, WTTE-RNN, Dynamic-DeepHit, SurvTransformer);
# _SAMPLING_MODEL_SPECS covers every GENERATIVE Glazier variant, which all
# need the extra n_samples/seed kwargs and share the same (much slower)
# generate-then-KM prediction recipe -- see each file's own docstring.
# ===========================================================================
_GENERIC_MODEL_SPECS = [
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
_SAMPLING_MODEL_SPECS = [
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
# Any label in here is treated as a slow, generative (sample-then-KM) model
# by predict_curves_with_progress below -- smaller progress-reporting chunks,
# since each individual is much more expensive than a discriminative model.
_SAMPLING_LABELS = {spec["label"] for spec in _SAMPLING_MODEL_SPECS}


def build_model_registry(scenario, glazier_n_samples=GLAZIER_N_SAMPLES, glazier_seed=DEMO_SEED):
    suffix = "relu" if scenario.name == "ReLU" else "weibull"
    g = globals()
    registry = []

    # CPH is a special case: it uses the generic `predict_survival` function
    # (from cph_model.py) plus a covariate_name kwarg, and has no separate
    # "meta" object the way the neural models do.
    cph_var = f"cph_{suffix}"
    if cph_var in g and "predict_survival" in g:
        cph_obj = g[cph_var]
        pf = g["predict_survival"]
        registry.append({
            "label": "CPH", "color": "tab:cyan",
            "predict_fn": lambda cov, t, _cph=cph_obj, _pf=pf: _pf(_cph, cov, t, covariate_name=scenario.covariate_name),
        })
    else:
        print(f"  [skip] CPH: '{cph_var}' or predict_survival not found in your notebook -- run the CPH cell first.")

    for spec in _GENERIC_MODEL_SPECS:
        model_var = f"{spec['model_prefix']}_{suffix}"
        meta_var = f"{spec['meta_prefix']}_{suffix}"
        fn_name = spec["predict_fn_name"]
        if model_var in g and meta_var in g and fn_name in g:
            model_obj, meta_obj, pf = g[model_var], g[meta_var], g[fn_name]
            registry.append({
                "label": spec["label"], "color": spec["color"],
                "predict_fn": lambda cov, t, _m=model_obj, _me=meta_obj, _pf=pf: _pf(_m, _me, cov, t),
            })
        else:
            print(f"  [skip] {spec['label']}: '{model_var}'/'{meta_var}' not found -- run that model's cell first.")

    for spec in _SAMPLING_MODEL_SPECS:
        model_var = f"{spec['model_prefix']}_{suffix}"
        meta_var = f"{spec['meta_prefix']}_{suffix}"
        fn_name = spec["predict_fn_name"]
        if model_var in g and meta_var in g and fn_name in g:
            model_obj, meta_obj, pf = g[model_var], g[meta_var], g[fn_name]
            registry.append({
                "label": spec["label"], "color": spec["color"],
                "predict_fn": lambda cov, t, _m=model_obj, _me=meta_obj, _pf=pf: _pf(
                    _m, _me, cov, t, n_samples=glazier_n_samples, seed=glazier_seed),
            })
        else:
            print(f"  [skip] {spec['label']}: '{model_var}'/'{meta_var}' not found -- run that model's cell first.")

    return registry


def predict_curves_with_progress(predict_fn, covariate_values, eval_times, label, chunk_size=500):
    """Calls predict_fn in chunks and prints running progress/ETA -- mainly
    useful for the Glazier variants, which are much slower than the
    discriminative models (see module docstring)."""
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
# LaTeX table generation -- plain tabular/hline formatting (no booktabs
# dependency required) so it pastes into any Overleaf document as-is.
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
# Plotting
# ===========================================================================
def plot_mae_comparison(scenario_label, eval_times, mae_curves, out_path):
    plt.figure(figsize=(7, 5))
    for entry in mae_curves:
        plt.plot(eval_times, entry["mae_t"], label=entry["label"], color=entry["color"])
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"MAE$_t$ comparison -- {scenario_label} scenario, full test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


def plot_survival_comparison(scenario, eval_times, demo_values, ground_truth_demo, predicted_demo_by_model, out_path):
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
    plt.suptitle(f"All models vs. ground truth -- {scenario.name} scenario")
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

    plt.suptitle(f"Summary metrics -- {scenario_label} scenario")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  Saved {out_path}")


# ===========================================================================
# End-to-end comparison for one scenario
# ===========================================================================
def run_comparison_for_scenario(scenario, cohort_path, cohort_train_for_censoring=None,
                                 n_eval=N_EVAL, gt_n_matched=GT_N_MATCHED,
                                 glazier_n_samples=GLAZIER_N_SAMPLES,
                                 concordance_max_n=CONCORDANCE_MAX_N,
                                 demo_values=None, seed=DEMO_SEED, out_dir=OUT_DIR):
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

    print(f"\n{'=' * 70}\n{scenario.name} scenario -- evaluating on {len(eval_idx)} test individuals\n{'=' * 70}")

    registry = build_model_registry(scenario, glazier_n_samples=glazier_n_samples, glazier_seed=seed)
    if not registry:
        print("  No trained models found in the notebook namespace -- nothing to compare. "
              "Run at least one model's cell before this one.")
        return None

    # --- ground truth (simulated), computed ONCE per scenario and shared across every model ---
    print(f"\n  Simulating ground-truth KM curves ({gt_n_matched} matched individuals each) "
          f"for {len(eval_idx)} test individuals -- this is the expensive shared step...")
    t0 = time.time()
    ground_truth = ground_truth_km_for_validation_set(  # reused from the SFSA cell
        scenario, covariate_eval, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
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
            chunk_size = 20   # slowest variant (O(T^2) generation) -- smaller chunks for more responsive progress prints
        elif label in _SAMPLING_LABELS:
            chunk_size = 50   # other Glazier variants -- generative, still much slower than discriminative models
        else:
            chunk_size = 500  # discriminative models -- one forward pass per individual
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
        _, sf_gt = ground_truth_km(scenario, v, seed=seed, n_matched=max(gt_n_matched, 2000), t_k=T_K)
        ground_truth_demo.append(sf_gt)

    os.makedirs(out_dir, exist_ok=True)
    scenario_slug = scenario.name.lower()

    print(f"\n  --- {scenario.name} comparison table ---")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    latex_cols = ["Model", "Mean MAE", "Max MAE", "Integrated MAE (AUC)", "Concordance Index", "Integrated Brier Score"]
    latex_str = to_latex_table(
        metrics_df[latex_cols],
        caption=f"Model comparison on the full {scenario.name} test set (n={len(eval_idx)}).",
        label=f"tab:comparison_{scenario_slug}",
    )
    tex_path = os.path.join(out_dir, f"comparison_table_{scenario_slug}.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"  Saved {tex_path} -- paste this directly into your Overleaf document:\n")
    print(latex_str)

    plot_mae_comparison(scenario.name, eval_times, mae_curves,
                         os.path.join(out_dir, f"mae_comparison_{scenario_slug}.png"))
    plot_survival_comparison(scenario, eval_times, demo_values, ground_truth_demo, predicted_demo_by_model,
                              os.path.join(out_dir, f"survival_comparison_{scenario_slug}.png"))
    plot_metric_bars(scenario.name, metrics_df,
                      os.path.join(out_dir, f"metric_bars_{scenario_slug}.png"))

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
    results_relu = run_comparison_for_scenario(RELU_SCENARIO, "/content/relu_cohort.npz")
    results_weibull = run_comparison_for_scenario(WEIBULL_SCENARIO, "/content/weibull_cohort.npz")

    if results_relu is not None and results_weibull is not None:
        combined_df = pd.concat(
            [results_relu["metrics_df"].assign(Scenario="ReLU"), results_weibull["metrics_df"].assign(Scenario="Weibull")],
            ignore_index=True,
        )
        combined_csv_path = os.path.join(OUT_DIR, "comparison_table_combined.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nSaved combined table across both scenarios: {combined_csv_path}")