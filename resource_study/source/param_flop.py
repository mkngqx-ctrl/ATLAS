"""
param_flop_comparison.py

Compares SurvTransformer (non-causal / full self-attention) and
Glazier-Transformer (causal self-attention) on two axes: parameter count
and computational cost (FLOPs). These are not the same comparison, and
the script reports both:

  1. PARAMETERS -- exact, via sum(p.numel() for p in model.parameters()).
     Run in Colab, on the actual trained model objects, so this reflects
     whatever d_model/nhead/num_layers/dim_feedforward you actually used,
     not assumed defaults.

  2. FLOPs -- SurvTransformer and Glazier-Transformer are used differently
     at prediction time, so "FLOPs per model" isn't one number each:
       - SurvTransformer: ONE non-causal forward pass over the full T-step
         sequence produces one PMF (see survtransformer_model.py). That's
         its whole prediction-time cost.
       - Glazier-Transformer: TRAINING is one causal forward pass over the
         full sequence (teacher-forced) -- architecturally comparable to
         SurvTransformer's single pass, same per-block cost formula. But
         GENERATION (what predict_survival_glazier_transformer actually
         runs) is T sequential forward passes over a growing sequence
         (length 1, 2, ..., T), because next_state_logits() re-runs the
         full causal forward pass from scratch every step -- there's no
         KV cache (see glazier_transformer_model.py's docstring on the
         O(T^2)-vs-O(T) generation cost vs. the GRU version). So this
         script reports THREE FLOP figures for Glazier-Transformer
         (training forward pass, one generation step, full generation)
         rather than pretending one number captures its cost.

FLOP CONVENTION: "1 FLOP" = one multiply OR one add (the common convention
in the efficiency literature, e.g. Kaplan et al. 2020's scaling-laws
paper), so a (batch x T x d_in) @ (d_in x d_out) matmul costs
2 * batch * T * d_in * d_out FLOPs, and attention's QK^T / softmax(.)@V
each cost 2 * batch * T_q * T_kv * d_model FLOPs (summed over all heads --
since nhead * head_dim = d_model, the per-head cost sums back up to a
d_model-wide computation regardless of how many heads it's split into).
These formulas were cross-checked against a brute-force nested-loop
multiply-add count for small toy sizes before shipping (see chat).

CAUSAL VS. MASKED SELF-ATTENTION, COMPUTATIONALLY: this is the practical
reason the causal-attention rewrite (glazier_transformer_model.py) can be
cheaper than the OLD masked-self-attention implementation it replaced, not
just "more correct." Masked self-attention (nn.TransformerEncoderLayer +
an explicit additive -inf mask) computes the FULL T x T attention-score
matrix and only zeroes out the disallowed upper triangle AFTERWARD -- the
QK^T and attn@V matmuls underneath still touch all T^2 query/key pairs, so
it gets NO FLOP savings from being causal, even though half its output is
thrown away. A genuinely causal attention kernel
(F.scaled_dot_product_attention(..., is_causal=True), what the current
model uses) *can*, on a backend that exploits it (e.g. FlashAttention),
skip the disallowed upper triangle entirely, touching only the T(T+1)/2
lower-triangular pairs -- a bit over half the dense attention-matrix FLOPs
for the T's used here. Whether that saving is actually realized depends on
which SDPA backend PyTorch selects at runtime (the "math" fallback
backend does NOT skip it; "flash"/"mem_efficient" do), so this script
reports BOTH figures side by side: "dense" (no savings -- matches masked
self-attention, and matches the causal model if the math backend is
selected) and "causal-optimized" (matches a backend that actually skips
the upper triangle).

HOW TO USE THIS IN COLAB:
Paste as a new cell after you've run the SurvTransformer and
Glazier-Transformer cells (needs survtransformer_model_relu and
glazier_transformer_model_relu -- or any other trained instances of
either class -- to already exist). Needs: numpy, matplotlib. Does not
need a fresh torch import (it only calls .parameters() and reads a few
attributes off the model objects you already have).
"""

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "/content/"


# ---------------------------------------------------------------------------
# Parameters -- exact, from the actual model object.
# ---------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# FLOPs -- closed-form formulas, introspected off the model's own submodule
# shapes (so this stays correct even if you changed d_model/nhead/etc. from
# the defaults) rather than hardcoded hyperparameters.
# ---------------------------------------------------------------------------
def _linear_flops(batch, T, in_features, out_features):
    """batch*T rows through an (in_features -> out_features) linear layer."""
    return 2 * batch * T * in_features * out_features


def _attention_core_flops(batch, T, d_model, causal_optimized=False):
    """QK^T + softmax(scores)@V for T-length self-attention. Summed over
    heads this only depends on d_model (nhead * head_dim = d_model), so
    nhead doesn't appear here. `causal_optimized=True` uses the exact
    T*(T+1)/2 lower-triangular pair count instead of the dense T*T (this is
    an exact formula, not a flat 0.5x approximation -- the true ratio is
    (T+1)/(2T), which is only asymptotically 0.5 as T grows)."""
    pairs = (T * (T + 1) // 2) if causal_optimized else (T * T)
    qk_flops = 2 * batch * pairs * d_model
    av_flops = 2 * batch * pairs * d_model
    return qk_flops + av_flops


def _block_flops(batch, T, d_model, dim_feedforward, causal=False, causal_optimized=False):
    """One self-attention block: QKV proj + attention core + output proj +
    2-layer MLP. Same shape of computation for both models' blocks (a
    TransformerEncoderLayer and a CausalTransformerBlock cost the same
    FLOPs for the same d_model/dim_feedforward/T -- the only difference is
    whether attention is causal, and whether that causality can be
    exploited for savings)."""
    qkv_proj = _linear_flops(batch, T, d_model, 3 * d_model)
    attn_core = _attention_core_flops(batch, T, d_model,
                                       causal_optimized=causal and causal_optimized)
    out_proj = _linear_flops(batch, T, d_model, d_model)
    mlp = (_linear_flops(batch, T, d_model, dim_feedforward)
           + _linear_flops(batch, T, dim_feedforward, d_model))
    return qkv_proj + attn_core + out_proj + mlp


# --- SurvTransformer -------------------------------------------------------
def _survtransformer_config(model):
    d_model = model.input_proj.out_features
    num_layers = len(model.encoder.layers)
    dim_feedforward = model.encoder.layers[0].linear1.out_features
    L = model.head.out_features
    return d_model, num_layers, dim_feedforward, L


def survtransformer_forward_flops(model, batch, T=None):
    """One non-causal forward pass over a length-T sequence, producing one
    PMF. This is SurvTransformer's whole prediction-time cost -- there's no
    separate "generation" phase (it's a one-shot discriminative predictor,
    not autoregressive)."""
    d_model, num_layers, dim_feedforward, L = _survtransformer_config(model)
    T = T if T is not None else L
    total = _linear_flops(batch, T, 2, d_model)  # input_proj: [covariate, step_idx] -> d_model, PER POSITION
    total += num_layers * _block_flops(batch, T, d_model, dim_feedforward, causal=False)
    total += 2 * batch * d_model * L  # head, applied ONCE after mean-pooling (not per-position)
    return total


# --- Glazier-Transformer ----------------------------------------------------
def _glazier_transformer_config(model):
    d_model = model.d_model
    num_layers = len(model.blocks)
    dim_feedforward = model.blocks[0].mlp[0].out_features
    n_states = model.n_states
    return d_model, num_layers, dim_feedforward, n_states


def _glazier_transformer_encode_flops(model, batch, T, causal_optimized=False):
    """Cost of _encode(): covariate_proj + the causal block stack. The
    covariate projection is applied ONCE per sequence (on a (batch,1,1)
    tensor) and then broadcast across all T positions via .expand() -- a
    free view, not a matmul -- unlike SurvTransformer's input_proj, which
    genuinely re-projects a distinct [covariate, step_idx] pair at every
    position. That's a small but real architectural difference: Glazier-
    Transformer's covariate conditioning is O(1) in T, not O(T)."""
    d_model, num_layers, dim_feedforward, _ = _glazier_transformer_config(model)
    total = 2 * batch * 1 * d_model  # covariate_proj: Linear(1, d_model) applied once, then broadcast
    total += num_layers * _block_flops(batch, T, d_model, dim_feedforward,
                                        causal=True, causal_optimized=causal_optimized)
    return total


def glazier_transformer_train_forward_flops(model, batch, T=None, causal_optimized=False):
    """forward_teacher_forced(): _encode() + head applied at EVERY position
    (next-state logits are supervised at every step during training)."""
    d_model, num_layers, dim_feedforward, n_states = _glazier_transformer_config(model)
    T = T if T is not None else model.pos_enc.pe.shape[1] - 1  # fallback if T not given
    total = _glazier_transformer_encode_flops(model, batch, T, causal_optimized=causal_optimized)
    total += _linear_flops(batch, T, d_model, n_states)  # head, PER POSITION
    return total


def _glazier_transformer_generation_step_flops(model, batch, t, causal_optimized=False):
    """next_state_logits() at generation step t: a full _encode() over the
    current length-t history, but the head is applied to ONLY the last
    position (not all t), since only the next state is needed."""
    d_model, num_layers, dim_feedforward, n_states = _glazier_transformer_config(model)
    total = _glazier_transformer_encode_flops(model, batch, t, causal_optimized=causal_optimized)
    total += 2 * batch * d_model * n_states  # head, ONE position only
    return total


def glazier_transformer_generation_flops(model, batch, T, causal_optimized=False):
    """Full autoregressive generation: T sequential steps over growing
    history lengths 1, 2, ..., T (no KV cache -- see module docstring).
    This is Glazier-Transformer's REAL inference cost, i.e. what
    predict_survival_glazier_transformer actually pays."""
    return sum(
        _glazier_transformer_generation_step_flops(model, batch, t, causal_optimized=causal_optimized)
        for t in range(1, T + 1)
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(n):
    n = float(n)
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def compare_models(survtransformer_model, glazier_transformer_model,
                    T=None, batch=1, n_gen_samples=150):
    """Prints and returns a comparison dict.

    T defaults to each model's own configured sequence length if not given.
    `batch` is the training-forward batch size to cost out; `n_gen_samples`
    defaults to GLAZIER_N_SAMPLES's value in model_comparison.py (150) --
    the number of trajectories actually generated per covariate value at
    prediction time (see predict_survival_glazier_transformer), so the
    "full generation" figure below reflects a realistic prediction-time
    cost, not just T=1.
    """
    surv_params = count_parameters(survtransformer_model)
    glazier_params = count_parameters(glazier_transformer_model)

    surv_fwd = survtransformer_forward_flops(survtransformer_model, batch, T)

    glazier_train_dense = glazier_transformer_train_forward_flops(
        glazier_transformer_model, batch, T, causal_optimized=False)
    glazier_train_causal = glazier_transformer_train_forward_flops(
        glazier_transformer_model, batch, T, causal_optimized=True)

    _, _, _, dim_ff_g = _glazier_transformer_config(glazier_transformer_model)
    T_glazier = T if T is not None else (glazier_transformer_model.pos_enc.pe.shape[1] - 1)
    glazier_gen_dense = glazier_transformer_generation_flops(
        glazier_transformer_model, n_gen_samples, T_glazier, causal_optimized=False)
    glazier_gen_causal = glazier_transformer_generation_flops(
        glazier_transformer_model, n_gen_samples, T_glazier, causal_optimized=True)

    rows = [
        ("Parameters", _fmt(surv_params), _fmt(glazier_params)),
        (f"FLOPs / forward pass (batch={batch}), dense", _fmt(surv_fwd), _fmt(glazier_train_dense)),
        (f"FLOPs / forward pass (batch={batch}), causal-optimized", "--", _fmt(glazier_train_causal)),
        (f"FLOPs / full generation (n={n_gen_samples} trajectories), dense", "n/a", _fmt(glazier_gen_dense)),
        (f"FLOPs / full generation (n={n_gen_samples} trajectories), causal-optimized", "n/a", _fmt(glazier_gen_causal)),
    ]

    label_w = max(len(r[0]) for r in rows) + 2
    print(f"{'Metric':<{label_w}} {'SurvTransformer':>18s} {'Glazier-Transformer':>20s}")
    print("-" * (label_w + 40))
    for label, a, b in rows:
        print(f"{label:<{label_w}} {a:>18s} {b:>20s}")
    print()
    print(f"Glazier-Transformer full generation vs. SurvTransformer's one forward pass "
          f"(dense): {glazier_gen_dense / surv_fwd:.1f}x more FLOPs")
    print(f"Glazier-Transformer training forward pass vs. SurvTransformer's forward pass "
          f"(dense): {glazier_train_dense / surv_fwd:.2f}x")

    return {
        "surv_params": surv_params,
        "glazier_params": glazier_params,
        "surv_forward_flops": surv_fwd,
        "glazier_train_forward_flops_dense": glazier_train_dense,
        "glazier_train_forward_flops_causal_optimized": glazier_train_causal,
        "glazier_generation_flops_dense": glazier_gen_dense,
        "glazier_generation_flops_causal_optimized": glazier_gen_causal,
        "n_gen_samples": n_gen_samples,
        "batch": batch,
        "T": T_glazier,
    }


# ---------------------------------------------------------------------------
# LaTeX table (Overleaf-pasteable, no booktabs dependency)
# ---------------------------------------------------------------------------
def _escape_latex(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def to_latex_param_flop_table(results, path=None, caption="Parameter count and computational cost: "
                               "SurvTransformer vs. Glazier-Transformer", label="tab:param_flop_comparison"):
    path = path or (OUT_DIR + "param_flop_comparison.tex")
    rows = [
        ("Parameters", _fmt(results["surv_params"]), _fmt(results["glazier_params"])),
        (f"FLOPs / forward pass (batch={results['batch']}, T={results['T']}), dense",
         _fmt(results["surv_forward_flops"]), _fmt(results["glazier_train_forward_flops_dense"])),
        (f"FLOPs / forward pass, causal-optimized",
         "--", _fmt(results["glazier_train_forward_flops_causal_optimized"])),
        (f"FLOPs / full generation (n={results['n_gen_samples']}), dense",
         "n/a", _fmt(results["glazier_generation_flops_dense"])),
        (f"FLOPs / full generation, causal-optimized",
         "n/a", _fmt(results["glazier_generation_flops_causal_optimized"])),
    ]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{|l|r|r|}")
    lines.append(r"\hline")
    lines.append(r"Metric & SurvTransformer & Glazier-Transformer \\")
    lines.append(r"\hline")
    for label_txt, a, b in rows:
        lines.append(f"{_escape_latex(label_txt)} & {_escape_latex(a)} & {_escape_latex(b)} \\\\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines)
    with open(path, "w") as f:
        f.write(tex)
    print(f"Saved LaTeX table to {path}")
    return tex


# ---------------------------------------------------------------------------
# Bar chart: parameters and FLOPs, log scale (magnitudes differ by orders)
# ---------------------------------------------------------------------------
def plot_param_flop_comparison(results, path=None):
    path = path or (OUT_DIR + "param_flop_comparison.png")
    labels = ["Parameters", "FLOPs/fwd pass\n(dense)", "FLOPs/full generation\n(dense)"]
    surv_vals = [results["surv_params"], results["surv_forward_flops"], np.nan]
    glazier_vals = [results["glazier_params"], results["glazier_train_forward_flops_dense"],
                     results["glazier_generation_flops_dense"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, surv_vals, width, label="SurvTransformer", color="tab:olive")
    ax.bar(x + width / 2, glazier_vals, width, label="Glazier-Transformer", color="tab:gray")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count (log scale)")
    ax.set_title("SurvTransformer vs. Glazier-Transformer: parameters and FLOPs")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved chart to {path}")


# ---------------------------------------------------------------------------
# Example usage (uncomment and run after training both models):
#
# results = compare_models(survtransformer_model_relu, glazier_transformer_model_relu)
# to_latex_param_flop_table(results)
# plot_param_flop_comparison(results)
# ---------------------------------------------------------------------------