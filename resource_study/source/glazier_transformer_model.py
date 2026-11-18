"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_cohort_npz(path):
    data = np.load(path)
    return {
        "covariate": data["covariate"],
        "states": data["states"],
        "duration": data["duration"],
        "event": data["event"],
        "train_index": data["train_index"],
        "test_index": data["test_index"],
        "val_index": data["val_index"],
    }


# Shared, byte-identical with survtransformer_model.py's copy -- harmless to
# redefine if both files are pasted into the same notebook.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


# ---------------------------------------------------------------------------
# Attention: a dedicated causal-attention primitive, not a bidirectional
# module plus a mask. `is_causal=True` is passed straight into the fused
# scaled_dot_product_attention kernel -- causality is native to the
# computation, not an additive correction applied to a general attention
# matrix after the fact.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.dropout_p = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (batch, T, d_model) -> (batch, T, d_model)."""
        batch, T, _ = x.shape
        qkv = self.qkv_proj(x)                                # (batch, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        # (batch, nhead, T, head_dim) -- split into heads for SDPA
        q = q.view(batch, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch, T, self.nhead, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,  # <-- causality is native to the kernel call, no mask tensor
        )                                                      # (batch, nhead, T, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, T, self.d_model)
        return self.out_proj(attn_out)


class CausalTransformerBlock(nn.Module):
    """GPT-style pre-LayerNorm decoder block: causal self-attention + MLP,
    each wrapped in a residual connection."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Model: causal (GPT-style) self-attention over the state history
# ---------------------------------------------------------------------------
class GlazierTransformerModel(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                 n_states=5, max_len=31, dropout=0.1):
        super().__init__()
        self.n_states = n_states
        self.d_model = d_model
        self.START_TOKEN = n_states  # same device as glazier_model.py -- a 6th embedding index
        self.embed = nn.Embedding(n_states + 1, d_model)
        self.covariate_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        # A stack of GPT-style causal decoder blocks -- see CausalSelfAttention
        # above. There is no separate "encoder" sequence to cross-attend to,
        # so this is self-attention only, same role nn.TransformerEncoder
        # played before, just built from a dedicated causal-attention kernel
        # instead of a bidirectional module plus an external mask.
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_states)

    def _encode(self, input_states, x_std):
        """input_states: (batch, T) long -- the *previous*-state sequence
        (START_TOKEN at position 0, same convention as glazier_model.py).
        Returns (batch, T, d_model)."""
        batch, T = input_states.shape
        emb = self.embed(input_states)                                        # (batch, T, d_model)
        x_rep = self.covariate_proj(x_std.view(batch, 1, 1)).expand(batch, T, self.d_model)
        h = emb + x_rep
        h = self.pos_enc(h)
        for block in self.blocks:
            h = block(h)                                                      # causal within each block
        h = self.ln_f(h)
        return h                                                              # (batch, T, d_model)

    def forward_teacher_forced(self, input_states, x_std):
        out = self._encode(input_states, x_std)
        return self.head(out)   # (batch, T, n_states) logits

    def next_state_logits(self, generated_so_far, x_std):
        """generated_so_far: (batch, t) long, the START_TOKEN plus every
        sampled state so far. Returns (batch, n_states) logits for the
        NEXT state, read off the last position of a fresh causal forward
        pass over the whole history."""
        out = self._encode(generated_so_far, x_std)
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------------------
# Loss -- copied verbatim from glazier_model.py (already validated there).
# ---------------------------------------------------------------------------
def glazier_loss(logits, target_states, input_states, dead_idx, censored_idx, eps=1e-9):
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(
        log_probs, dim=2, index=target_states.unsqueeze(-1)
    ).squeeze(-1)
    mask = ((input_states != dead_idx) & (input_states != censored_idx)).float()
    n_valid = mask.sum().clamp(min=1.0)
    return -(target_log_probs * mask).sum() / n_valid


# ---------------------------------------------------------------------------
# Training (teacher-forced, identical structure to glazier_model.py)
# ---------------------------------------------------------------------------
def train_glazier_transformer(covariate_train, states_train, covariate_test, states_test,
                               d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1,
                               lr=1e-3, batch_size=256, max_epochs=200, patience=15,
                               seed=0, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    n_states = N_STATES
    dead_idx, censored_idx = DEAD, CENSORED
    T_k = states_train.shape[1]

    x_mean = float(np.mean(covariate_train))
    x_sd = float(np.std(covariate_train)) or 1.0

    def standardize(x):
        return (np.asarray(x, dtype=np.float64) - x_mean) / x_sd

    def make_tensors(covariate, states):
        x = torch.tensor(standardize(covariate), dtype=torch.float32, device=device)
        target = torch.tensor(states, dtype=torch.long, device=device)
        start_col = torch.full((target.shape[0], 1), n_states, dtype=torch.long, device=device)
        input_states = torch.cat([start_col, target[:, :-1]], dim=1)
        return x, target, input_states

    x_tr, tgt_tr, inp_tr = make_tensors(covariate_train, states_train)
    x_te, tgt_te, inp_te = make_tensors(covariate_test, states_test)

    model = GlazierTransformerModel(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                     dim_feedforward=dim_feedforward, n_states=n_states,
                                     max_len=T_k + 1, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = x_tr.shape[0]
    best_test_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_train_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, tb, ib = x_tr[batch_idx], tgt_tr[batch_idx], inp_tr[batch_idx]
            logits = model.forward_teacher_forced(ib, xb)
            loss = glazier_loss(logits, tb, ib, dead_idx, censored_idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            logits_te = model.forward_teacher_forced(inp_te, x_te)
            test_loss = glazier_loss(logits_te, tgt_te, inp_te, dead_idx, censored_idx).item()

        improved = test_loss < best_test_loss - 1e-5
        if improved:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 10 == 0 or epochs_since_improve >= patience):
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} test_loss={test_loss:.4f}"
                  f"{'  (best)' if improved else ''}")

        if epochs_since_improve >= patience:
            print(f"  early stopping at epoch {epoch} (best test_loss={best_test_loss:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"x_mean": x_mean, "x_sd": x_sd, "device": device, "T_K": T_k}


# ---------------------------------------------------------------------------
# Generation -- unlike the GRU version, there is no persistent hidden state
# to carry: each step re-runs the full causal forward pass over the whole
# history generated so far (see module docstring re: O(T^2) cost).
# ---------------------------------------------------------------------------
def generate_sequences_transformer(model, meta, covariate_value, n_samples, seed):
    device = meta["device"]
    T_k = meta["T_K"]
    x_std = (float(covariate_value) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.full((n_samples,), x_std, dtype=torch.float32, device=device)

    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        generated = torch.full((n_samples, 1), model.START_TOKEN, dtype=torch.long, device=device)
        states_out = torch.empty((n_samples, T_k), dtype=torch.long, device=device)
        for t in range(T_k):
            logits = model.next_state_logits(generated, x_t)             # (n_samples, n_states)
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = torch.multinomial(probs, num_samples=1, generator=cpu_gen).squeeze(1).to(device)
            states_out[:, t] = nxt
            generated = torch.cat([generated, nxt.unsqueeze(1)], dim=1)   # grow the history by one step

    return states_out.cpu().numpy().astype(np.int8)


def predict_survival_glazier_transformer(model, meta, covariate_values, eval_times, n_samples=500, seed=4242):
    covariate_values = np.atleast_1d(covariate_values)
    curves = np.empty((len(covariate_values), len(eval_times)), dtype=np.float64)
    for i, v in enumerate(covariate_values):
        gen_states = generate_sequences_transformer(model, meta, v, n_samples, seed=seed + i)
        duration, event = _extract_duration_event(gen_states, meta["T_K"])
        kmf = KaplanMeierFitter()
        kmf.fit(duration, event_observed=event)
        curves[i] = kmf.survival_function_at_times(eval_times).to_numpy()
    return curves


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors glazier_model.py's runner)
# ---------------------------------------------------------------------------
def run_glazier_transformer_for_scenario(scenario, cohort_path, n_mae_sample=100, gt_n_matched=2000,
                                          gen_n_samples=500, demo_values=None, seed=4242, **train_kwargs):
    cohort = load_cohort_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== Glazier-Transformer fit on {scenario.name} scenario (covariate: {scenario.covariate_name}) ===")
    model, meta = train_glazier_transformer(
        cohort["covariate"][train_idx], cohort["states"][train_idx],
        cohort["covariate"][test_idx], cohort["states"][test_idx],
        seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted = predict_survival_glazier_transformer(model, meta, sample_covariate, eval_times,
                                                       n_samples=gen_n_samples, seed=seed)
    ground_truth = ground_truth_km_for_validation_set(
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
    )
    mae_t = mae_per_timestep(predicted, ground_truth)

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gen_n_samples} generated trajectories each vs. {gt_n_matched}-person ground truth each):")
    print(f"  mean MAE across all t: {mae_t.mean():.4f}")
    for t in (0, 10, 20, 30):
        print(f"  MAE at t={t}: {mae_t[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t, label="Glazier-Transformer", color="tab:gray")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"Glazier-Transformer MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    sf_pred_all = predict_survival_glazier_transformer(model, meta, demo_values, eval_times,
                                                         n_samples=max(gen_n_samples, 1000), seed=seed)
    fig, axes = plt.subplots(1, len(demo_values), figsize=(5 * len(demo_values), 4), sharey=True)
    demo_curves = {}
    for ax, v, sf_pred in zip(np.atleast_1d(axes), demo_values, sf_pred_all):
        t_gt, sf_gt = ground_truth_km(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        demo_curves[v] = {"ground_truth": sf_gt, "glazier_transformer": sf_pred}

        ax.step(t_gt, sf_gt, where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, sf_pred, where="post", label="Glazier-Transformer", color="tab:gray")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.set_xlabel("Time")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("Survival probability")
    plt.suptitle(f"Glazier-Transformer vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    glazier_transformer_model_relu, glazier_transformer_meta_relu, eval_times_relu_glazier_transformer, mae_t_relu_glazier_transformer, demo_relu_glazier_transformer = run_glazier_transformer_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort.npz",
    )
    glazier_transformer_model_weibull, glazier_transformer_meta_weibull, eval_times_weibull_glazier_transformer, mae_t_weibull_glazier_transformer, demo_weibull_glazier_transformer = run_glazier_transformer_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort.npz",
    )