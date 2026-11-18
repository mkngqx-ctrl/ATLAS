"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# Model -- copied VERBATIM from glazier_rnn_model.py. No architecture
# changes (`self.rnn = nn.RNN(...)`, same as the base file).
# ---------------------------------------------------------------------------
class GlazierRNNModel(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2, embed_size=8, n_states=5, nonlinearity="tanh"):
        super().__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.START_TOKEN = n_states
        self.embed = nn.Embedding(n_states + 1, embed_size)
        self.rnn = nn.RNN(input_size=embed_size + 1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, nonlinearity=nonlinearity)
        self.head = nn.Linear(hidden_size, n_states)

    def forward_teacher_forced(self, input_states, x_std):
        batch, T = input_states.shape
        emb = self.embed(input_states)
        x_rep = x_std.view(batch, 1, 1).expand(batch, T, 1)
        inp = torch.cat([emb, x_rep], dim=-1)
        out, _ = self.rnn(inp)
        return self.head(out)

    def step(self, prev_state, x_std, hidden):
        emb = self.embed(prev_state).unsqueeze(1)
        x_rep = x_std.view(-1, 1, 1)
        inp = torch.cat([emb, x_rep], dim=-1)
        out, hidden = self.rnn(inp, hidden)
        logits = self.head(out.squeeze(1))
        return logits, hidden


# ---------------------------------------------------------------------------
# Loss -- copied VERBATIM from glazier_rnn_model.py / glazier_model.py.
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
# Training -- copied VERBATIM from glazier_rnn_model.py. Same hyperparameter
# defaults; only the cohort passed in at the call site differs.
# ---------------------------------------------------------------------------
def train_glazier_rnn(covariate_train, states_train, covariate_test, states_test,
                       hidden_size=32, num_layers=2, embed_size=8,
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

    model = GlazierRNNModel(hidden_size=hidden_size, num_layers=num_layers,
                             embed_size=embed_size, n_states=n_states).to(device)
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
            # Vanilla RNNs are more exploding-gradient-prone than GRU/LSTM
            # over 30 steps -- gradient clipping matters more here, not less.
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
# Generation / prediction -- copied VERBATIM from glazier_rnn_model.py.
# ---------------------------------------------------------------------------
def generate_sequences_rnn(model, meta, covariate_value, n_samples, seed):
    device = meta["device"]
    T_k = meta["T_K"]
    x_std = (float(covariate_value) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.full((n_samples,), x_std, dtype=torch.float32, device=device)

    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        prev = torch.full((n_samples,), model.START_TOKEN, dtype=torch.long, device=device)
        hidden = None
        states_out = torch.empty((n_samples, T_k), dtype=torch.long, device=device)
        for t in range(T_k):
            logits, hidden = model.step(prev, x_t, hidden)
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = torch.multinomial(probs, num_samples=1, generator=cpu_gen).squeeze(1).to(device)
            states_out[:, t] = nxt
            prev = nxt

    return states_out.cpu().numpy().astype(np.int8)


def predict_survival_glazier_rnn(model, meta, covariate_values, eval_times, n_samples=500, seed=4242):
    covariate_values = np.atleast_1d(covariate_values)
    curves = np.empty((len(covariate_values), len(eval_times)), dtype=np.float64)
    for i, v in enumerate(covariate_values):
        gen_states = generate_sequences_rnn(model, meta, v, n_samples, seed=seed + i)
        duration, event = _extract_duration_event(gen_states, meta["T_K"])
        kmf = KaplanMeierFitter()
        kmf.fit(duration, event_observed=event)
        curves[i] = kmf.survival_function_at_times(eval_times).to_numpy()
    return curves


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ===========================================================================
# NEW: prefix-conditioning generation for the LANDMARK protocol. Identical
# in spirit and structure to `glazier_model_sm.py`'s
# `generate_sequences_from_prefix` -- only `model.step` dispatches to
# `GlazierRNNModel`'s own recurrence (`nn.RNN` rather than `nn.GRU`). See
# that file's (and this file's module) docstring for the full rationale.
# ===========================================================================
def generate_sequences_rnn_from_prefix(model, meta, covariate_value, observed_prefix_states,
                                        n_samples, seed):
    """
    observed_prefix_states: 1-D array of length L -- the TRUE states
    observed at absolute steps 1..L for ONE individual at this covariate
    value (every entry must be a live state; this individual must not
    already be Dead/Censored within the prefix).

    Returns an (n_samples, T_K - L) int8 array: the CONTINUATION only
    (steps L+1..T_K), sampled `n_samples` times from the model's learned
    dynamics conditioned on having actually observed this individual's true
    prefix.
    """
    device = meta["device"]
    T_k = meta["T_K"]
    observed_prefix_states = np.asarray(observed_prefix_states)
    L = len(observed_prefix_states)
    remaining = T_k - L
    x_std = (float(covariate_value) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.full((n_samples,), x_std, dtype=torch.float32, device=device)

    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        # Teacher-force through the TRUE observed prefix, forcing the KNOWN
        # true state at each step (never sampling) -- this is exactly what
        # "conditioning on an observed landmark" means. Logits computed
        # along the way are deliberately discarded; only `hidden` at the
        # end of this loop matters.
        prev = torch.full((n_samples,), model.START_TOKEN, dtype=torch.long, device=device)
        hidden = None
        for l in range(L):
            _logits, hidden = model.step(prev, x_t, hidden)
            forced = int(observed_prefix_states[l])
            prev = torch.full((n_samples,), forced, dtype=torch.long, device=device)

        # `hidden` now encodes this individual's true observed prefix;
        # continue sampling freely for the remaining T_K - L steps.
        out = torch.empty((n_samples, remaining), dtype=torch.long, device=device)
        for t in range(remaining):
            logits, hidden = model.step(prev, x_t, hidden)
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = torch.multinomial(probs, num_samples=1, generator=cpu_gen).squeeze(1).to(device)
            out[:, t] = nxt
            prev = nxt

    return out.cpu().numpy().astype(np.int8)


def predict_survival_glazier_rnn_landmark_sm(model, meta, covariate_values, observed_prefixes,
                                              landmark_step, eval_times, n_samples, seed, t_k=None):
    """
    Landmark-conditional prediction: for each individual, continue
    generation from their own TRUE observed prefix (via
    `generate_sequences_rnn_from_prefix`), prepend that true prefix to
    every sampled continuation to reconstruct the FULL (T_K,) trajectory,
    run `_extract_duration_event` on it, KM-fit, and evaluate at
    `eval_times`. Mirrors `predict_survival_glazier_landmark` in
    `model_comparison.py`, but calling this file's RNN-specific
    hidden-state-threading prefix function.

    observed_prefixes: (n, landmark_step) actual observed states per
    individual.
    """
    if t_k is None:
        t_k = meta["T_K"]
    covariate_values = np.atleast_1d(np.asarray(covariate_values, dtype=np.float64))
    observed_prefixes = np.asarray(observed_prefixes)
    assert observed_prefixes.shape[1] == landmark_step, (
        f"observed_prefixes has {observed_prefixes.shape[1]} columns, expected landmark_step={landmark_step}"
    )
    curves = np.empty((len(covariate_values), len(eval_times)), dtype=np.float64)
    for i, (v, prefix) in enumerate(zip(covariate_values, observed_prefixes)):
        continuation = generate_sequences_rnn_from_prefix(
            model, meta, v, prefix, n_samples, seed=seed + i
        )  # (n_samples, t_k - landmark_step)
        prefix_rep = np.tile(np.asarray(prefix, dtype=np.int8), (n_samples, 1))
        full_states = np.concatenate([prefix_rep, continuation], axis=1)  # (n_samples, t_k)
        duration, event = _extract_duration_event(full_states, t_k)
        kmf = KaplanMeierFitter()
        kmf.fit(duration, event_observed=event)
        curves[i] = kmf.survival_function_at_times(eval_times).to_numpy()
    return curves


# ---------------------------------------------------------------------------
# End-to-end PROSPECTIVE run for one scenario, trained on a semi-Markov
# cohort (mirrors glazier_rnn_model.py's run_glazier_rnn_for_scenario; only
# the ground-truth calls change, to the semi-Markov simulator's
# `ground_truth_km_sm`/`ground_truth_km_sm_for_validation_set`).
#
# NOTE (documented expectation, not a bug): at the PROSPECTIVE (t=0)
# protocol this is expected to show little/no advantage over the
# discriminative models -- every individual's sojourn time is trivially
# 0/1 at t=0. The landmark protocol (via
# `predict_survival_glazier_rnn_landmark_sm` above) is where this model's
# ability to condition on an observed prefix actually shows up.
# ---------------------------------------------------------------------------
def run_glazier_rnn_sm_for_scenario(scenario, cohort_path, n_mae_sample=100, gt_n_matched=2000,
                                     gen_n_samples=500, demo_values=None, seed=4242,
                                     hazard_kwargs=None, **train_kwargs):
    hazard_kwargs = dict(DEFAULT_HAZARD_PARAMS) if hazard_kwargs is None else dict(hazard_kwargs)
    cohort = load_cohort_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== Glazier-RNN fit on {scenario.name} semi-Markov scenario "
          f"(covariate: {scenario.covariate_name}) ===")
    model, meta = train_glazier_rnn(
        cohort["covariate"][train_idx], cohort["states"][train_idx],
        cohort["covariate"][test_idx], cohort["states"][test_idx],
        seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted = predict_survival_glazier_rnn(model, meta, sample_covariate, eval_times,
                                              n_samples=gen_n_samples, seed=seed)
    ground_truth = ground_truth_km_sm_for_validation_set(
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K, **hazard_kwargs
    )
    mae_t = mae_per_timestep(predicted, ground_truth)

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gen_n_samples} generated trajectories each vs. {gt_n_matched}-person semi-Markov ground truth each):")
    print(f"  mean MAE across all t: {mae_t.mean():.4f}")
    for t in (0, 10, 20, 30):
        print(f"  MAE at t={t}: {mae_t[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t, label="Glazier-RNN (semi-Markov)", color="tab:pink")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"Glazier-RNN MAE_t on {scenario.name} semi-Markov validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    sf_pred_all = predict_survival_glazier_rnn(model, meta, demo_values, eval_times,
                                                n_samples=max(gen_n_samples, 1000), seed=seed)
    fig, axes = plt.subplots(1, len(demo_values), figsize=(5 * len(demo_values), 4), sharey=True)
    demo_curves = {}
    for ax, v, sf_pred in zip(np.atleast_1d(axes), demo_values, sf_pred_all):
        t_gt, sf_gt = ground_truth_km_sm(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K, **hazard_kwargs)
        demo_curves[v] = {"ground_truth": sf_gt, "glazier_rnn": sf_pred}

        ax.step(t_gt, sf_gt, where="post", label="Ground truth (semi-Markov)", color="tab:red")
        ax.step(eval_times, sf_pred, where="post", label="Glazier-RNN", color="tab:pink")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.set_xlabel("Time")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("Survival probability")
    plt.suptitle(f"Glazier-RNN vs. semi-Markov ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    glazier_rnn_model_sm_relu, glazier_rnn_meta_sm_relu, eval_times_relu_glazier_rnn_sm, mae_t_relu_glazier_rnn_sm, demo_relu_glazier_rnn_sm = run_glazier_rnn_sm_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort_sm.npz",
    )
    glazier_rnn_model_sm_weibull, glazier_rnn_meta_sm_weibull, eval_times_weibull_glazier_rnn_sm, mae_t_weibull_glazier_rnn_sm, demo_weibull_glazier_rnn_sm = run_glazier_rnn_sm_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort_sm.npz",
    )
