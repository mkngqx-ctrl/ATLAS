"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading a saved competing-risks cohort (written by an
# sfsa_competing_risks.py-based generation cell) -- needs the raw `states`
# sequences, not just (duration, cause).
# ---------------------------------------------------------------------------
def load_cohort_cr_npz(path):
    data = np.load(path)
    return {
        "covariate": data["covariate"],
        "states": data["states"],
        "duration": data["duration"],
        "cause": data["cause"],
        "train_index": data["train_index"],
        "test_index": data["test_index"],
        "val_index": data["val_index"],
    }


# ---------------------------------------------------------------------------
# Model -- identical to glazier_model.py's GlazierModel, just with a 6-entry
# (not 5-entry) state space. See module docstring point 1.
# ---------------------------------------------------------------------------
class GlazierCRModel(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2, embed_size=8, n_states=6):
        super().__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.START_TOKEN = n_states  # a 7th embedding index, distinct from real states 0..5
        self.embed = nn.Embedding(n_states + 1, embed_size)
        self.gru = nn.GRU(input_size=embed_size + 1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_states)

    def forward_teacher_forced(self, input_states, x_std):
        """input_states: (batch, T) LongTensor of *previous*-state indices
        (already shifted, START_TOKEN at position 0). x_std: (batch,) float.
        Returns logits (batch, T, n_states)."""
        batch, T = input_states.shape
        emb = self.embed(input_states)                       # (batch, T, embed_size)
        x_rep = x_std.view(batch, 1, 1).expand(batch, T, 1)
        inp = torch.cat([emb, x_rep], dim=-1)                 # (batch, T, embed_size+1)
        out, _ = self.gru(inp)                                # (batch, T, hidden)
        return self.head(out)                                 # (batch, T, n_states)

    def step(self, prev_state, x_std, hidden):
        """One generation step. prev_state: (batch,) long. x_std: (batch,)
        float. hidden: (num_layers, batch, hidden) or None. Returns
        (logits (batch, n_states), new_hidden)."""
        emb = self.embed(prev_state).unsqueeze(1)             # (batch, 1, embed_size)
        x_rep = x_std.view(-1, 1, 1)
        inp = torch.cat([emb, x_rep], dim=-1)                 # (batch, 1, embed_size+1)
        out, hidden = self.gru(inp, hidden)                   # out: (batch, 1, hidden)
        logits = self.head(out.squeeze(1))                    # (batch, n_states)
        return logits, hidden


# ---------------------------------------------------------------------------
# Loss -- glazier_loss widened from a 2-way to a 3-way absorbing-state mask
# (DEAD_DISEASE, DEAD_OTHER, CENSORED instead of DEAD, CENSORED). The
# gather/log-softmax machinery is otherwise byte-identical to glazier_loss,
# so the same brute-force NumPy verification from glazier_model.py's
# docstring carries over directly -- only the mask's boolean expression
# changed, and masking a third absorbing state out can only ever remove
# additional all-repeat, zero-information steps from the loss, exactly the
# same way the 2-way mask already did.
# ---------------------------------------------------------------------------
def glazier_cr_loss(logits, target_states, input_states,
                     dead_disease_idx, dead_other_idx, censored_idx, eps=1e-9):
    """
    logits: (batch, T, n_states) raw scores. target_states: (batch, T) long,
    the real state at each step. input_states: (batch, T) long, the real
    *previous* state fed in at each step (used only to build the mask).
    Steps whose input state is already DEAD_DISEASE, DEAD_OTHER, or
    CENSORED are masked out (see module docstring for why).
    """
    batch, T, n_states = logits.shape
    log_probs = torch.log_softmax(logits, dim=-1)                       # (batch, T, n_states)
    target_log_probs = torch.gather(
        log_probs, dim=2, index=target_states.unsqueeze(-1)
    ).squeeze(-1)                                                        # (batch, T)

    mask = (
        (input_states != dead_disease_idx)
        & (input_states != dead_other_idx)
        & (input_states != censored_idx)
    ).float()
    n_valid = mask.sum().clamp(min=1.0)
    return -(target_log_probs * mask).sum() / n_valid


# ---------------------------------------------------------------------------
# Training -- byte-identical structure to glazier_model.py's train_glazier,
# just n_states = N_STATES_CR, three absorbing indices, and glazier_cr_loss.
# ---------------------------------------------------------------------------
def train_glazier_cr(covariate_train, states_train, covariate_test, states_test,
                      hidden_size=32, num_layers=2, embed_size=8,
                      lr=1e-3, batch_size=256, max_epochs=200, patience=15,
                      seed=0, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    n_states = N_STATES_CR  # reused from the SFSA-competing-risks cell already in this notebook
    dead_disease_idx, dead_other_idx, censored_idx = DEAD_DISEASE, DEAD_OTHER, CENSORED
    T_k = states_train.shape[1]

    x_mean = float(np.mean(covariate_train))
    x_sd = float(np.std(covariate_train)) or 1.0

    def standardize(x):
        return (np.asarray(x, dtype=np.float64) - x_mean) / x_sd

    def make_tensors(covariate, states):
        x = torch.tensor(standardize(covariate), dtype=torch.float32, device=device)
        target = torch.tensor(states, dtype=torch.long, device=device)  # (n, T_k)
        start_col = torch.full((target.shape[0], 1), n_states, dtype=torch.long, device=device)
        input_states = torch.cat([start_col, target[:, :-1]], dim=1)     # (n, T_k), shifted
        return x, target, input_states

    x_tr, tgt_tr, inp_tr = make_tensors(covariate_train, states_train)
    x_te, tgt_te, inp_te = make_tensors(covariate_test, states_test)

    model = GlazierCRModel(hidden_size=hidden_size, num_layers=num_layers,
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
            loss = glazier_cr_loss(logits, tb, ib, dead_disease_idx, dead_other_idx, censored_idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            logits_te = model.forward_teacher_forced(inp_te, x_te)
            test_loss = glazier_cr_loss(logits_te, tgt_te, inp_te,
                                         dead_disease_idx, dead_other_idx, censored_idx).item()

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
# Generation -- byte-identical structure to glazier_model.py's
# generate_sequences (entirely agnostic to how many states there are).
# ---------------------------------------------------------------------------
def generate_sequences_cr(model, meta, covariate_value, n_samples, seed):
    """Autoregressively sample `n_samples` full (T_K,) state trajectories
    over the 6-state competing-risks state space, all sharing
    `covariate_value`. Returns an (n_samples, T_K) int array, directly
    compatible with `_extract_duration_cause` from the SFSA-competing-risks
    cell."""
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
            probs = torch.softmax(logits, dim=-1).cpu()          # multinomial sampling done on CPU for a reproducible generator
            nxt = torch.multinomial(probs, num_samples=1, generator=cpu_gen).squeeze(1).to(device)
            states_out[:, t] = nxt
            prev = nxt

    return states_out.cpu().numpy().astype(np.int8)


def predict_cif_glazier(model, meta, covariate_values, eval_times, n_samples=500, seed=4242):
    """For each value in `covariate_values`, generate `n_samples` synthetic
    trajectories and run Aalen-Johansen (`aalen_johansen_cif`, reused
    verbatim from sfsa_competing_risks.py) on their (duration, cause) pairs
    to get the predicted overall survival curve and both cause-specific
    CIFs -- the competing-risks analogue of the base file's
    "simulate + Kaplan-Meier" recipe (see module docstring).

    `eval_times` is expected to be (a subset/permutation of) the integer
    grid `np.arange(0, meta["T_K"] + 1)` -- `aalen_johansen_cif`'s own
    `eval_times` output is exactly that grid, so indexing its `cif` array by
    the integer values of `eval_times` is a direct lookup, not an
    interpolation.

    Returns an array of shape (len(covariate_values), 2, len(eval_times));
    axis 1 index 0 = disease CIF, index 1 = other-cause CIF.
    """
    covariate_values = np.atleast_1d(covariate_values)
    eval_times = np.asarray(eval_times)
    eval_idx = eval_times.astype(int)
    curves = np.empty((len(covariate_values), 2, len(eval_times)), dtype=np.float64)
    for i, v in enumerate(covariate_values):
        gen_states = generate_sequences_cr(model, meta, v, n_samples, seed=seed + i)
        duration, cause = _extract_duration_cause(gen_states, meta["T_K"])  # reused from SFSA-CR cell
        _, _, cif = aalen_johansen_cif(duration, cause, t_k=meta["T_K"], n_causes=2)  # reused from SFSA-CR cell
        curves[i] = cif[:, eval_idx]
    return curves


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors glazier_model.py's runner, scored
# against Aalen-Johansen ground truth instead of Kaplan-Meier ground truth)
# ---------------------------------------------------------------------------
def run_glazier_cr_for_scenario(scenario, cohort_path, n_mae_sample=100, gt_n_matched=2000,
                                 gen_n_samples=500, demo_values=None, seed=4242, **train_kwargs):
    cohort = load_cohort_cr_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== Glazier-GRU (competing risks) fit on {scenario.name} scenario "
          f"(covariate: {scenario.covariate_name}) ===")
    model, meta = train_glazier_cr(
        cohort["covariate"][train_idx], cohort["states"][train_idx],
        cohort["covariate"][test_idx], cohort["states"][test_idx],
        seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    # NOTE: generation is much slower than a closed-form predictive head
    # (T_K sequential GRU steps x n_samples x n_mae_sample individuals), so
    # n_mae_sample and gen_n_samples default lower here, matching
    # glazier_model.py's own runner -- raise them once you've confirmed
    # this runs.
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted = predict_cif_glazier(model, meta, sample_covariate, eval_times,
                                     n_samples=gen_n_samples, seed=seed)
    pred_disease, pred_other = predicted[:, 0, :], predicted[:, 1, :]
    pred_survival = 1.0 - pred_disease - pred_other

    ground_truth = ground_truth_cif_for_validation_set(
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
    )
    mae_t_survival = mae_per_timestep(pred_survival, ground_truth["survival"])
    mae_t_disease = mae_per_timestep(pred_disease, ground_truth["cif_disease"])
    mae_t_other = mae_per_timestep(pred_other, ground_truth["cif_other"])

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gen_n_samples} generated trajectories each vs. {gt_n_matched}-person Aalen-Johansen "
          f"ground truth each):")
    print(f"  mean MAE across all t -- survival: {mae_t_survival.mean():.4f}, "
          f"disease CIF: {mae_t_disease.mean():.4f}, other CIF: {mae_t_other.mean():.4f}")
    for t in (0, 10, 20, 30):
        print(f"  MAE at t={t} -- survival: {mae_t_survival[t]:.4f}, "
              f"disease CIF: {mae_t_disease[t]:.4f}, other CIF: {mae_t_other[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t_survival, label="Survival", color="tab:purple")
    plt.plot(eval_times, mae_t_disease, label="Disease CIF", color="tab:red", linestyle="--")
    plt.plot(eval_times, mae_t_other, label="Other CIF", color="tab:blue", linestyle=":")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"Glazier-GRU (competing risks) MAE_t on {scenario.name} validation subsample "
              f"(n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    pred_demo_all = predict_cif_glazier(model, meta, demo_values, eval_times,
                                         n_samples=max(gen_n_samples, 1000), seed=seed)
    fig, axes = plt.subplots(3, len(demo_values), figsize=(5 * len(demo_values), 11), sharex=True)
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 3:
        axes = axes.T
    demo_curves = {}
    row_labels = ("Overall survival", "Disease CIF", "Other-cause CIF")
    for col, (v, pred_cif) in enumerate(zip(demo_values, pred_demo_all)):
        gt = ground_truth_cif(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        pred_disease_v, pred_other_v = pred_cif[0], pred_cif[1]
        pred_survival_v = 1.0 - pred_disease_v - pred_other_v
        demo_curves[v] = {
            "ground_truth": {"survival": gt["survival"], "cif_disease": gt["cif_disease"],
                              "cif_other": gt["cif_other"]},
            "glazier": {"survival": pred_survival_v, "cif_disease": pred_disease_v,
                        "cif_other": pred_other_v},
        }

        gt_curves = (gt["survival"], gt["cif_disease"], gt["cif_other"])
        pred_curves = (pred_survival_v, pred_disease_v, pred_other_v)
        for row in range(3):
            ax = axes[row, col]
            ax.step(eval_times, gt_curves[row], where="post", label="Ground truth", color="tab:red")
            ax.step(eval_times, pred_curves[row], where="post", label="Glazier-GRU", color="tab:purple")
            if row == 0:
                ax.set_title(f"{scenario.covariate_name}={v}")
            if row == 2:
                ax.set_xlabel("Time")
            ax.legend(fontsize=8)
    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row])
    plt.suptitle(f"Glazier-GRU (competing risks) vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t_survival, mae_t_disease, mae_t_other, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    (glazier_model_cr_relu, glazier_meta_cr_relu, eval_times_relu_glazier_cr,
     mae_t_survival_relu_glazier_cr, mae_t_disease_relu_glazier_cr, mae_t_other_relu_glazier_cr,
     demo_relu_glazier_cr) = run_glazier_cr_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort_cr.npz",
    )
    (glazier_model_cr_weibull, glazier_meta_cr_weibull, eval_times_weibull_glazier_cr,
     mae_t_survival_weibull_glazier_cr, mae_t_disease_weibull_glazier_cr, mae_t_other_weibull_glazier_cr,
     demo_weibull_glazier_cr) = run_glazier_cr_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort_cr.npz",
    )
