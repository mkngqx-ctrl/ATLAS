"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


ALPHA = 0.5  # same loss-weighting role/default as the base drsa_model.py's ALPHA


# ---------------------------------------------------------------------------
# Loading a saved competing-risks cohort (written by generate_cohort_cr /
# train_test_val_split_cr in sfsa_competing_risks.py)
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
# Model: same LSTM backbone as DRSAModel, 3-way softmax head
# ---------------------------------------------------------------------------
class DRSACRModel(nn.Module):
    """
    Identical backbone to drsa_model.py's DRSAModel (LSTM reading
    [covariate, normalized time index] at each of L steps). The only change
    is the head: 3 logits per step instead of 1, softmax-normalized into
    [P(survive interval l), P(die of disease in l), P(die of other in l)]
    instead of a single sigmoid hazard.
    """

    def __init__(self, hidden_size=32, num_layers=1, L=30):
        super().__init__()
        self.L = L
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 3)  # logits: [survive, disease, other]

    def forward(self, x_std):
        # x_std: (batch,) standardized covariate values
        batch = x_std.shape[0]
        device = x_std.device
        l_idx = torch.arange(1, self.L + 1, device=device, dtype=torch.float32) / self.L
        x_rep = x_std.view(batch, 1, 1).expand(batch, self.L, 1)
        l_rep = l_idx.view(1, self.L, 1).expand(batch, self.L, 1)
        inp = torch.cat([x_rep, l_rep], dim=-1)  # (batch, L, 2)
        out, _ = self.lstm(inp)                  # (batch, L, hidden)
        probs = torch.softmax(self.head(out), dim=-1)  # (batch, L, 3)
        return probs


# ---------------------------------------------------------------------------
# Loss -- direct 3-way generalization of drsa_loss (see module docstring)
# ---------------------------------------------------------------------------
def drsa_cr_loss(probs, duration, cause, alpha=ALPHA, eps=1e-7):
    """
    probs: (batch, L, 3) softmax probabilities, columns [survive, disease, other].
    duration: (batch,) LongTensor, 1-indexed interval of death/censoring, in [1, L].
    cause: (batch,) LongTensor, 0 = censored, 1 = disease, 2 = other.
    """
    log_survive = torch.log(probs[..., 0].clamp(min=eps))
    cum_log_survive = torch.cumsum(log_survive, dim=1)  # cum_log_survive[:, l] = log S(l+1), l 0-indexed

    L = probs.shape[1]
    idx = (duration - 1).clamp(0, L - 1).long()
    b = torch.arange(probs.shape[0], device=probs.device)

    prev_idx = (idx - 1).clamp(min=0)
    log_S_prev = torch.where(idx > 0, cum_log_survive[b, prev_idx],
                              torch.zeros_like(idx, dtype=probs.dtype))
    log_S_at_idx = cum_log_survive[b, idx]

    is_event = (cause > 0)
    cause_col = cause.clamp(min=0).long()  # column 0 unused when censored; cause 1/2 already match head layout
    log_p_cause_at_idx = torch.log(probs[b, idx, cause_col].clamp(min=eps))

    Lz_terms = -(log_p_cause_at_idx + log_S_prev)
    Lz = (Lz_terms * is_event.float()).sum()

    S_at_idx = torch.exp(log_S_at_idx)
    Lunc = (-torch.log((1 - S_at_idx).clamp(min=eps)) * is_event.float()).sum()
    Lcens = (-log_S_at_idx * (~is_event).float()).sum()

    n = probs.shape[0]
    return alpha * (Lz / n) + (1 - alpha) * ((Lunc + Lcens) / n)


# ---------------------------------------------------------------------------
# Training (byte-identical loop structure to train_drsa, new loss/targets)
# ---------------------------------------------------------------------------
def train_drsa_cr(covariate_train, duration_train, cause_train,
                   covariate_test, duration_test, cause_test,
                   L=30, hidden_size=32, alpha=ALPHA, lr=1e-3, batch_size=256,
                   max_epochs=200, patience=15, seed=0, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    x_mean = float(np.mean(covariate_train))
    x_sd = float(np.std(covariate_train)) or 1.0

    def standardize(x):
        return (np.asarray(x, dtype=np.float64) - x_mean) / x_sd

    x_tr = torch.tensor(standardize(covariate_train), dtype=torch.float32, device=device)
    d_tr = torch.tensor(duration_train, dtype=torch.long, device=device)
    c_tr = torch.tensor(cause_train, dtype=torch.long, device=device)
    x_te = torch.tensor(standardize(covariate_test), dtype=torch.float32, device=device)
    d_te = torch.tensor(duration_test, dtype=torch.long, device=device)
    c_te = torch.tensor(cause_test, dtype=torch.long, device=device)

    model = DRSACRModel(hidden_size=hidden_size, L=L).to(device)
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
            xb, db, cb = x_tr[batch_idx], d_tr[batch_idx], c_tr[batch_idx]
            probs = model(xb)
            loss = drsa_cr_loss(probs, db, cb, alpha=alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            test_loss = drsa_cr_loss(model(x_te), d_te, c_te, alpha=alpha).item()

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
    return model, {"x_mean": x_mean, "x_sd": x_sd, "device": device, "L": L}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_cif_drsa(model, meta, covariate_values, eval_times):
    """
    Predict cause-specific CIFs at each integer time in `eval_times`
    (expected to be 0..L, matching np.arange(0, T_K+1) used elsewhere).
    Returns an array of shape (n_individuals, 2, len(eval_times)): axis
    index 0 = disease CIF, axis index 1 = other CIF. Non-negative,
    non-decreasing in t, disease+other <= 1 (+1e-6 floating slack), and 0 at
    eval_times[0] == 0.
    """
    device = meta["device"]
    x_std = (np.asarray(covariate_values, dtype=np.float64) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        probs = model(x_t).cpu().numpy()  # (n, L, 3)

    n, L, _ = probs.shape
    log_survive = np.log(np.clip(probs[..., 0], 1e-12, None))
    cum_log_survive = np.cumsum(log_survive, axis=1)          # log S(l), l=1..L (0-indexed l-1)
    S = np.exp(cum_log_survive)                               # (n, L) S at times 1..L
    S_full = np.concatenate([np.ones((n, 1)), S], axis=1)     # (n, L+1) S at times 0..L
    S_prev = S_full[:, :-1]                                   # S(l-1) for l=1..L, shape (n, L)

    pmf_disease = S_prev * probs[:, :, 1]
    pmf_other = S_prev * probs[:, :, 2]
    cif_disease = np.cumsum(pmf_disease, axis=1)              # (n, L), times 1..L
    cif_other = np.cumsum(pmf_other, axis=1)
    cif_disease_full = np.concatenate([np.zeros((n, 1)), cif_disease], axis=1)  # times 0..L
    cif_other_full = np.concatenate([np.zeros((n, 1)), cif_other], axis=1)

    eval_times = np.asarray(eval_times).astype(int)
    cif_d = cif_disease_full[:, eval_times]
    cif_o = cif_other_full[:, eval_times]
    return np.stack([cif_d, cif_o], axis=1)  # (n, 2, len(eval_times))


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors run_drsa_for_scenario, but
# compares against ground_truth_cif_for_validation_set / ground_truth_cif
# instead of the single-risk ground_truth_km equivalents)
# ---------------------------------------------------------------------------
def run_drsa_cr_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                              demo_values=None, seed=4242, **train_kwargs):
    cohort = load_cohort_cr_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== DRSA (competing risks) fit on {scenario.name} scenario "
          f"(covariate: {scenario.covariate_name}) ===")
    model, meta = train_drsa_cr(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["cause"][train_idx],
        cohort["covariate"][test_idx], cohort["duration"][test_idx], cohort["cause"][test_idx],
        L=T_K, seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted_cif = predict_cif_drsa(model, meta, sample_covariate, eval_times)  # (n, 2, T+1)
    predicted_disease, predicted_other = predicted_cif[:, 0, :], predicted_cif[:, 1, :]
    predicted_survival = 1.0 - predicted_disease - predicted_other

    ground_truth = ground_truth_cif_for_validation_set(  # defined in the sfsa_competing_risks.py cell
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
    )
    mae_t_survival = mae_per_timestep(predicted_survival, ground_truth["survival"])
    mae_t_disease = mae_per_timestep(predicted_disease, ground_truth["cif_disease"])
    mae_t_other = mae_per_timestep(predicted_other, ground_truth["cif_other"])

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gt_n_matched}-person ground truth each):")
    print(f"  mean MAE across all t -- survival: {mae_t_survival.mean():.4f}  "
          f"disease CIF: {mae_t_disease.mean():.4f}  other CIF: {mae_t_other.mean():.4f}")
    for t in (0, 10, 20, 30):
        print(f"  MAE at t={t} -- survival: {mae_t_survival[t]:.4f}  "
              f"disease: {mae_t_disease[t]:.4f}  other: {mae_t_other[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t_survival, label="Survival", color="tab:brown", linestyle="-")
    plt.plot(eval_times, mae_t_disease, label="Disease CIF", color="tab:brown", linestyle="--")
    plt.plot(eval_times, mae_t_other, label="Other CIF", color="tab:brown", linestyle=":")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"DRSA-CR MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    pred_cif_demo = predict_cif_drsa(model, meta, demo_values, eval_times)  # (n_demo, 2, T+1)
    fig, axes = plt.subplots(3, len(demo_values), figsize=(5 * len(demo_values), 11), sharex=True)
    axes = np.atleast_2d(axes).reshape(3, len(demo_values))
    demo_cif = {}
    for col, v in enumerate(demo_values):
        gt = ground_truth_cif(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        pred_disease = pred_cif_demo[col, 0, :]
        pred_other = pred_cif_demo[col, 1, :]
        pred_survival = 1.0 - pred_disease - pred_other
        demo_cif[v] = {
            "ground_truth_survival": gt["survival"], "ground_truth_disease": gt["cif_disease"],
            "ground_truth_other": gt["cif_other"],
            "pred_survival": pred_survival, "pred_disease": pred_disease, "pred_other": pred_other,
        }

        ax = axes[0, col]
        ax.step(eval_times, gt["survival"], where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, pred_survival, where="post", label="DRSA-CR", color="tab:brown")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.step(eval_times, gt["cif_disease"], where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, pred_disease, where="post", label="DRSA-CR", color="tab:brown")
        ax.legend(fontsize=8)

        ax = axes[2, col]
        ax.step(eval_times, gt["cif_other"], where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, pred_other, where="post", label="DRSA-CR", color="tab:brown")
        ax.set_xlabel("Time")
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel("Survival probability")
    axes[1, 0].set_ylabel("Disease CIF")
    axes[2, 0].set_ylabel("Other CIF")
    plt.suptitle(f"DRSA-CR vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t_disease, mae_t_other, demo_cif


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    (drsa_model_cr_relu, drsa_meta_cr_relu, eval_times_cr_relu_drsa,
     mae_t_disease_cr_relu_drsa, mae_t_other_cr_relu_drsa, demo_cr_relu_drsa) = run_drsa_cr_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort_cr.npz",
    )
    (drsa_model_cr_weibull, drsa_meta_cr_weibull, eval_times_cr_weibull_drsa,
     mae_t_disease_cr_weibull_drsa, mae_t_other_cr_weibull_drsa, demo_cr_weibull_drsa) = run_drsa_cr_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort_cr.npz",
    )
