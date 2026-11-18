"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


ALPHA = 0.5  # DRSA loss weighting between L_z and (L_uncensored + L_censored); not specified by the paper


# ---------------------------------------------------------------------------
# Loading a saved cohort (written by sfsa_data_generation.main())
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DRSAModel(nn.Module):
    """
    At every one of the L discrete intervals, the LSTM sees [covariate,
    normalized time index] and outputs one hazard probability for that
    interval. The covariate is repeated at every step (it's static in our
    data, per paper Section 3.2: "we simplify the simulation to one
    covariate only") -- the "recurrent" part is purely about walking through
    time, accumulating a hidden state as it goes, not about time-varying
    features.
    """

    def __init__(self, hidden_size=32, num_layers=1, L=30):
        super().__init__()
        self.L = L
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_std):
        # x_std: (batch,) standardized covariate values
        batch = x_std.shape[0]
        device = x_std.device
        l_idx = torch.arange(1, self.L + 1, device=device, dtype=torch.float32) / self.L
        x_rep = x_std.view(batch, 1, 1).expand(batch, self.L, 1)
        l_rep = l_idx.view(1, self.L, 1).expand(batch, self.L, 1)
        inp = torch.cat([x_rep, l_rep], dim=-1)  # (batch, L, 2)
        out, _ = self.lstm(inp)                  # (batch, L, hidden)
        y = torch.sigmoid(self.head(out)).squeeze(-1)  # (batch, L) hazards for intervals 1..L
        return y


# ---------------------------------------------------------------------------
# Loss (verified against a brute-force NumPy reference -- see module docstring)
# ---------------------------------------------------------------------------
def drsa_loss(y, duration, event, alpha=ALPHA, eps=1e-7):
    """
    y: (batch, L) hazards, one row per individual.
    duration: (batch,) LongTensor, 1-indexed interval of death/censoring, in [1, L].
    event: (batch,) LongTensor/FloatTensor, 1 = died, 0 = censored.
    """
    y = y.clamp(min=eps, max=1 - eps)
    log_y = torch.log(y)
    log_1my = torch.log(1 - y)
    cum_log_1my = torch.cumsum(log_1my, dim=1)  # cum_log_1my[:, l] = sum_{k=0}^{l} log(1-y_k)

    L = y.shape[1]
    idx = (duration - 1).clamp(0, L - 1).long()
    idx_col = idx.view(-1, 1)

    log_S_at_idx = torch.gather(cum_log_1my, 1, idx_col).squeeze(1)
    log_y_at_idx = torch.gather(log_y, 1, idx_col).squeeze(1)

    prev_idx = (idx - 1).clamp(min=0)
    log_S_prev = torch.gather(cum_log_1my, 1, prev_idx.view(-1, 1)).squeeze(1)
    log_S_prev = torch.where(idx > 0, log_S_prev, torch.zeros_like(log_S_prev))

    event_f = event.float()

    Lz_terms = -(log_y_at_idx + log_S_prev)
    Lz = (Lz_terms * event_f).sum()

    S_at_idx = torch.exp(log_S_at_idx)
    Lunc_terms = -torch.log((1 - S_at_idx).clamp(min=eps))
    Lunc = (Lunc_terms * event_f).sum()

    Lcens_terms = -log_S_at_idx
    Lcens = (Lcens_terms * (1 - event_f)).sum()

    n = y.shape[0]
    total = alpha * (Lz / n) + (1 - alpha) * ((Lunc + Lcens) / n)
    return total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_drsa(covariate_train, duration_train, event_train,
                covariate_test, duration_test, event_test,
                L=30, hidden_size=32, alpha=ALPHA, lr=1e-3, batch_size=256,
                max_epochs=200, patience=15, seed=0, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Standardize the covariate -- an LSTM trains much better on roughly
    # unit-scale inputs than on raw age (~30-120) or calcium (~7-13) values.
    x_mean = float(np.mean(covariate_train))
    x_sd = float(np.std(covariate_train)) or 1.0

    def standardize(x):
        return (np.asarray(x, dtype=np.float64) - x_mean) / x_sd

    x_tr = torch.tensor(standardize(covariate_train), dtype=torch.float32, device=device)
    d_tr = torch.tensor(duration_train, dtype=torch.long, device=device)
    e_tr = torch.tensor(event_train, dtype=torch.long, device=device)
    x_te = torch.tensor(standardize(covariate_test), dtype=torch.float32, device=device)
    d_te = torch.tensor(duration_test, dtype=torch.long, device=device)
    e_te = torch.tensor(event_test, dtype=torch.long, device=device)

    model = DRSAModel(hidden_size=hidden_size, L=L).to(device)
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
            xb, db, eb = x_tr[batch_idx], d_tr[batch_idx], e_tr[batch_idx]
            y = model(xb)
            loss = drsa_loss(y, db, eb, alpha=alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            test_loss = drsa_loss(model(x_te), d_te, e_te, alpha=alpha).item()

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
def predict_survival_drsa(model, meta, covariate_values, eval_times):
    """
    Predict S(t | x) at each integer time in `eval_times` (expected to be
    0..L, matching np.arange(0, T_K+1) used elsewhere). Returns an array of
    shape (n_individuals, len(eval_times)).
    """
    device = meta["device"]
    x_std = (np.asarray(covariate_values, dtype=np.float64) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        y = model(x_t).cpu().numpy()  # (n, L) hazards for intervals/times 1..L

    S = np.cumprod(1 - y, axis=1)                     # S at times 1..L
    S_full = np.concatenate([np.ones((y.shape[0], 1)), S], axis=1)  # prepend S(0)=1 -> times 0..L

    eval_times = np.asarray(eval_times).astype(int)
    return S_full[:, eval_times]


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors run_cph_for_scenario)
# ---------------------------------------------------------------------------
def run_drsa_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                           demo_values=None, seed=4242, **train_kwargs):
    cohort = load_cohort_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== DRSA fit on {scenario.name} scenario (covariate: {scenario.covariate_name}) ===")
    model, meta = train_drsa(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["event"][train_idx],
        cohort["covariate"][test_idx], cohort["duration"][test_idx], cohort["event"][test_idx],
        L=T_K, seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted = predict_survival_drsa(model, meta, sample_covariate, eval_times)
    ground_truth = ground_truth_km_for_validation_set(  # defined in the SFSA cell already run
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
    )
    mae_t = mae_per_timestep(predicted, ground_truth)

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gt_n_matched}-person ground truth each):")
    print(f"  mean MAE across all t: {mae_t.mean():.4f}")
    for t in (0, 10, 20, 30):
        print(f"  MAE at t={t}: {mae_t[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t, label="DRSA", color="tab:brown")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"DRSA MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    sf_pred_all = predict_survival_drsa(model, meta, demo_values, eval_times)
    fig, axes = plt.subplots(1, len(demo_values), figsize=(5 * len(demo_values), 4), sharey=True)
    demo_curves = {}
    for ax, v, sf_pred in zip(np.atleast_1d(axes), demo_values, sf_pred_all):
        t_gt, sf_gt = ground_truth_km(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        demo_curves[v] = {"ground_truth": sf_gt, "drsa": sf_pred}

        ax.step(t_gt, sf_gt, where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, sf_pred, where="post", label="DRSA", color="tab:brown")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.set_xlabel("Time")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("Survival probability")
    plt.suptitle(f"DRSA vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    drsa_model_relu, drsa_meta_relu, eval_times_relu_drsa, mae_t_relu_drsa, demo_relu_drsa = run_drsa_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort.npz",
    )
    drsa_model_weibull, drsa_meta_weibull, eval_times_weibull_drsa, mae_t_weibull_drsa, demo_weibull_drsa = run_drsa_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort.npz",
    )