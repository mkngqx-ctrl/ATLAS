"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
# Model: same GRU + additive-attention-pooling backbone, joint (cause, time)
# softmax head
# ---------------------------------------------------------------------------
class DynamicDeepHitCRModel(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1, L=30, attn_size=16):
        super().__init__()
        self.L = L
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        # additive attention: score_t = v^T tanh(W h_t) -- identical to the base file
        self.attn_W = nn.Linear(hidden_size, attn_size)
        self.attn_v = nn.Linear(attn_size, 1, bias=False)
        # joint (cause, time) PMF head: reads [context; last hidden state] ->
        # 2*L logits -> reshape (2, L) -> ONE softmax over the flattened 2*L entries
        self.head = nn.Linear(hidden_size * 2, 2 * L + 1)

    def forward(self, x_std):
        batch = x_std.shape[0]
        device = x_std.device
        l_idx = torch.arange(1, self.L + 1, device=device, dtype=torch.float32) / self.L
        x_rep = x_std.view(batch, 1, 1).expand(batch, self.L, 1)
        l_rep = l_idx.view(1, self.L, 1).expand(batch, self.L, 1)
        inp = torch.cat([x_rep, l_rep], dim=-1)          # (batch, L, 2)

        out, h_n = self.gru(inp)                         # out: (batch, L, hidden), h_n: (1, batch, hidden)
        last_hidden = h_n[-1]                             # (batch, hidden)

        scores = self.attn_v(torch.tanh(self.attn_W(out)))  # (batch, L, 1)
        weights = torch.softmax(scores, dim=1)               # (batch, L, 1), sums to 1 over steps
        context = (weights * out).sum(dim=1)                 # (batch, hidden)

        combined = torch.cat([context, last_hidden], dim=-1)  # (batch, 2*hidden)
        logits = self.head(combined)   # (batch, 2, L)
        pmf = torch.softmax(logits, dim=-1)[:, :-1].reshape(batch, 2, self.L)  # single joint softmax
        return pmf, weights.squeeze(-1)  # also return attention weights, useful for inspection


# ---------------------------------------------------------------------------
# Losses -- direct generalization of l1_loss/l2_loss/combined_loss (see
# module docstring)
# ---------------------------------------------------------------------------
def l1_loss_cr(pmf, duration, cause, eps=1e-7):
    """Joint (cause, time) discrete failure NLL.
    pmf: (batch, 2, L) joint-softmax probabilities. duration: (batch,)
    1-indexed in [1, L]. cause: (batch,) 0=censored, 1=disease, 2=other."""
    batch, K, L = pmf.shape
    cif = torch.cumsum(pmf, dim=2)                      # (batch, 2, L) cause-specific CIF over steps
    idx = (duration - 1).clamp(0, L - 1).long()
    b = torch.arange(batch, device=pmf.device)

    overall_cif_at_idx = cif.sum(dim=1)[b, idx]          # P(any event by duration)
    is_event = (cause > 0)
    cause_col = (cause - 1).clamp(min=0).long()          # 0=disease, 1=other column index
    pmf_at_idx = pmf[b, cause_col, idx]

    loss_event = -torch.log(pmf_at_idx.clamp(min=eps))
    loss_censored = -torch.log((1 - overall_cif_at_idx).clamp(min=eps))
    per_i = torch.where(is_event, loss_event, loss_censored)

    cif_event_cause_at_idx = cif[b, cause_col, idx]      # cause-specific CIF at own time, for L2
    return per_i.mean(), cif_event_cause_at_idx, cause_col


def l2_loss_cr(cif, cause_col, duration, cause, rank_sigma=1.0):
    """Use cause of earlier event; later outcome need not have that cause."""
    if rank_sigma <= 0: raise ValueError('rank_sigma must be positive')
    idx = duration.long() - 1
    own = cif[torch.arange(len(idx), device=cif.device), cause_col, idx]
    other = cif[:, cause_col, idx].T
    mask = (duration[:, None] < duration[None, :]) & (cause[:, None] > 0)
    terms = torch.exp(-(own[:, None] - other) / rank_sigma)
    return terms[mask].sum() / mask.sum().clamp(min=1)


def combined_loss_cr(pmf, duration, cause, rank_weight=1.0, rank_sigma=1.0, eps=1e-7):
    l1, cif_at_idx, cause_col = l1_loss_cr(pmf, duration, cause, eps=eps)
    l2 = l2_loss_cr(torch.cumsum(pmf, dim=2), cause_col, duration, cause, rank_sigma=rank_sigma)
    return l1 + rank_weight * l2, l1, l2


# ---------------------------------------------------------------------------
# Training (byte-identical loop structure to train_dynamic_deephit, new
# loss/targets)
# ---------------------------------------------------------------------------
def train_dynamic_deephit_cr(covariate_train, duration_train, cause_train,
                              covariate_test, duration_test, cause_test,
                              L=30, hidden_size=32, rank_weight=1.0, rank_sigma=1.0,
                              lr=1e-3, batch_size=256, max_epochs=200, patience=15,
                              seed=0, device=None, verbose=True):
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

    model = DynamicDeepHitCRModel(hidden_size=hidden_size, L=L).to(device)
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
            pmf, _ = model(xb)
            loss, l1, l2 = combined_loss_cr(pmf, db, cb, rank_weight=rank_weight, rank_sigma=rank_sigma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            pmf_te, _ = model(x_te)
            test_loss, l1_te, l2_te = combined_loss_cr(pmf_te, d_te, c_te,
                                                        rank_weight=rank_weight, rank_sigma=rank_sigma)
            test_loss = test_loss.item()

        improved = test_loss < best_test_loss - 1e-5
        if improved:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 10 == 0 or epochs_since_improve >= patience):
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                  f"(L1={l1_te.item():.4f} L2={l2_te.item():.4f}){'  (best)' if improved else ''}")

        if epochs_since_improve >= patience:
            print(f"  early stopping at epoch {epoch} (best test_loss={best_test_loss:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"x_mean": x_mean, "x_sd": x_sd, "device": device, "L": L}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_cif_dh(model, meta, covariate_values, eval_times):
    """cif_k(t) = cumsum(pmf[:, k, :])[t], indexed at eval_times; cif=0 for
    t<=0, held flat past t=L (no more mass is ever placed beyond the L
    discrete bins). eval_times is assumed to be an integer grid 0..L
    matching the model's L discrete bins. Returns (n, 2, len(eval_times)):
    axis index 0 = disease CIF, axis index 1 = other CIF."""
    device = meta["device"]
    L = meta["L"]
    x_std = (np.asarray(covariate_values, dtype=np.float64) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pmf, _ = model(x_t)
        pmf_np = pmf.cpu().numpy()  # (n, 2, L)

    cif = np.cumsum(pmf_np, axis=2)  # (n, 2, L), cif[:, k, l] = P(cause k by bin l+1)
    eval_times = np.asarray(eval_times)
    n = pmf_np.shape[0]
    cif_out = np.zeros((n, 2, len(eval_times)))
    for j, t in enumerate(eval_times):
        if t <= 0:
            cif_out[:, :, j] = 0.0
        elif t >= L:
            cif_out[:, :, j] = cif[:, :, L - 1]
        else:
            cif_out[:, :, j] = cif[:, :, int(t) - 1]
    return cif_out


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors run_dh_for_scenario, but compares
# against ground_truth_cif_for_validation_set / ground_truth_cif instead of
# the single-risk ground_truth_km equivalents)
# ---------------------------------------------------------------------------
def run_dh_cr_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                            demo_values=None, seed=4242, **train_kwargs):
    cohort = load_cohort_cr_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== DynamicDeepHIt-Lite (competing risks) fit on {scenario.name} scenario "
          f"(covariate: {scenario.covariate_name}) ===")
    model, meta = train_dynamic_deephit_cr(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["cause"][train_idx],
        cohort["covariate"][test_idx], cohort["duration"][test_idx], cohort["cause"][test_idx],
        L=T_K, seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted_cif = predict_cif_dh(model, meta, sample_covariate, eval_times)  # (n, 2, T+1)
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
    plt.plot(eval_times, mae_t_survival, label="Survival", color="tab:green", linestyle="-")
    plt.plot(eval_times, mae_t_disease, label="Disease CIF", color="tab:green", linestyle="--")
    plt.plot(eval_times, mae_t_other, label="Other CIF", color="tab:green", linestyle=":")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"DynamicDeepHIt-Lite-CR MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    pred_cif_demo = predict_cif_dh(model, meta, demo_values, eval_times)  # (n_demo, 2, T+1)
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
        ax.step(eval_times, pred_survival, where="post", label="DynamicDeepHIt-Lite-CR", color="tab:green")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.step(eval_times, gt["cif_disease"], where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, pred_disease, where="post", label="DynamicDeepHIt-Lite-CR", color="tab:green")
        ax.legend(fontsize=8)

        ax = axes[2, col]
        ax.step(eval_times, gt["cif_other"], where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, pred_other, where="post", label="DynamicDeepHIt-Lite-CR", color="tab:green")
        ax.set_xlabel("Time")
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel("Survival probability")
    axes[1, 0].set_ylabel("Disease CIF")
    axes[2, 0].set_ylabel("Other CIF")
    plt.suptitle(f"DynamicDeepHIt-Lite-CR vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t_disease, mae_t_other, demo_cif


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    (dh_model_cr_relu, dh_meta_cr_relu, eval_times_cr_relu_dh,
     mae_t_disease_cr_relu_dh, mae_t_other_cr_relu_dh, demo_cr_relu_dh) = run_dh_cr_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort_cr.npz",
    )
    (dh_model_cr_weibull, dh_meta_cr_weibull, eval_times_cr_weibull_dh,
     mae_t_disease_cr_weibull_dh, mae_t_other_cr_weibull_dh, demo_cr_weibull_dh) = run_dh_cr_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort_cr.npz",
    )
