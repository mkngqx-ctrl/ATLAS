"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
# Standard sinusoidal positional encoding (shared, byte-identical, with
# glazier_transformer_model.py -- redefining it there is harmless since
# both copies are functionally identical).
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


# ---------------------------------------------------------------------------
# Model: non-causal self-attention encoder + mean-pool + softmax-over-L-bins
# ---------------------------------------------------------------------------
class SurvTransformerModel(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, L=30, dropout=0.1):
        super().__init__()
        self.L = L
        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=L + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, L + 1)

    def forward(self, x_std):
        batch = x_std.shape[0]
        device = x_std.device
        l_idx = torch.arange(1, self.L + 1, device=device, dtype=torch.float32) / self.L
        x_rep = x_std.view(batch, 1, 1).expand(batch, self.L, 1)
        l_rep = l_idx.view(1, self.L, 1).expand(batch, self.L, 1)
        inp = torch.cat([x_rep, l_rep], dim=-1)          # (batch, L, 2)
        h = self.input_proj(inp)                          # (batch, L, d_model)
        h = self.pos_enc(h)
        out = self.encoder(h)                             # (batch, L, d_model) -- full, non-causal self-attention
        pooled = out.mean(dim=1)                           # (batch, d_model) -- mean-pool readout
        logits = self.head(pooled)                         # (batch, L)
        pmf = torch.softmax(logits, dim=-1)
        return pmf


# ---------------------------------------------------------------------------
# Losses -- copied verbatim from dynamic_deephit_model.py, where they were
# validated against brute-force references. Safe to redefine here under the
# same names if both files are pasted into the same notebook (identical
# implementations).
# ---------------------------------------------------------------------------
def l1_loss(pmf, duration, event, eps=1e-7):
    batch = pmf.shape[0]
    idx = duration - 1
    cum = torch.cumsum(pmf, dim=1)
    cif_at_idx = cum[torch.arange(batch, device=pmf.device), idx]
    pmf_at_idx = pmf[torch.arange(batch, device=pmf.device), idx]

    event_f = event.float()
    loss_uncensored = -torch.log(pmf_at_idx.clamp(min=eps))
    loss_censored = -torch.log((1 - cif_at_idx).clamp(min=eps))
    per_i = event_f * loss_uncensored + (1 - event_f) * loss_censored
    return per_i.mean(), cif_at_idx


def l2_loss(cif, duration, event, rank_sigma=1.0):
    """Compare both patients at the earlier observed event time."""
    if rank_sigma <= 0: raise ValueError('rank_sigma must be positive')
    idx = duration.long() - 1
    own = cif[torch.arange(len(idx), device=cif.device), idx]
    other = cif[:, idx].T
    mask = (duration[:, None] < duration[None, :]) & (event[:, None] == 1)
    terms = torch.exp(-(own[:, None] - other) / rank_sigma)
    return terms[mask].sum() / mask.sum().clamp(min=1)


def combined_loss(pmf, duration, event, rank_weight=1.0, rank_sigma=1.0, eps=1e-7):
    l1, cif_at_idx = l1_loss(pmf, duration, event, eps=eps)
    l2 = l2_loss(torch.cumsum(pmf[:, :-1], dim=1), duration, event, rank_sigma=rank_sigma)
    return l1 + rank_weight * l2, l1, l2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_survtransformer(covariate_train, duration_train, event_train,
                           covariate_test, duration_test, event_test,
                           L=30, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                           rank_weight=1.0, rank_sigma=1.0, dropout=0.1,
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
    e_tr = torch.tensor(event_train, dtype=torch.long, device=device)
    x_te = torch.tensor(standardize(covariate_test), dtype=torch.float32, device=device)
    d_te = torch.tensor(duration_test, dtype=torch.long, device=device)
    e_te = torch.tensor(event_test, dtype=torch.long, device=device)

    model = SurvTransformerModel(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                  dim_feedforward=dim_feedforward, L=L, dropout=dropout).to(device)
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
            pmf = model(xb)
            loss, l1, l2 = combined_loss(pmf, db, eb, rank_weight=rank_weight, rank_sigma=rank_sigma)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            pmf_te = model(x_te)
            test_loss, l1_te, l2_te = combined_loss(pmf_te, d_te, e_te, rank_weight=rank_weight, rank_sigma=rank_sigma)
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
def predict_survival_survtransformer(model, meta, covariate_values, eval_times):
    device = meta["device"]
    L = meta["L"]
    x_std = (np.asarray(covariate_values, dtype=np.float64) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pmf = model(x_t)
        pmf_np = pmf.cpu().numpy()

    cif = np.cumsum(pmf_np, axis=1)
    eval_times = np.asarray(eval_times)
    S = np.zeros((pmf_np.shape[0], len(eval_times)))
    for j, t in enumerate(eval_times):
        if t <= 0:
            S[:, j] = 1.0
        elif t >= L:
            S[:, j] = 1.0 - cif[:, L - 1]
        else:
            S[:, j] = 1.0 - cif[:, int(t) - 1]
    return S


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors the other runners)
# ---------------------------------------------------------------------------
def run_survtransformer_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                                      demo_values=None, seed=4242, **train_kwargs):
    cohort = load_cohort_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== SurvTransformer-Lite fit on {scenario.name} scenario (covariate: {scenario.covariate_name}) ===")
    model, meta = train_survtransformer(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["event"][train_idx],
        cohort["covariate"][test_idx], cohort["duration"][test_idx], cohort["event"][test_idx],
        L=T_K, seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted = predict_survival_survtransformer(model, meta, sample_covariate, eval_times)
    ground_truth = ground_truth_km_for_validation_set(
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
    )
    mae_t = mae_per_timestep(predicted, ground_truth)

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gt_n_matched}-person ground truth each):")
    print(f"  mean MAE across all t: {mae_t.mean():.4f}")
    for t in (0, 10, 20, 30):
        print(f"  MAE at t={t}: {mae_t[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t, label="SurvTransformer-Lite", color="tab:olive")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"SurvTransformer-Lite MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    sf_pred_all = predict_survival_survtransformer(model, meta, demo_values, eval_times)
    fig, axes = plt.subplots(1, len(demo_values), figsize=(5 * len(demo_values), 4), sharey=True)
    demo_curves = {}
    for ax, v, sf_pred in zip(np.atleast_1d(axes), demo_values, sf_pred_all):
        t_gt, sf_gt = ground_truth_km(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        demo_curves[v] = {"ground_truth": sf_gt, "survtransformer": sf_pred}

        ax.step(t_gt, sf_gt, where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, sf_pred, where="post", label="SurvTransformer-Lite", color="tab:olive")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.set_xlabel("Time")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("Survival probability")
    plt.suptitle(f"SurvTransformer-Lite vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    survtransformer_model_relu, survtransformer_meta_relu, eval_times_relu_survtransformer, mae_t_relu_survtransformer, demo_relu_survtransformer = run_survtransformer_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort.npz",
    )
    survtransformer_model_weibull, survtransformer_meta_weibull, eval_times_weibull_survtransformer, mae_t_weibull_survtransformer, demo_weibull_survtransformer = run_survtransformer_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort.npz",
    )