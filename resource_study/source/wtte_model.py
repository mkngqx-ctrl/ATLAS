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
# Model: one (alpha, beta) Weibull pair per individual, straight from their
# covariate -- no artificial time-step unroll (see module docstring).
# ---------------------------------------------------------------------------
class WTTEModel(nn.Module):
    def __init__(self, hidden_size=32, init_alpha=15.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_size, 2)  # -> raw alpha, raw beta

        # Same alpha-bias initialization trick as v1 (this part wasn't the
        # problem): start the network's initial alpha guess near a typical
        # duration scale (now grounded in the TRAINING set's actual mean
        # duration, passed in by train_wtte, rather than a generic L/2
        # guess) so early training isn't fighting a badly-scaled starting
        # point.
        with torch.no_grad():
            init_alpha = max(float(init_alpha), 2.0)
            self.head.bias[0] = np.log(np.exp(init_alpha) - 1)  # softplus^-1(init_alpha)
            self.head.bias[1] = 0.0  # softplus(0) ~ 0.69, a mild starting beta

    def forward(self, x_std):
        h = self.net(x_std.view(-1, 1))
        raw = self.head(h)
        eps = 1e-4
        alpha = torch.nn.functional.softplus(raw[:, 0]) + eps
        beta = torch.nn.functional.softplus(raw[:, 1]) + eps
        return alpha, beta  # each (batch,)


# ---------------------------------------------------------------------------
# Loss: exact standard censored Weibull log-likelihood, one term per
# individual (verified against scipy.stats.weibull_min -- see module
# docstring)
# ---------------------------------------------------------------------------
def wtte_loss(alpha, beta, duration, event, eps=1e-7):
    """Discrete Weibull: event mass S(t-1)-S(t), censoring mass S(t)."""
    t = duration.to(alpha.dtype)
    if torch.any(t < 1) or torch.any(t != t.floor()): raise ValueError('integer times >= 1 required')
    a, b = alpha.clamp(min=eps), beta.clamp(min=eps)
    # Avoid differentiating 0**beta at t=1.
    prev = ((t - 1).clamp(min=1) / a) ** b
    prev = torch.where(t > 1, prev, torch.zeros_like(prev))
    cur = (t / a) ** b
    log_event = -prev + torch.log(-torch.expm1(-(cur-prev).clamp(min=torch.finfo(a.dtype).tiny)))
    return -torch.where(event.bool(), log_event, -cur).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_wtte(covariate_train, duration_train, event_train,
                covariate_test, duration_test, event_test,
                hidden_size=32, lr=1e-3, batch_size=256,
                max_epochs=300, patience=20, seed=0, device=None, verbose=True,
                max_alpha=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    x_mean = float(np.mean(covariate_train))
    x_sd = float(np.std(covariate_train)) or 1.0
    init_alpha = float(np.mean(duration_train))  # grounded in the actual data now, not a generic guess

    def standardize(x):
        return (np.asarray(x, dtype=np.float64) - x_mean) / x_sd

    x_tr = torch.tensor(standardize(covariate_train), dtype=torch.float32, device=device)
    d_tr = torch.tensor(duration_train, dtype=torch.float32, device=device)
    e_tr = torch.tensor(event_train, dtype=torch.long, device=device)
    x_te = torch.tensor(standardize(covariate_test), dtype=torch.float32, device=device)
    d_te = torch.tensor(duration_test, dtype=torch.float32, device=device)
    e_te = torch.tensor(event_test, dtype=torch.long, device=device)

    model = WTTEModel(hidden_size=hidden_size, init_alpha=init_alpha).to(device)
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
            alpha, beta = model(xb)
            if max_alpha is not None:
                alpha = alpha.clamp(max=max_alpha)  # optional safety net, not required for the fix itself
            loss = wtte_loss(alpha, beta, db, eb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            alpha_te, beta_te = model(x_te)
            if max_alpha is not None:
                alpha_te = alpha_te.clamp(max=max_alpha)
            test_loss = wtte_loss(alpha_te, beta_te, d_te, e_te).item()

        improved = test_loss < best_test_loss - 1e-5
        if improved:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 10 == 0 or epochs_since_improve >= patience):
            with torch.no_grad():
                alpha_sample, beta_sample = model(x_te[:5])
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} test_loss={test_loss:.4f}"
                  f"{'  (best)' if improved else ''}"
                  f"  | sample alpha={alpha_sample.cpu().numpy().round(2)} beta={beta_sample.cpu().numpy().round(2)}")

        if epochs_since_improve >= patience:
            print(f"  early stopping at epoch {epoch} (best test_loss={best_test_loss:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"x_mean": x_mean, "x_sd": x_sd, "device": device, "max_alpha": max_alpha}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_survival_wtte(model, meta, covariate_values, eval_times):
    """S(t) = exp[-(t/alpha)^beta] using each individual's single (alpha,
    beta) pair. Shape (n_individuals, len(eval_times))."""
    device = meta["device"]
    x_std = (np.asarray(covariate_values, dtype=np.float64) - meta["x_mean"]) / meta["x_sd"]
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        alpha, beta = model(x_t)
        if meta.get("max_alpha") is not None:
            alpha = alpha.clamp(max=meta["max_alpha"])
        alpha_np = alpha.cpu().numpy()
        beta_np = beta.cpu().numpy()

    t = np.asarray(eval_times, dtype=np.float64)[None, :]  # (1, n_times)
    a = alpha_np[:, None]
    b = beta_np[:, None]
    S = np.exp(-((t / a) ** b))
    S[:, 0] = 1.0  # exact S(0)=1, avoids any floating-point drift at t=0
    return S


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors run_cph_for_scenario / run_drsa_for_scenario)
# ---------------------------------------------------------------------------
def run_wtte_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                           demo_values=None, seed=4242, **train_kwargs):
    cohort = load_cohort_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== WTTE-RNN fit on {scenario.name} scenario (covariate: {scenario.covariate_name}) ===")
    model, meta = train_wtte(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["event"][train_idx],
        cohort["covariate"][test_idx], cohort["duration"][test_idx], cohort["event"][test_idx],
        seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted = predict_survival_wtte(model, meta, sample_covariate, eval_times)
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
    plt.plot(eval_times, mae_t, label="WTTE-RNN", color="tab:orange")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"WTTE-RNN MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    sf_pred_all = predict_survival_wtte(model, meta, demo_values, eval_times)
    fig, axes = plt.subplots(1, len(demo_values), figsize=(5 * len(demo_values), 4), sharey=True)
    demo_curves = {}
    for ax, v, sf_pred in zip(np.atleast_1d(axes), demo_values, sf_pred_all):
        t_gt, sf_gt = ground_truth_km(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        demo_curves[v] = {"ground_truth": sf_gt, "wtte": sf_pred}

        ax.step(t_gt, sf_gt, where="post", label="Ground truth", color="tab:red")
        ax.step(eval_times, sf_pred, where="post", label="WTTE-RNN", color="tab:orange")
        ax.set_title(f"{scenario.covariate_name}={v}")
        ax.set_xlabel("Time")
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("Survival probability")
    plt.suptitle(f"WTTE-RNN vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    wtte_model_relu, wtte_meta_relu, eval_times_relu_wtte, mae_t_relu_wtte, demo_relu_wtte = run_wtte_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort.npz",
    )
    wtte_model_weibull, wtte_meta_weibull, eval_times_weibull_wtte, mae_t_weibull_wtte, demo_weibull_wtte = run_wtte_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort.npz",
    )