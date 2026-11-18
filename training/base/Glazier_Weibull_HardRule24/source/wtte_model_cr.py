"""Revision working implementation. See README.md for corrections, validation, and pending runs.
Legacy model labels identify adapted implementations, not faithful reproductions.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading a saved competing-risks cohort
# ---------------------------------------------------------------------------
def load_cohort_cr_npz(path):
    """Competing-risks analogue of wtte_model.py's `load_cohort_npz`. Also
    carries `states` and a `cause` column (0=censored, 1=disease, 2=other)
    in place of the single-risk file's binary `event` column."""
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
# Model: TWO (alpha, beta) Weibull pairs per individual -- one per cause --
# from the same feedforward backbone the base WTTEModel uses, via a
# 4-output head (alpha_d, beta_d, alpha_o, beta_o) instead of a 2-output one.
# ---------------------------------------------------------------------------
class WTTEModelCR(nn.Module):
    def __init__(self, hidden_size=32, init_alpha=15.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_size, 4)  # -> raw alpha_d, raw beta_d, raw alpha_o, raw beta_o

        # Same alpha-bias grounding trick as the base WTTEModel (start the
        # network's initial alpha guess near a typical duration scale, so
        # early training isn't fighting a badly-scaled starting point),
        # duplicated across BOTH cause-specific alpha sub-heads (columns 0
        # and 2) -- there's no reason one cause should start better-scaled
        # than the other, so both get the same overall-duration-based init.
        with torch.no_grad():
            init_alpha = max(float(init_alpha), 2.0)
            log_init = np.log(np.exp(init_alpha) - 1)  # softplus^-1(init_alpha)
            self.head.bias[0] = log_init  # alpha_d
            self.head.bias[1] = 0.0       # beta_d  (softplus(0) ~ 0.69, mild starting beta)
            self.head.bias[2] = log_init  # alpha_o
            self.head.bias[3] = 0.0       # beta_o

    def forward(self, x_std):
        h = self.net(x_std.view(-1, 1))
        raw = self.head(h)
        eps = 1e-4
        alpha_d = torch.nn.functional.softplus(raw[:, 0]) + eps
        beta_d = torch.nn.functional.softplus(raw[:, 1]) + eps
        alpha_o = torch.nn.functional.softplus(raw[:, 2]) + eps
        beta_o = torch.nn.functional.softplus(raw[:, 3]) + eps
        return alpha_d, beta_d, alpha_o, beta_o  # each (batch,)


# ---------------------------------------------------------------------------
# Loss: two-cause Weibull mixture log-likelihood (see module docstring for
# the derivation and the single-risk-collapse sanity check).
# ---------------------------------------------------------------------------
def wtte_cr_loss(alpha_d, beta_d, alpha_o, beta_o, duration, cause, eps=1e-7):
    """
    alpha_d, beta_d, alpha_o, beta_o: (batch,) FloatTensors. duration:
    (batch,) duration values (>0). cause: (batch,) long in {0, 1, 2}
    (0=censored, 1=disease, 2=other).
    """
    y = duration.float().clamp(min=eps)
    a_d, b_d = alpha_d.clamp(min=eps), beta_d.clamp(min=eps)
    a_o, b_o = alpha_o.clamp(min=eps), beta_o.clamp(min=eps)

    def log_hazard(a, b):
        # log h(y) = log(b) - log(a) + (b-1) * log(y/a)  -- log of the
        # instantaneous cause-specific Weibull hazard at the observed time.
        return torch.log(b) - torch.log(a) + (b - 1) * torch.log(y / a)

    def cum_hazard(a, b):
        # Lambda(y) = (y/a)^b  -- cause-specific cumulative hazard at y.
        return (y / a) ** b

    # Overall survival uses the SUM of both causes' cumulative hazards --
    # this is the additive-hazards identity described in the module
    # docstring, and is what lets a two-cause model share one "still alive"
    # term instead of needing two separate survival curves.
    logS = -(cum_hazard(a_d, b_d) + cum_hazard(a_o, b_o))

    is_d = (cause == 1).float()
    is_o = (cause == 2).float()
    loglik = is_d * log_hazard(a_d, b_d) + is_o * log_hazard(a_o, b_o) + logS
    return -loglik.mean()


# ---------------------------------------------------------------------------
# Training (byte-for-byte the base file's Adam/mini-batch/early-stopping
# loop, just with the two-cause model, loss and grad-clip applied to both
# alpha heads)
# ---------------------------------------------------------------------------
def train_wtte_cr(covariate_train, duration_train, cause_train,
                   covariate_test, duration_test, cause_test,
                   hidden_size=32, lr=1e-3, batch_size=256,
                   max_epochs=300, patience=20, seed=0, device=None, verbose=True,
                   max_alpha=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    x_mean = float(np.mean(covariate_train))
    x_sd = float(np.std(covariate_train)) or 1.0
    # Same grounding trick as the base file (init_alpha = mean observed
    # duration), duplicated to both cause heads -- see WTTEModelCR.__init__.
    init_alpha = float(np.mean(duration_train))

    def standardize(x):
        return (np.asarray(x, dtype=np.float64) - x_mean) / x_sd

    x_tr = torch.tensor(standardize(covariate_train), dtype=torch.float32, device=device)
    d_tr = torch.tensor(duration_train, dtype=torch.float32, device=device)
    c_tr = torch.tensor(cause_train, dtype=torch.long, device=device)
    x_te = torch.tensor(standardize(covariate_test), dtype=torch.float32, device=device)
    d_te = torch.tensor(duration_test, dtype=torch.float32, device=device)
    c_te = torch.tensor(cause_test, dtype=torch.long, device=device)

    model = WTTEModelCR(hidden_size=hidden_size, init_alpha=init_alpha).to(device)
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
            alpha_d, beta_d, alpha_o, beta_o = model(xb)
            if max_alpha is not None:
                alpha_d = alpha_d.clamp(max=max_alpha)  # optional safety net, same role as the base file's
                alpha_o = alpha_o.clamp(max=max_alpha)
            loss = wtte_cr_loss(alpha_d, beta_d, alpha_o, beta_o, db, cb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_train_loss += loss.item() * len(batch_idx)
        train_loss = total_train_loss / n

        model.eval()
        with torch.no_grad():
            alpha_d_te, beta_d_te, alpha_o_te, beta_o_te = model(x_te)
            if max_alpha is not None:
                alpha_d_te = alpha_d_te.clamp(max=max_alpha)
                alpha_o_te = alpha_o_te.clamp(max=max_alpha)
            test_loss = wtte_cr_loss(alpha_d_te, beta_d_te, alpha_o_te, beta_o_te, d_te, c_te).item()

        improved = test_loss < best_test_loss - 1e-5
        if improved:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 10 == 0 or epochs_since_improve >= patience):
            with torch.no_grad():
                ad_s, bd_s, ao_s, bo_s = model(x_te[:5])
            print(f"  epoch {epoch}: train_loss={train_loss:.4f} test_loss={test_loss:.4f}"
                  f"{'  (best)' if improved else ''}"
                  f"  | sample alpha_d={ad_s.cpu().numpy().round(2)} beta_d={bd_s.cpu().numpy().round(2)}"
                  f" alpha_o={ao_s.cpu().numpy().round(2)} beta_o={bo_s.cpu().numpy().round(2)}")

        if epochs_since_improve >= patience:
            print(f"  early stopping at epoch {epoch} (best test_loss={best_test_loss:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"x_mean": x_mean, "x_sd": x_sd, "device": device, "max_alpha": max_alpha}


# ---------------------------------------------------------------------------
# Prediction: integrate the competing-Weibull mixture into cause-specific CIFs
# ---------------------------------------------------------------------------
def predict_cif_wtte(model, meta, covariate_values, eval_times):
    """Integrate the fitted continuous competing-Weibull model accurately.

    CIF_k(t) = integral_0^t exp(-H_d(u)-H_o(u)) h_k(u) du.
    Adaptive quadrature handles an integrable hazard singularity at zero.
    Each interval's total event mass is the exact survival decrement;
    the other-cause mass is its complement to the disease-cause integral.
    This preserves nonnegative increments and S + F_d + F_o = 1 without
    time-varying rescaling of cumulative incidence curves.
    """
    from scipy.integrate import quad
    grid=np.asarray(eval_times,dtype=np.float64)
    if grid.ndim!=1 or len(grid)==0 or not np.isfinite(grid).all() or np.any(grid<0) or np.any(np.diff(grid)<=0):
        raise ValueError('eval_times must be finite, nonnegative and strictly increasing')
    x=(np.asarray(covariate_values,dtype=np.float64)-meta['x_mean'])/meta['x_sd']
    model.eval()
    with torch.no_grad():
        ad,bd,ao,bo=model(torch.tensor(x,dtype=torch.float32,device=meta['device']))
        if meta.get('max_alpha') is not None:
            ad=ad.clamp(max=meta['max_alpha']);ao=ao.clamp(max=meta['max_alpha'])
        params=[v.detach().cpu().numpy().astype(np.float64).reshape(-1) for v in (ad,bd,ao,bo)]
    if not all(np.isfinite(v).all() and (v>0).all() for v in params):
        raise ValueError('Weibull parameters must be finite and positive')
    result=np.zeros((len(x),2,len(grid)),dtype=np.float64)
    for i,(ad,bd,ao,bo) in enumerate(zip(*params)):
        def survival(t):return np.exp(-((t/ad)**bd+(t/ao)**bo))
        def density(t):
            # Quadrature nodes lie inside the interval, avoiding t=0.
            hd=(t/ad)**bd;ho=(t/ao)**bo
            return np.exp(-hd-ho+np.log(bd/ad)+(bd-1)*np.log(t/ad))
        previous=0.;sd_previous=1.;fd=0.;fo=0.
        for j,t in enumerate(grid):
            if t==0:continue
            st=survival(t);mass=max(0.,sd_previous-st)
            d,error=quad(density,previous,t,epsabs=1e-10,epsrel=1e-9,limit=200)
            if error>1e-7 or d < -1e-8 or d>mass+1e-8:
                raise ArithmeticError('Competing-Weibull quadrature failed its mass check')
            d=min(max(d,0.),mass)  # roundoff only, after strict accuracy check
            fd+=d;fo+=mass-d
            result[i,:,j]=(fd,fo)
            previous=t;sd_previous=st
    return result


def mae_per_timestep(predicted, ground_truth):
    return np.abs(predicted - ground_truth).mean(axis=0)


# ---------------------------------------------------------------------------
# End-to-end run for one scenario (mirrors run_wtte_for_scenario / run_cph_cr_for_scenario)
# ---------------------------------------------------------------------------
def run_wtte_cr_for_scenario(scenario, cohort_path, n_mae_sample=300, gt_n_matched=2000,
                              demo_values=None, seed=4242, **train_kwargs):
    """
    Returns (model, meta, eval_times, mae_t_survival, mae_t_disease,
    mae_t_other, demo_curves).
    """
    cohort = load_cohort_cr_npz(cohort_path)
    train_idx, test_idx, val_idx = cohort["train_index"], cohort["val_index"], cohort["test_index"] # local test_idx is the early-stopping validation partition

    print(f"\n=== WTTE-RNN (competing risks) fit on {scenario.name} scenario "
          f"(covariate: {scenario.covariate_name}) ===")
    model, meta = train_wtte_cr(
        cohort["covariate"][train_idx], cohort["duration"][train_idx], cohort["cause"][train_idx],
        cohort["covariate"][test_idx], cohort["duration"][test_idx], cohort["cause"][test_idx],
        seed=seed, **train_kwargs,
    )

    eval_times = np.arange(0, T_K + 1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(val_idx, size=min(n_mae_sample, len(val_idx)), replace=False)
    sample_covariate = cohort["covariate"][sample_idx]

    predicted_cif = predict_cif_wtte(model, meta, sample_covariate, eval_times)  # (n, 2, T1)
    predicted_survival = 1.0 - predicted_cif[:, 0, :] - predicted_cif[:, 1, :]

    ground_truth = ground_truth_cif_for_validation_set(
        scenario, sample_covariate, base_seed=seed, n_matched=gt_n_matched, t_k=T_K
    )

    mae_t_survival = mae_per_timestep(predicted_survival, ground_truth["survival"])
    mae_t_disease = mae_per_timestep(predicted_cif[:, 0, :], ground_truth["cif_disease"])
    mae_t_other = mae_per_timestep(predicted_cif[:, 1, :], ground_truth["cif_other"])

    print(f"\nMAE_t summary over {len(sample_idx)} validation individuals "
          f"({gt_n_matched}-person ground truth each):")
    print(f"  mean MAE (survival):   {mae_t_survival.mean():.4f}")
    print(f"  mean MAE (disease CIF): {mae_t_disease.mean():.4f}")
    print(f"  mean MAE (other CIF):   {mae_t_other.mean():.4f}")
    for t in (0, 10, 20, 30):
        if t < len(eval_times):
            print(f"  t={t}: survival MAE={mae_t_survival[t]:.4f}  "
                  f"disease MAE={mae_t_disease[t]:.4f}  other MAE={mae_t_other[t]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(eval_times, mae_t_survival, label="Survival", color="tab:orange")
    plt.plot(eval_times, mae_t_disease, label="Disease CIF", color="tab:red", linestyle="--")
    plt.plot(eval_times, mae_t_other, label="Other CIF", color="tab:gray", linestyle=":")
    plt.xlabel("Time step")
    plt.ylabel("Mean Absolute Prediction Error")
    plt.title(f"WTTE-RNN (competing risks) MAE_t on {scenario.name} validation subsample (n={len(sample_idx)})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- demo individuals (3 rows: overall survival, disease CIF, other CIF) ---
    if demo_values is None:
        demo_values = [65, 70, 75] if scenario.name == "ReLU" else [8.0, 12.5, 13.0]

    demo_cif = predict_cif_wtte(model, meta, demo_values, eval_times)  # (len(demo_values), 2, T1)
    fig, axes = plt.subplots(3, len(demo_values), figsize=(5 * len(demo_values), 11), sharex=True)
    if len(demo_values) == 1:
        axes = axes.reshape(3, 1)  # plt.subplots squeezes size-1 dims away by default

    demo_curves = {}
    for col, v in enumerate(demo_values):
        gt = ground_truth_cif(scenario, v, seed=seed, n_matched=gt_n_matched, t_k=T_K)
        sf_pred = 1.0 - demo_cif[col, 0, :] - demo_cif[col, 1, :]
        demo_curves[v] = {
            "ground_truth_survival": gt["survival"], "ground_truth_disease": gt["cif_disease"],
            "ground_truth_other": gt["cif_other"],
            "survival": sf_pred, "disease": demo_cif[col, 0, :], "other": demo_cif[col, 1, :],
        }

        axes[0, col].step(gt["eval_times"], gt["survival"], where="post", label="Ground truth", color="tab:red")
        axes[0, col].step(eval_times, sf_pred, where="post", label="WTTE-RNN", color="tab:orange")
        axes[0, col].set_title(f"{scenario.covariate_name}={v}\nOverall survival")
        axes[0, col].legend(fontsize=8)

        axes[1, col].step(gt["eval_times"], gt["cif_disease"], where="post", color="tab:red")
        axes[1, col].step(eval_times, demo_cif[col, 0, :], where="post", color="tab:orange")
        axes[1, col].set_title("Disease CIF")

        axes[2, col].step(gt["eval_times"], gt["cif_other"], where="post", color="tab:red")
        axes[2, col].step(eval_times, demo_cif[col, 1, :], where="post", color="tab:orange")
        axes[2, col].set_title("Other CIF")
        axes[2, col].set_xlabel("Time")

    axes[0, 0].set_ylabel("Survival probability")
    axes[1, 0].set_ylabel("CIF")
    axes[2, 0].set_ylabel("CIF")
    plt.suptitle(f"WTTE-RNN (competing risks) vs. ground truth -- {scenario.name} scenario")
    plt.tight_layout()
    plt.show()

    return model, meta, eval_times, mae_t_survival, mae_t_disease, mae_t_other, demo_curves


# ---------------------------------------------------------------------------
# Run for both scenarios. `wtte_model_cr_relu` / `wtte_meta_cr_relu` (and the
# Weibull equivalents) are the two names the model-comparison harness's
# auto-detection registry looks for (`model_prefix`/`meta_prefix` =
# "wtte_model_cr"/"wtte_meta_cr"); the remaining unpacked names just follow
# the base file's full-return-tuple convention (eval_times/MAE arrays/demo
# curves) so nothing from run_wtte_cr_for_scenario's return is discarded.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    wtte_model_cr_relu, wtte_meta_cr_relu, eval_times_cr_relu_wtte, mae_t_survival_cr_relu_wtte, mae_t_disease_cr_relu_wtte, mae_t_other_cr_relu_wtte, demo_cr_relu_wtte = run_wtte_cr_for_scenario(
        RELU_SCENARIO, "/content/relu_cohort_cr.npz",
    )
    wtte_model_cr_weibull, wtte_meta_cr_weibull, eval_times_cr_weibull_wtte, mae_t_survival_cr_weibull_wtte, mae_t_disease_cr_weibull_wtte, mae_t_other_cr_weibull_wtte, demo_cr_weibull_wtte = run_wtte_cr_for_scenario(
        WEIBULL_SCENARIO, "/content/weibull_cohort_cr.npz",
    )
