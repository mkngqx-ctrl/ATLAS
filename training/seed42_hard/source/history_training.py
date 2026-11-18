"""Prefix-expanded training and prediction for the history comparators.

Patients are split BEFORE prefixes are expanded. No generator/reference curves
are imported here: learning uses observed training/validation records only.
"""
from __future__ import annotations

import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from history_models import (history_inputs, DynamicDeepHitHistory, deephit_objective,
                            pmf_survival)

HISTORY_VERSION = 2


class PrefixDataset(Dataset):
    """Every eligible prefix L=0,...,min(duration-1,H-1), without copying paths."""
    def __init__(self, covariate, states, duration, event, mean, sd, horizon=30):
        x, states = np.asarray(covariate), np.asarray(states)
        duration, event = np.asarray(duration), np.asarray(event)
        n = len(x)
        if not n or states.shape != (n, horizon) or duration.shape != (n,) or event.shape != (n,):
            raise ValueError('Expected nonempty, aligned SFSA cohort arrays')
        if not np.isfinite(x).all() or not np.isfinite(duration).all() or not np.isfinite(event).all():
            raise ValueError('Nonfinite covariate or outcome')
        if np.any(duration != duration.astype(int)) or np.any((duration < 1) | (duration > horizon)):
            raise ValueError('Duration must be an integer step within the horizon')
        if not np.isin(event, [0, 1]).all():
            raise ValueError('This hard-rule integration supports a single death event')
        if sd <= 0:
            raise ValueError('Covariate scale must be positive')
        live = np.arange(horizon)[None, :] < (duration[:, None] - 1)
        if not np.isin(states[live], [0, 1, 2]).all():
            raise ValueError('An absorbing/invalid state occurs before the recorded endpoint')
        self.x = torch.tensor((x - mean) / sd, dtype=torch.float32)
        self.states = torch.tensor(states, dtype=torch.long)
        self.duration = torch.tensor(duration, dtype=torch.long)
        self.event = torch.tensor(event, dtype=torch.long)
        counts = duration.astype(int)
        self.patient = np.repeat(np.arange(n), counts)
        self.landmark = np.concatenate([np.arange(c) for c in counts])

    def __len__(self):
        return len(self.patient)

    def __getitem__(self, index):
        i = int(self.patient[index])
        return dict(x=self.x[i], states=self.states[i], duration=self.duration[i],
                    cause=self.event[i], landmark=int(self.landmark[index]), patient=i)


def on_device(record, device, horizon):
    record = {k: v.to(device) for k, v in record.items()}
    batch = history_inputs(record['x'], record['states'], record['landmark'], horizon)
    return record, batch


def model_survival(model, output, batch):
    if isinstance(model, DynamicDeepHitHistory):
        return pmf_survival(output['log_pmf'], batch['landmark'], model.horizon, model.causes)
    return model.survival(output, batch['landmark'])


def dynamic_concordance(survival, duration, cause, landmark, patient):
    """Equal-elapsed-time concordance, ties worth 1/2; single-event diagnostic.

    Calculated exactly on a fixed validation-prefix subset, excluding same-person
    pairs. This is not an IPCW estimate or the full clinical CF evaluation suite.
    """
    residual = duration - landmark
    wins, pairs = 0., 0
    for start in range(0, len(duration), 128):
        idx = np.arange(start, min(start + 128, len(duration)))
        mask = ((cause[idx, None] > 0) & (residual[idx, None] < residual[None, :])
                & (patient[idx, None] != patient[None, :]))
        ii, jj = np.nonzero(mask)
        ii = idx[ii]
        first = 1 - survival[ii, duration[ii]]
        second = 1 - survival[jj, landmark[jj] + residual[ii]]
        wins += np.sum(first > second) + .5 * np.sum(first == second)
        pairs += len(ii)
    return float(wins / pairs) if pairs else float('nan')


def train_history(covariate_train, states_train, duration_train, event_train,
                  covariate_val, states_val, duration_val, event_val,
                  architecture='deephit', model_config=None, loss_config=None,
                  lr=1e-3, batch_size=256, max_epochs=200, patience=15,
                  weight_decay=0., seed=42, device=None, verbose=True,
                  monitor=None, validation_prefix_limit=2048, burn_in_epochs=0):
    """Train without consulting test outcomes or analytical SFSA references.

    All training prefixes are used each epoch. The fixed validation subset caps
    quadratic dynamic-ranking/concordance work and is recorded in metadata.
    Hyperparameters are explicit; these defaults are not claimed to be tuned.
    """
    from history_models import SurvTRACEHistory, survtrace_objective
    if architecture not in ('deephit', 'survtrace'):
        raise ValueError('Unknown history architecture')
    if min(batch_size, max_epochs, patience, validation_prefix_limit) < 1 or burn_in_epochs < 0:
        raise ValueError('Training budgets must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = dict(model_config or {})
    loss_config = dict(loss_config or {})
    model_class = DynamicDeepHitHistory if architecture == 'deephit' else SurvTRACEHistory
    model = model_class(**model_config).to(device)
    objective = deephit_objective if architecture == 'deephit' else survtrace_objective
    monitor = monitor or ('concordance' if architecture == 'deephit' else 'loss')
    if monitor not in ('concordance', 'loss'):
        raise ValueError('monitor must be concordance or loss')
    mean, sd = float(np.mean(covariate_train)), float(np.std(covariate_train)) or 1.
    train_set = PrefixDataset(covariate_train, states_train, duration_train, event_train,
                              mean, sd, model.horizon)
    val_set = PrefixDataset(covariate_val, states_val, duration_val, event_val,
                            mean, sd, model.horizon)
    rng = np.random.default_rng(seed + 103)
    val_index = np.sort(rng.choice(len(val_set), min(len(val_set), validation_prefix_limit), replace=False))
    val_loader = DataLoader(torch.utils.data.Subset(val_set, val_index), batch_size=batch_size)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, stale, best_state, log = -float('inf'), 0, None, []
    for epoch in range(max_epochs + burn_in_epochs):
        burn_in = epoch < burn_in_epochs
        model.train()
        train_sum, count = 0., 0
        for record in loader:
            record, batch = on_device(record, device, model.horizon)
            output = model(batch)
            loss, parts = objective(model, output, batch, record['duration'], record['cause'],
                                     patient=record['patient'], **loss_config)
            if burn_in:
                if architecture != 'deephit':
                    raise ValueError('Burn-in is only implemented for Dynamic-DeepHit')
                loss = parts['next_measurement']
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            train_sum += float(loss.detach()) * len(record['x'])
            count += len(record['x'])
        model.eval()
        val_sum, val_n, collected, breakdown = 0., 0, [], {}
        with torch.no_grad():
            for record in val_loader:
                record, batch = on_device(record, device, model.horizon)
                output = model(batch)
                loss, parts = objective(model, output, batch, record['duration'], record['cause'],
                                         patient=record['patient'], **loss_config)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite validation loss')
                b = len(record['x'])
                val_sum += float(loss) * b
                val_n += b
                for k, v in parts.items():
                    breakdown[k] = breakdown.get(k, 0.) + float(v) * b
                collected.append((model_survival(model, output, batch).cpu().numpy(),
                                  *[record[k].cpu().numpy() for k in ('duration', 'cause', 'landmark', 'patient')]))
        merged = [np.concatenate([item[k] for item in collected]) for k in range(5)]
        concordance = dynamic_concordance(*merged)
        val_loss = val_sum / val_n
        score = concordance if monitor == 'concordance' and np.isfinite(concordance) else -val_loss
        row = dict(epoch=epoch + 1, burn_in=burn_in, train_loss=train_sum/count,
                   validation_loss=val_loss, validation_concordance=concordance,
                   **{'val_' + k: v / val_n for k, v in breakdown.items()})
        log.append(row)
        if verbose:
            print(f"  {architecture} epoch {epoch+1}: train={row['train_loss']:.5f} "
                  f"validation={val_loss:.5f} concordance={concordance:.4f}", flush=True)
        if burn_in:
            continue
        if score > best + 1e-6:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError('Training produced no checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(history_version=HISTORY_VERSION, architecture=architecture,
        model_config=copy.deepcopy(model.config), loss_config=loss_config,
        x_mean=mean, x_sd=sd, L=model.horizon, device=str(device), training_log=log,
        training_prefixes=len(train_set), validation_prefixes=val_n,
        validation_index_seed=seed+103, validation_prefix_limit=validation_prefix_limit,
        monitor=monitor, best_epoch=best_epoch, best_score=best, seed=seed,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        burn_in_epochs=burn_in_epochs, max_epochs=max_epochs, patience=patience)


def predict_history(model, meta, covariate, eval_times, prefix=None, landmark=0, batch_size=512):
    x = np.asarray(covariate, dtype=float)
    if x.ndim != 1:
        raise ValueError('Expected a one-dimensional covariate array')
    t = np.asarray(eval_times)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t != t.astype(int)) or np.any((t < 0) | (t > model.horizon)):
        raise ValueError('Evaluation times must be integer steps within the horizon')
    L = np.asarray(landmark)
    if L.ndim == 0:
        L = np.full(len(x), L)
    if L.shape != x.shape or not np.isfinite(L).all() or np.any(L != L.astype(int)):
        raise ValueError('Landmarks must be aligned integer times')
    if np.any(t[None, :] < L[:, None]):
        raise ValueError('Cannot request predictions before the observed landmark')
    if prefix is None:
        if np.any(L != 0):
            raise ValueError('A nonzero landmark requires its observed prefix')
        prefix = np.zeros((len(x), 0), dtype=int)
    prefix = np.asarray(prefix)
    if len(x) == 0:
        return np.empty((0, len(t)))
    curves = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = start + batch_size
            batch = history_inputs(torch.tensor((x[start:stop]-meta['x_mean'])/meta['x_sd'],
                                                 dtype=torch.float32, device=device),
                                   prefix[start:stop], L[start:stop], model.horizon)
            curves.append(model_survival(model, model(batch), batch)[:, t.astype(int)].cpu().numpy())
    return np.concatenate(curves)
