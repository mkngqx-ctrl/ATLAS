"""History-conditioned SFSA comparators. See SCENARIO_AND_METHODS.md for fidelity.

Independent PyTorch implementations of the published mathematical constructions;
not copies of the authors' TensorFlow/BERT source or pretrained models.
Only states observed by the supplied landmark enter a prediction.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def history_inputs(x, states, landmark, horizon=30, n_live=7):
    """Mask every future state before embedding; x is patient-by-3."""
    x=torch.as_tensor(x,dtype=torch.float32)
    device=x.device
    if x.ndim!=2 or x.shape[1]!=3 or not torch.isfinite(x).all():
        raise ValueError('Expected finite patient-by-3 covariates')
    landmark=torch.as_tensor(landmark,device=device,dtype=torch.long)
    if landmark.ndim==0:landmark=landmark.expand(len(x))
    if landmark.shape!=(len(x),) or ((landmark<0)|(landmark>horizon)).any():
        raise ValueError('Invalid landmarks')
    states=torch.as_tensor(states,device=device)
    if states.ndim!=2 or len(states)!=len(x) or (landmark>states.shape[1]).any():
        raise ValueError('Missing observed history')
    observed=torch.arange(1,horizon+1,device=device)[None,:]<=landmark[:,None]
    padded=torch.zeros((len(x),horizon),device=device,dtype=states.dtype)
    width=min(horizon,states.shape[1]);padded[:,:width]=states[:,:width]
    safe=torch.where(observed,padded,torch.zeros_like(padded))
    if not torch.isfinite(safe).all() or ((safe<0)|(safe>=n_live)|(safe!=safe.round())).any():
        raise ValueError('Observed history must contain only live states')
    safe=safe.long();onehot=F.one_hot(safe,n_live).float()*observed[:,:,None]
    D=3+n_live+1
    values=torch.zeros((len(x),horizon+1,D),device=device)
    valid=torch.arange(horizon+1,device=device)[None,:]<=landmark[:,None]
    values[:,:,:3]=x[:,None,:];values[:,1:,3:3+n_live]=onehot
    values[:,:,-1]=torch.arange(horizon+1,device=device)/horizon
    missing=torch.zeros_like(values);missing[:,0,3:3+n_live]=1.
    values*=valid[:,:,None]
    missing=torch.where(valid[:,:,None],missing,torch.ones_like(missing))
    gap=(torch.arange(horizon+1,device=device)[None,:]<landmark[:,None]).float()/horizon
    return dict(x=x,states=safe,observed=observed,landmark=landmark,
                values=values,missing=missing,gap=gap,valid=valid)


class DynamicDeepHitHistory(nn.Module):
    """Paper equations (3)-(9): past GRU, latest-conditioned attention, CS heads.

    The latest measurement bypasses the recurrent past, as in the article.
    Seven live-state indicators are binary auxiliary targets (masked BCE).
    One shared tail mass is the finite SFSA horizon adaptation.
    """
    def __init__(self, horizon=30, hidden=64, layers=1, attention_hidden=64,
                 head_hidden=64, head_layers=2, dropout=0.1, causes=2, n_live=7, n_covariates=3):
        super().__init__()
        self.config = dict(horizon=horizon, hidden=hidden, layers=layers,
                           attention_hidden=attention_hidden, head_hidden=head_hidden,
                           head_layers=head_layers, dropout=dropout, causes=causes, n_live=n_live, n_covariates=n_covariates)
        self.horizon, self.causes = horizon, causes
        self.n_live, self.n_covariates = n_live, n_covariates
        D = n_covariates + n_live + 1
        self.gru = nn.GRU(2*D+1, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0)
        self.attention = nn.Sequential(nn.Linear(hidden + 2*D, attention_hidden),
                                       nn.Tanh(), nn.Linear(attention_hidden, 1))
        self.next_state = nn.Linear(hidden, n_live)
        self.heads = nn.ModuleList()
        for _ in range(causes):
            blocks, width = [], hidden + 2*D
            for _ in range(head_layers):
                blocks += [nn.Linear(width, head_hidden), nn.ReLU(), nn.Dropout(dropout)]
                width = head_hidden
            blocks += [nn.Linear(width, horizon)]
            self.heads.append(nn.Sequential(*blocks))
        self.tail = nn.Linear(hidden + 2*D, 1)

    def forward(self, batch):
        values, missing, L = batch['values'], batch['missing'], batch['landmark']
        row = torch.arange(len(L), device=L.device)
        latest = torch.cat([values[row, L], missing[row, L]], -1)
        past = torch.cat([values[:, :-1], missing[:, :-1], batch['gap'][:, :-1, None]], -1)
        # Packed lengths exclude the latest observation; L=0 uses a masked dummy.
        packed = nn.utils.rnn.pack_padded_sequence(past, L.clamp_min(1).cpu(),
                                                   batch_first=True, enforce_sorted=False)
        packed_h, _ = self.gru(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True,
                                                     total_length=self.horizon)
        mask = torch.arange(self.horizon, device=L.device)[None, :] < L[:, None]
        context_input = torch.cat([hidden, latest[:, None, :].expand(-1, self.horizon, -1)], -1)
        scores = self.attention(context_input).squeeze(-1).masked_fill(~mask, -1e9)
        weights = scores.softmax(-1) * mask
        context = (weights[:, :, None] * hidden).sum(1)
        shared = torch.cat([context, latest], -1)
        logits = torch.cat([head(shared) for head in self.heads] + [self.tail(shared)], -1)
        log_pmf = logits.log_softmax(-1)
        return dict(log_pmf=log_pmf, next_logits=self.next_state(hidden), attention=weights)


def conditional_log_pmf(log_pmf, landmark, horizon=30, causes=1):
    """Stable equation-(5) normalization, including a positive tail category."""
    times = torch.arange(1, horizon + 1, device=log_pmf.device).repeat(causes)
    future = torch.cat([times[None, :] > landmark[:, None],
                        torch.ones((len(landmark), 1), dtype=torch.bool, device=log_pmf.device)], 1)
    surviving = log_pmf.masked_fill(~future, -torch.inf)
    centered = surviving - surviving.amax(-1, keepdim=True)
    return centered - torch.logsumexp(centered, -1, keepdim=True)


def pmf_survival(log_pmf, landmark, horizon=30, causes=1):
    cp = conditional_log_pmf(log_pmf, landmark, horizon, causes).exp()
    mass = cp[:, :-1].reshape(-1, causes, horizon).sum(1)
    # Reverse summation avoids cancellation for near-zero late survival.
    return torch.cat([mass.flip(1).cumsum(1).flip(1) + cp[:, -1:], cp[:, -1:]], 1)


def dynamic_rank(log_pmf, duration, cause, landmark, patient=None,
                 horizon=30, causes=1, sigma=0.1, reduction='per_patient'):
    """Article eq. (8): compare equal ELAPSED horizons since each latest visit.

    Exclude two prefixes of the same person. Dividing the published sum by batch
    size matches the mean NLL normalization; 'pairs' is an explicit alternative.
    """
    if sigma <= 0 or reduction not in ('per_patient', 'pairs'):
        raise ValueError('Invalid ranking configuration')
    cp = conditional_log_pmf(log_pmf, landmark, horizon, causes).exp()
    cif = F.pad(cp[:, :-1].reshape(-1, causes, horizon).cumsum(-1), (1, 0))
    residual = duration - landmark
    pair = (cause[:, None] > 0) & (residual[:, None] < residual[None, :])
    if patient is not None:
        pair &= patient[:, None] != patient[None, :]
    i, j = pair.nonzero(as_tuple=True)
    if not len(i):
        return log_pmf.sum() * 0
    k = cause[i] - 1
    first = cif[i, k, duration[i]]
    second = cif[j, k, landmark[j] + residual[i]]
    terms = torch.exp(-(first - second) / sigma)
    return terms.sum() / (len(log_pmf) if reduction == 'per_patient' else len(i))


def deephit_objective(model, output, batch, duration, cause, patient=None,
                      rank_weight=0.1, rank_sigma=0.1, auxiliary_weight=0.1,
                      l1_weight=1e-5, rank_reduction='per_patient'):
    H, K, L = model.horizon, model.causes, batch['landmark']
    cp = conditional_log_pmf(output['log_pmf'], L, H, K)
    event_idx = (cause.clamp_min(1) - 1) * H + duration - 1
    event_log = cp.gather(1, event_idx[:, None]).squeeze(1)
    times = torch.arange(1, H + 1, device=L.device).repeat(K)
    after = torch.cat([times[None, :] > duration[:, None], torch.ones_like(duration[:, None], dtype=torch.bool)], 1)
    censor_log = torch.logsumexp(cp.masked_fill(~after, -torch.inf), 1)
    nll = -torch.where(cause > 0, event_log, censor_log).mean()
    rank = dynamic_rank(output['log_pmf'], duration, cause, L, patient, H, K,
                        rank_sigma, rank_reduction)
    fields=slice(model.n_covariates,model.n_covariates+model.n_live)
    target = batch['values'][:, 1:, fields]
    seen = (1 - batch['missing'][:, 1:, fields]) * batch['valid'][:, 1:, None]
    # Only predict the next measurement already within this training prefix.
    auxiliary = (F.binary_cross_entropy_with_logits(output['next_logits'], target,
                                                    reduction='none') * seen).sum() / len(L)
    penalty = sum(p.abs().sum() for name, p in model.named_parameters() if 'weight' in name)
    total = nll + rank_weight * rank + auxiliary_weight * auxiliary + l1_weight * penalty
    return total, dict(nll=nll, ranking=rank, next_measurement=auxiliary, l1=penalty)


class FeatureAttentionBlock(nn.Module):
    """Article equations (9)-(10): SELU residual attention and feed-forward block."""
    def __init__(self, width, heads, intermediate, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(nn.Linear(width, intermediate), nn.SELU(),
                                          nn.Linear(intermediate, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, present):
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=~present,
                                     need_weights=False)
        h = F.selu(tokens + self.dropout(attended))
        h = F.selu(h + self.dropout(self.feed_forward(h)))
        return h * present[:, :, None]


class SurvTRACEHistory(nn.Module):
    """SurvTRACE article architecture with each observed SFSA step as a feature.

    Feature identity carries absolute time. This is an explicitly longitudinal
    adaptation of a baseline tabular model, not a published SurvTRACE variant.
    The article's original-embedding shortcut, SELU alignment and both auxiliary
    task heads are retained. No propensity loss: the user selected the released-code likelihood.
    """
    def __init__(self, horizon=30, width=16, heads=2, layers=3, intermediate=64,
                 head_layers=2, dropout=0.1, likelihood='pch', causes=2, n_live=7, n_covariates=3):
        super().__init__()
        if likelihood not in ('pch', 'discrete'):
            raise ValueError('likelihood must be pch or discrete')
        self.config = dict(horizon=horizon, width=width, heads=heads, layers=layers,
                           intermediate=intermediate, head_layers=head_layers,
                           dropout=dropout, likelihood=likelihood, causes=causes, n_live=n_live, n_covariates=n_covariates)
        self.horizon, self.likelihood = horizon, likelihood
        self.causes, self.n_live, self.n_covariates = causes, n_live, n_covariates
        # Distinct categorical field for each clock step (seven live categories).
        self.state_embedding = nn.Embedding(horizon * n_live, width)
        self.numeric_embedding = nn.Parameter(torch.empty(n_covariates+1, width))  # covariates and landmark
        nn.init.normal_(self.numeric_embedding, std=.02)
        self.blocks = nn.ModuleList([FeatureAttentionBlock(width, heads, intermediate, dropout)
                                     for _ in range(layers)])
        self.alignment = nn.Sequential(nn.Linear(2 * (horizon + n_covariates + 1) * width, intermediate), nn.SELU())
        def task_head(out_features):
            seq = []
            for _ in range(head_layers):
                seq += [nn.Linear(intermediate, intermediate), nn.ReLU(), nn.Dropout(dropout)]
            return nn.Sequential(*seq, nn.Linear(intermediate, out_features))
        self.hazard_heads = nn.ModuleList([task_head(horizon) for _ in range(causes)])
        self.mortality_head = task_head(1)
        self.duration_head = task_head(1)

    def forward(self, batch):
        L, x = batch['landmark'], batch['x']
        field = torch.arange(self.horizon, device=x.device) * self.n_live
        categorical = self.state_embedding(batch['states'] + field[None, :])
        numeric = torch.cat([x, L.float()[:,None]/self.horizon], 1)[:, :, None] * self.numeric_embedding
        present = torch.cat([torch.ones((len(x), self.n_covariates+1), dtype=torch.bool, device=x.device),
                             batch['observed']], 1)
        raw = torch.cat([numeric, categorical], 1) * present[:, :, None]
        encoded = raw
        for block in self.blocks:
            encoded = block(encoded, present)
        shared = self.alignment(torch.cat([encoded.flatten(1), raw.flatten(1)], 1))
        logits = torch.stack([head(shared) for head in self.hazard_heads],dim=1)
        return dict(hazard_logits=logits, hazard=F.softplus(logits),
                    mortality_logits=self.mortality_head(shared).squeeze(-1),
                    duration_prediction=self.duration_head(shared).squeeze(-1))

    def curves(self, output, landmark):
        return hazard_curves(output['hazard'],landmark)


def pmf_curves(log_pmf,landmark,horizon=30,causes=2):
    cp=conditional_log_pmf(log_pmf,landmark,horizon,causes).exp()
    cif=F.pad(cp[:,:-1].reshape(-1,causes,horizon).cumsum(-1),(1,0))
    survival=pmf_survival(log_pmf,landmark,horizon,causes)
    return torch.cat([survival[:,None,:],cif],1)


def hazard_curves(hazard,landmark):
    """Coherent cause-specific PCH -> [S,CIF_A,CIF_B] at grid endpoints.

    Per-interval mass is (1-exp(-sum a_k))*a_k/sum a_k, so no post-hoc
    clipping/renormalization and no incorrect use of 1-S_k as a joint CIF.
    """
    steps=torch.arange(1,hazard.shape[-1]+1,device=hazard.device)
    a=hazard*(steps[None,None,:]>landmark[:,None,None])
    total=a.sum(1)
    survival=torch.exp(-total.cumsum(-1))
    previous=F.pad(survival[:,:-1],(1,0),value=1.)
    mass=-torch.expm1(-total)
    fraction=a/total[:,None,:].clamp_min(torch.finfo(a.dtype).tiny)
    cif=(previous[:,None,:]*mass[:,None,:]*fraction).cumsum(-1)
    return torch.cat([F.pad(survival,(1,0),value=1.)[:,None,:],F.pad(cif,(1,0))],1)


def survtrace_objective(model,output,batch,duration,cause,patient=None,
                        mortality_weight=1.,duration_weight=1.):
    """Sum of published-code cause-specific PCH losses; no IPS weighting.

    Discrete sensitivity variant fits mutually exclusive step events exactly,
    using the same competing rates -> probabilities mapping as prediction.
    """
    a=output['hazard'];steps=torch.arange(1,model.horizon+1,device=a.device)
    seen=(steps[None,:]>batch['landmark'][:,None])&(steps[None,:]<=duration[:,None])
    total=a.sum(1); cumulative=(total*seen).sum(1)
    row=torch.arange(len(a),device=a.device); k=(cause-1).clamp_min(0)
    selected=a[row,k,duration-1].clamp_min(torch.finfo(a.dtype).tiny)
    event=cause>0
    if model.likelihood=='pch':
        record=cumulative-event*selected.log()
    else:
        at=total[row,duration-1].clamp_min(torch.finfo(a.dtype).tiny)
        log_probability=torch.log(-torch.expm1(-at))+selected.log()-at.log()
        record=cumulative-event*(at+log_probability)
    nll=record.mean()
    mortality=F.binary_cross_entropy_with_logits(output['mortality_logits'],event.float())
    duration_loss=F.mse_loss(output['duration_prediction'],duration.float()/model.horizon)
    return (nll+mortality_weight*mortality+duration_weight*duration_loss,
            dict(nll=nll,mortality=mortality,observed_duration=duration_loss))
